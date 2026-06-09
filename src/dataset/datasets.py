import json
import logging
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import trange
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate

from .constants import DATASET_DICT, DATASETS, MEAN, STD, FEAT_TYPES, ONLINE_FEAT_TYPES, SEMANTIC_ID_TO_IDX_DICT
from .data_utils import resize_depth, resize_flow, to_float_tensor, to_tensor, depth2xyz

logger = logging.getLogger("PerceptualModel")


class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets, dataset_names=None, per_dataset_sample_nums: dict = None):
        """
        Extended concatenated dataset class that tracks sub-dataset names and supports sample size configuration by name.

        Supports two scenarios:
        1. With `per_dataset_sample_nums`: Allocate contiguous sequence ranges based on specified sample sizes.
           Out-of-length mapping is handled by the sub-dataset's own modulo logic (enables repeated sampling).
        2. Without `per_dataset_sample_nums`: Allocate sequence ranges based on original dataset lengths (1:1 index mapping).

        Args:
            datasets: List of sub-datasets to concatenate.
            dataset_names: Optional list of names corresponding to sub-datasets.
                If None (default), sub-datasets will be named as "dataset_0", "dataset_1", ..., "dataset_N-1".
            per_dataset_sample_nums: Optional dictionary specifying sample sizes for each sub-dataset (key: dataset name, value: sample count).
                Takes precedence over default dataset lengths when provided; must contain all sub-dataset names.
        """
        super().__init__(datasets)

        if dataset_names is None:
            self.dataset_names = [f"dataset_{i}" for i in range(len(datasets))]
        else:
            assert len(datasets) == len(dataset_names)
            self.dataset_names = dataset_names
        self.name_to_idx = {name: idx for idx, name in enumerate(self.dataset_names)}

        # Original data basic information
        self.sub_real_lens = [len(ds) for ds in self.datasets]  # Original length of each sub-dataset
        self.sub_real_offsets = [0]  # Global offsets for original data
        for ds_len in self.sub_real_lens[:-1]:
            self.sub_real_offsets.append(self.sub_real_offsets[-1] + ds_len)

        self.has_specified_sample_nums = per_dataset_sample_nums is not None
        if self.has_specified_sample_nums:
            self.per_dataset_sample_nums = per_dataset_sample_nums
            missing_names = [name for name in per_dataset_sample_nums.keys() if name not in self.name_to_idx]
            assert not missing_names, f"Dataset names {missing_names} not found!"
            for name, sample_num in per_dataset_sample_nums.items():
                assert isinstance(sample_num, int) and sample_num > 0, f"Sample num for {name} must be a positive integer!"
                ds_idx = self.name_to_idx[name]
                ds_len = self.sub_real_lens[ds_idx]
                assert sample_num % ds_len == 0, \
                    f"Sample num for dataset '{name}' (={sample_num}) must be a multiple of its length (={ds_len})! " \
                    f"Current: {sample_num} % {ds_len} = {sample_num % ds_len} (not 0)."
            # Calculate sequence ranges (start_idx, end_idx) for each dataset
            self.sub_seq_ranges = {}
            current_start = 0
            for name in self.dataset_names:
                assert name in per_dataset_sample_nums
                sample_num = per_dataset_sample_nums[name]
                self.sub_seq_ranges[name] = (current_start, current_start + sample_num - 1)
                current_start += sample_num
            self.total_seq_len = current_start

        else:
            self.sub_seq_ranges = {}
            for idx, name in enumerate(self.dataset_names):
                start = self.sub_real_offsets[idx]
                end = self.sub_real_offsets[idx] + self.sub_real_lens[idx] - 1
                self.sub_seq_ranges[name] = (start, end)
            self.total_seq_len = sum(self.sub_real_lens)

    def __getitem__(self, seq_idx: int, *args, **kwargs):
        """Map global sequence index to a sample from the corresponding sub-dataset."""
        seq_idx = seq_idx % self.total_seq_len

        # Find the dataset that the sequence index belongs to
        dataset_name = None
        for name, (start, end) in self.sub_seq_ranges.items():
            if start <= seq_idx <= end:
                dataset_name = name
                break
        assert dataset_name is not None

        # Calculate local index (no modulo operation here; handled by sub-dataset itself)
        ds_idx = self.name_to_idx[dataset_name]
        if self.has_specified_sample_nums:
            seq_offset = seq_idx - self.sub_seq_ranges[dataset_name][0]
            local_idx = seq_offset  # Modulo is handled by the sub-dataset
        else:
            local_idx = seq_idx - self.sub_real_offsets[ds_idx]

        subset = self.datasets[ds_idx]
        return subset.__getitem__(local_idx, *args, **kwargs)

    def __len__(self) -> int:
        """Return total sequence length: sum of specified sample sizes (if provided) or sum of original dataset lengths."""
        return self.total_seq_len


class PerceptualModelDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        annotation_txt_file_list: Union[str, List[str]],
        target_size: Tuple[int, int] = (160, 240),
        num_context_timesteps: int = 4,
        num_target_timesteps: int = 4,
        num_max_cams: Literal[1, 3, 5, 6, 7] = 3,
        timespan: float = 2.0,  # 2.0 seconds
        subset_indices: Optional[List[int]] = None,
        num_replicas: int = 1,
        equispaced: bool = True,
        load_depth: bool = True,
        load_pseudo_depth: bool = False,
        load_flow: bool = False,
        load_dynamic_mask: bool = False,
        load_ground_label: bool = False,
        load_semantic_label: bool = False,
        return_context_as_target: bool = False,
        skip_sky_mask: bool = False,
        # load_feat_types: List[str] = [],
        feat_input_size: Tuple[int, int] = (320, 480),
        online_feat: bool = False,
        img_norm_for_online_feat: bool = False,
        only_interp: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.target_size = target_size
        self.num_context_timesteps = num_context_timesteps
        self.num_target_timesteps = num_target_timesteps
        self.num_max_cams = num_max_cams
        self.timespan = timespan
        self.load_depth = load_depth
        self.load_flow = load_flow
        self.load_pseudo_depth = load_pseudo_depth
        self.load_dynamic_mask = load_dynamic_mask
        self.load_ground_label = load_ground_label
        self.load_semantic_label = load_semantic_label
        self.skip_sky_mask = skip_sky_mask
        # self.load_feat_types = [s.strip() for s in load_feat_types]
        # assert all(s in FEAT_TYPES for s in self.load_feat_types), \
        #     f"The feat types {self.load_feat_types} should be in {FEAT_TYPES}. "
        self.online_feat = online_feat
        # if online_feat:
        #     assert all(s in ONLINE_FEAT_TYPES for s in self.load_feat_types), \
        #         f"The feat types {self.load_feat_types} should be in {ONLINE_FEAT_TYPES}. "
        if isinstance(annotation_txt_file_list, str):
            annotation_txt_file_list = [annotation_txt_file_list]

        # Read all lines from annotation_txt_file_list (scene path/JSON)
        annotation_paths = []
        for annotation_txt_file in annotation_txt_file_list:
            with open(annotation_txt_file, "r") as f:
                annotation_paths += [line.strip() for line in f.readlines() if line.strip()]

        if subset_indices is not None:
            annotation_paths = [annotation_paths[i] for i in subset_indices]

        self.annotations = []
        for annotation_path in annotation_paths:
            with open(os.path.join(data_root, annotation_path), "r") as f:
                self.annotations.append(json.load(f))
        logger.info(f"Loaded {len(self.annotations)} annotations.")

        self.num_replicas = num_replicas
        if self.num_replicas > 1:
            self.annotations *= self.num_replicas

        self.equispaced = equispaced
        self.return_context_as_target = return_context_as_target

        self.img_transformation = transforms.Compose(
            [
                transforms.Resize(target_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
                # transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
        # original size transformation for online feature extraction
        if img_norm_for_online_feat:
            img_to_extract_feat_transformation_list = [
                transforms.Resize(feat_input_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),  # Larger deviation from offline features, TODO: determine whether to do img norm based on implementation results
            ]
        else:
            img_to_extract_feat_transformation_list = [
                transforms.Resize(feat_input_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
            ]
        self.img_to_extract_feat_transformation = transforms.Compose(img_to_extract_feat_transformation_list)

        self.only_interp = only_interp

    def __len__(self) -> int:
        return len(self.annotations)

    def get_interval(self, fps):
        # The evaluation/inference strategy is determined based on the training strategy.
        num_max_future_frames = int(self.timespan * fps)
        max_idx = np.arange(
                0,
                num_max_future_frames,
                num_max_future_frames // self.num_context_timesteps,
        ).max()
        if self.equispaced and self.only_interp:
            interval = max_idx + 1
        else:
            interval = num_max_future_frames
        return interval

    def get_frame(
        self,
        scene_json: Dict[str, Any],
        frame_idx: int,
        source_frame_idx: int = -1,
    ) -> Dict[str, Any]:
        """Retrieve a single frame from the dataset."""
        normalized_intrinsics = scene_json["normalized_intrinsics"]
        dataset_name = scene_json["dataset"]
        cam_to_world = scene_json["camera_to_world"]

        images, depths, sky_masks, flows = [], [], [], []
        pseudo_depths, pseudo_depth_confs = [], []
        pts3ds, valid_masks = [], []
        camtoworlds, intrinsics = [], []
        dynamic_masks, ground_masks = [], []
        semantic_labels = []
        semantic_labels_mask = []
        images_to_extract_feat = []  # using for online feature extraction

        if source_frame_idx < 0:
            source_frame_idx = frame_idx

        camera_list = DATASET_DICT[dataset_name]["camera_list"][self.num_max_cams]
        ref_camera_name = DATASET_DICT[dataset_name]["ref_camera"]

        # world_to_canonical
        world_to_canonical = np.linalg.inv(torch.tensor(cam_to_world[ref_camera_name][source_frame_idx]))

        feats = {}  # Support simultaneous distillation of multiple features
        for camera in camera_list:
            # Get img_relative_path
            img_relative_path = scene_json["relative_image_path"][camera][frame_idx]
            # driving_sim: training/040/rgb_front/00000.jpg
            if dataset_name in ["waymo", "nuscenes", "argoverse2"]:
                img_relative_path = img_relative_path.replace("images", f"images_4")
                img_relative_path = img_relative_path.replace("sweeps", f"sweeps_4")
                img_relative_path = img_relative_path.replace("samples", f"samples_4")

            # Get RGB
            img_path = os.path.join(self.data_root, "datasets", dataset_name, img_relative_path)
            raw_img = Image.open(img_path).convert("RGB")
            img = self.img_transformation(raw_img)
            images.append(img)

            # if len(self.load_feat_types) > 0 and self.online_feat:
            if self.online_feat:
                if dataset_name == "driving_sim" or dataset_name == "b2d":
                    # driving_sim/b2d has no downsampled directory, use original RGB path directly for online feature extraction
                    img_to_extract_feat = self.img_to_extract_feat_transformation(raw_img)
                else:
                    # Get original size RGB for online feature extraction
                    img_name = img_path.split('/')[-2].strip()
                    img_to_extract_feat_path = img_path.replace(img_name, "images")
                    img_to_extract_feat = Image.open(img_to_extract_feat_path).convert("RGB")
                    img_to_extract_feat = self.img_to_extract_feat_transformation(img_to_extract_feat)
                images_to_extract_feat.append(img_to_extract_feat)

            semantic_image_np = None
            if dataset_name == "b2d" and (not self.skip_sky_mask or self.load_dynamic_mask or self.load_ground_label):
                # Load semantic_image when sky/dynamic/ground is needed
                semantic_path = img_path.replace("rgb", "semantic").replace(".jpg", ".png")
                semantic_image = Image.open(semantic_path).resize(self.target_size[::-1], resample=Image.NEAREST)
                # class ref: https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
                semantic_image_np = np.array(semantic_image)
                if len(semantic_image_np.shape) > 2:
                    semantic_image_np = semantic_image_np[..., 0]  # NOTE: Current data has two versions

            if dataset_name == "driving_sim" and (self.load_dynamic_mask or self.load_ground_label):
                # Load semantic_image when sky/dynamic/ground is needed
                semantic_path = img_path.replace("rgb", "semantic").replace(".jpg", ".png")
                semantic_image_np = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
                semantic_image_np = cv2.resize(semantic_image_np, self.target_size[::-1], interpolation=cv2.INTER_NEAREST)

            # Get sky mask
            if dataset_name in ["waymo", "nuscenes", "argoverse2", "driving_sim"]:
                if dataset_name == "nuscenes":
                    sky_path = img_path.replace("samples", "samples_sky_mask")
                    sky_path = sky_path.replace("sweeps", "sweeps_sky_mask")
                elif dataset_name == "waymo":
                    sky_path = img_path.replace("images_4", "sky_masks")
                elif dataset_name == "argoverse2":
                    sky_path = img_path.replace("images_4", "sky_masks")
                elif dataset_name == "driving_sim":
                    sky_path = img_path.replace(f"rgb_{camera}", f"sky_mask_{camera}")
                sky_path = sky_path.replace("jpg", "png")
                if self.skip_sky_mask:
                    sky = torch.zeros(self.target_size[0], self.target_size[1]).float()
                else:
                    try:
                        new_sky_path = sky_path.replace("STORM2", "STORM_masks")
                        sky = Image.open(new_sky_path).convert("L").resize(self.target_size[::-1])
                    except FileNotFoundError:
                        sky = Image.open(sky_path).convert("L").resize(self.target_size[::-1])
                sky = to_tensor(np.array(sky) > 0).float()
                sky_masks.append(sky)
            elif dataset_name == "b2d" and not self.skip_sky_mask:
                sky = to_tensor(semantic_image_np == 11).float()
                sky_masks.append(sky)

            # Get dynamic mask, this is for dynamic region evaluation.
            if self.load_dynamic_mask:
                if dataset_name == "waymo":
                    dynamic_path = img_path.replace("images_8", "dynamic_masks")
                    dynamic_path = dynamic_path.replace("images_4", "dynamic_masks")
                    dynamic_path = dynamic_path.replace("jpg", "png")
                    if not os.path.exists(dynamic_path):
                        dynamic_path = dynamic_path.replace("STORM2", "STORM")
                    dynamic_mask = Image.open(dynamic_path).convert("L").resize(self.target_size[::-1])
                    dynamic_mask = to_tensor(np.array(dynamic_mask) > 0).float()
                    dynamic_masks.append(dynamic_mask)
                elif dataset_name == "b2d":
                    dynamic = to_tensor(semantic_image_np == 21).float()  # TODO: dynamic does not include car
                    dynamic_masks.append(dynamic)
                elif dataset_name == "driving_sim":
                    dynamic = to_tensor(semantic_image_np == 24).float()
                    dynamic_masks.append(dynamic)

            # Get ground label, this is for flow evaluation, i.e., we use this to exclude the ground lidar points.
            if self.load_ground_label:
                if dataset_name == "waymo":
                    ground_path = img_path.replace("images", "ground_label")
                    ground_path = ground_path.replace("jpg", "png")
                    ground = Image.open(ground_path).convert("L").resize(self.target_size[::-1])
                    ground = to_tensor(np.array(ground) > 0).float()
                    ground_masks.append(ground)
                elif dataset_name == "b2d":
                    ground = to_tensor(semantic_image_np == 1).float()  # 1 represents road, 25 represents ground
                    ground_masks.append(ground)
                elif dataset_name == "driving_sim":
                    ground = to_tensor(semantic_image_np == 1).float()  # 1 represents road, 19 represents ground
                    ground_masks.append(ground)

            if self.load_semantic_label:
                semantic = to_tensor(np.zeros(self.target_size)).float()
                has_semantic = torch.tensor(False)
                mapping = torch.arange(29, dtype=torch.int)  # Initialize mapping table, 29 is the current maximum number of classes
                if dataset_name == "waymo":
                    semantic_path = img_path.replace("images", "semantic_segs")
                    semantic_path = semantic_path.replace("jpg", "npy")
                    if os.path.exists(semantic_path):
                        semantic = np.load(semantic_path)
                        semantic = cv2.resize(semantic, self.target_size[::-1], interpolation=cv2.INTER_NEAREST)
                        semantic = to_tensor(semantic.astype(np.float32)).float()
                        has_semantic = torch.tensor(True)
                elif dataset_name == "b2d":
                    semantic_path = img_path.replace("rgb", "semantic").replace(".jpg", ".png")
                    semantic = Image.open(semantic_path).resize(self.target_size[::-1], resample=Image.NEAREST)
                    semantic = np.array(semantic)
                    if len(semantic.shape) > 2:
                        semantic = semantic[..., 0]  # NOTE: Current data has two versions
                    semantic = to_tensor(semantic.astype(np.float32)).float()
                    has_semantic = torch.tensor(True)
                elif dataset_name == "driving_sim":
                    semantic_path = img_path.replace("rgb", "semantic").replace(".jpg", ".png")
                    semantic = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
                    semantic = cv2.resize(semantic, self.target_size[::-1], interpolation=cv2.INTER_NEAREST)
                    semantic = to_tensor(semantic.astype(np.float32)).float()
                    has_semantic = torch.tensor(True)

                # cls id remapping
                for key, value in SEMANTIC_ID_TO_IDX_DICT[dataset_name].items():
                    mapping[key] = value
                semantic = mapping[semantic.to(torch.int32)]
                semantic_labels.append(semantic)
                semantic_labels_mask.append(has_semantic)

            # cam_to_world
            camera2world = cam_to_world[camera][frame_idx]

            camtoworld = (
                DATASETS[dataset_name]["canonical_to_flu"]
                @ world_to_canonical
                @ camera2world
                @ DATASETS[dataset_name]["opencv2dataset"]
            )
            camtoworld = to_tensor(camtoworld)
            camtoworlds.append(camtoworld)

            # intrinsics
            fx, fy, cx, cy = np.array(normalized_intrinsics[camera])

            fx = fx * self.target_size[1]
            fy = fy * self.target_size[0]
            cx = cx * self.target_size[1]
            cy = cy * self.target_size[0]
            intrinsics.append(
                torch.tensor(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ]
                ).float()
            )

            if self.load_depth or self.load_flow:
                if dataset_name == "waymo":
                    depth_path = img_path.replace("images", "depth_flows").replace("jpg", "npy")
                    depth_and_flow = np.load(depth_path)
                    if self.load_depth:
                        depth = depth_and_flow[..., 0]
                        depth = torch.tensor(depth).float()
                        depth = resize_depth(depth, self.target_size)
                        depths.append(depth)
                        # pts3d and valid_mask
                        pts3d = depth2xyz(np.array(depth),
                                          fxfycxcy=np.array(normalized_intrinsics[camera]),
                                          cam2world=np.array(camtoworld),
                                          return_pixel=True)
                        pts3d = torch.tensor(pts3d).float()
                        pts3ds.append(pts3d)
                        valid_mask = depth > 0.0
                        valid_masks.append(valid_mask)
                    if self.load_flow:
                        flow = depth_and_flow[..., 1:]
                        flow = torch.tensor(flow).float()
                        flow = resize_flow(flow, self.target_size)
                        # flow definition: value is in the ego coordinate system (+x: front, +y: left, +z up), index is in pixel coordinate system (already opencv-style).
                        # there must be a better way to do this: rotate the flow to the canonical view
                        flow = (
                            flow
                            @ torch.tensor(
                                (
                                    world_to_canonical
                                    @ cam_to_world[camera][frame_idx]
                                    @ np.linalg.inv(scene_json["camera_to_ego"][camera])
                                )
                            )
                            .float()[:3, :3]
                            .T.contiguous()
                        )
                        flows.append(flow)
                if dataset_name == "nuscenes":
                    depth_path = img_path.replace("samples", "samples_depth")
                    depth_path = depth_path.replace("sweeps", "sweeps_depth")
                    depth_path = depth_path.replace("jpg", "npy")
                    depth = np.load(depth_path)
                    depth = torch.tensor(depth).float()
                    depth = resize_depth(depth, self.target_size)
                    depths.append(depth)

                if dataset_name == "argoverse2":
                    depth_path = img_path.replace("images", "depth")
                    depth_path = depth_path.replace("jpg", "npy")
                    depth = np.load(depth_path)
                    depth = torch.tensor(depth).float()
                    depth = resize_depth(depth, self.target_size)
                    depths.append(depth)

                if dataset_name == "driving_sim":
                    if self.load_depth:
                        depth_path = img_path.replace(f"rgb_{camera}", f"depth_{camera}")
                        depth_path = depth_path.replace("jpg", "png")

                        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                        depth = depth.astype(np.float32) / 100.
                        depth = torch.tensor(depth).float()
                        depth = resize_depth(depth, self.target_size)
                        depths.append(depth)
                        # pts3d and valid_mask
                        pts3d = depth2xyz(np.array(depth),
                                          fxfycxcy=np.array(normalized_intrinsics[camera]),
                                          cam2world=np.array(camtoworld),
                                          return_pixel=True)
                        pts3d = torch.tensor(pts3d).float()
                        pts3ds.append(pts3d)
                        valid_mask = (depth > 0.0) & (depth < 200)
                        valid_masks.append(valid_mask)
                    if self.load_flow:
                        # Flow for frame 0 (00000.png) does not exist, return all zeros
                        if frame_idx == 0:
                            flow = torch.zeros((self.target_size[0], self.target_size[1], 3), dtype=torch.float32)
                        else:
                            flow_path = img_path.replace(f"rgb_{camera}", f"scene_flow_{camera}")
                            flow_path = flow_path.replace("jpg", "png")

                            flow_data = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
                            flow = flow_data.astype(np.float32)
                            flow = (flow / 65535 * 2 - 1) * 5  # Unit: meters
                            flow = flow * 20  # 20Hz -> Unit: m/s
                            flow = torch.tensor(flow).float()
                            flow = resize_flow(flow, self.target_size)
                            # flow definition: value is in the ego coordinate system (+x: front, +y: left, +z up), index is in pixel coordinate system (already opencv-style).
                            # there must be a better way to do this: rotate the flow to the canonical view
                            flow = (
                                flow
                                @ torch.tensor(
                                    (
                                        world_to_canonical
                                        @ cam_to_world[camera][frame_idx]
                                        @ np.linalg.inv(scene_json["camera_to_ego"][camera])
                                    )
                                )
                                .float()[:3, :3]
                                .T.contiguous()
                            )
                        flows.append(flow)

                if dataset_name == "b2d":
                    if self.load_depth:
                        depth_path = img_path.replace("rgb", "depth").replace(".jpg", ".png")
                        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.01
                        depth[depth > 400] = 0.0
                        depth = torch.tensor(depth).float()
                        depth = resize_depth(depth, self.target_size)
                        depths.append(depth)
                        # pts3d and valid_mask
                        pts3d = depth2xyz(np.array(depth),
                                          fxfycxcy=np.array(normalized_intrinsics[camera]),
                                          cam2world=np.array(camtoworld),
                                          return_pixel=True)
                        pts3d = torch.tensor(pts3d).float()
                        pts3ds.append(pts3d)
                        valid_mask = (depth > 0.0) & (depth < 200)
                        valid_masks.append(valid_mask)
                    if self.load_flow:
                        flow = torch.zeros((self.target_size[0], self.target_size[1], 3), dtype=torch.float32)
                        flows.append(flow)

            if dataset_name == "waymo" and self.load_pseudo_depth:
                pseudo_depth_path = img_path.replace("images", "pseudo_depth").replace("jpg", "npy")
                pseudo_depth = np.load(pseudo_depth_path)
                pseudo_depth = torch.tensor(pseudo_depth).float()
                pseudo_depth = resize_depth(pseudo_depth, self.target_size)
                pseudo_depths.append(pseudo_depth)

                pseudo_depth_conf_path = img_path.replace("images", "pseudo_depth").replace(".jpg", "_conf.npy")
                pseudo_depth_conf = np.load(pseudo_depth_conf_path)
                pseudo_depth_conf = torch.tensor(pseudo_depth_conf).float()
                pseudo_depth_conf = resize_depth(pseudo_depth_conf, self.target_size)
                pseudo_depth_confs.append(pseudo_depth_conf)

            # Remove offline features
            # if not self.online_feat:
            #     for feat_type in self.load_feat_types:
            #         assert dataset_name == "waymo", "Only waymo can load feature currently. "
            #         if feat_type not in feats:
            #             feats[feat_type] = []
            #         feat_path = img_path.replace(img_name, f"features/{feat_type}").replace("jpg", "npz")
            #         feats[feat_type].append(torch.tensor(np.load(feat_path)['x'])) # c, h, w

        frame_images = torch.stack(images)
        frame_depths = torch.stack(depths) if len(depths) > 0 else None
        frame_pts3ds = torch.stack(pts3ds) if len(pts3ds) > 0 else None
        frame_valid_masks = torch.stack(valid_masks) if len(valid_masks) > 0 else None
        frame_sky_masks = torch.stack(sky_masks) if len(sky_masks) > 0 else None
        frame_flows = torch.stack(flows) if len(flows) > 0 else None
        frame_dynamic_masks = torch.stack(dynamic_masks) if len(dynamic_masks) > 0 else None
        frame_camtoworlds = torch.stack(camtoworlds)
        frame_intrinsics = torch.stack(intrinsics)
        ground_masks = torch.stack(ground_masks) if len(ground_masks) > 0 else None
        semantic_labels = torch.stack(semantic_labels) if len(semantic_labels) > 0 else None
        semantic_labels_mask = torch.stack(semantic_labels_mask) if len(semantic_labels_mask) > 0 else None
        frame_pseudo_depths = torch.stack(pseudo_depths) if len(pseudo_depths) > 0 else None
        frame_pseudo_depth_confs = torch.stack(pseudo_depth_confs) if len(pseudo_depth_confs) > 0 else None
        data_dict = {
            "image": frame_images,
            "camtoworld": frame_camtoworlds,
            "intrinsics": frame_intrinsics,
            "frame_idx": frame_idx,
            "depth": frame_depths,
            "pts3d": frame_pts3ds,
            "valid_masks": frame_valid_masks,
            "sky_masks": frame_sky_masks,
            "flow": frame_flows,
            "dynamic_masks": frame_dynamic_masks,
            "ground_masks": ground_masks,
            "semantic_labels": semantic_labels,
            "semantic_labels_mask": semantic_labels_mask,
            "pseudo_depth": frame_pseudo_depths,
            "pseudo_depth_conf": frame_pseudo_depth_confs
        }

        # if self.online_feat and len(self.load_feat_types) > 0:  # online features extraction
        if self.online_feat:  # online features extraction
            frame_images_to_extract_feat = torch.stack(images_to_extract_feat)
            data_dict["frame_images_to_extract_feat"] = frame_images_to_extract_feat
        # Remove offline features
        # else:  # load offline features
        #     for feat_idx, feat_type in enumerate(self.load_feat_types):
        #         frame_feats = torch.stack(feats[feat_type])
        #         data_dict[f"feat_{feat_idx}"] = frame_feats

        return {k: v for k, v in data_dict.items() if v is not None}

    def get_segment(
        self, index: int, context_frame_idx: int = -1, return_all=False
    ) -> Dict[str, Any]:
        scene_json = self.annotations[index % len(self.annotations)]
        scene_id = scene_json["scene_id"]
        num_timesteps = scene_json["num_timesteps"]
        fps = scene_json["fps"]
        num_max_future_frames = int(self.timespan * fps)
        if num_max_future_frames > num_timesteps:
            num_max_future_frames = int(fps)  # make it 1 seconds
        time_in_seconds = scene_json["normalized_time"]
        interval = self.get_interval(fps)
        if context_frame_idx < 0 or context_frame_idx + interval > num_timesteps:
            context_frame_idx = np.random.randint(0, num_timesteps - interval + 1)
        assert (
            context_frame_idx + interval <= num_timesteps
        ), f"scene_id: {scene_id}, context_frame_idx: {context_frame_idx}, num_timesteps: {num_timesteps}, num_max_future_frames: {num_max_future_frames}"
        # print('context_frame_idx', context_frame_idx)

        if self.equispaced:
            context_frame_idxs = np.arange(
                context_frame_idx,
                context_frame_idx + num_max_future_frames,
                num_max_future_frames // self.num_context_timesteps,
            )
        else:
            context_frame_idxs = np.random.choice(
                np.arange(
                    context_frame_idx,
                    context_frame_idx + num_max_future_frames,
                ),
                size=self.num_context_timesteps,
                replace=False,
            )
            context_frame_idxs = sorted(context_frame_idxs)

        if self.only_interp:
            # only interpolated frames
            target_frame_idxs = np.arange(
                context_frame_idxs[0],
                context_frame_idxs[-1] + 1,
            )
        else:
            # extrapolate a few frames
            # return all frames between context_frame_idx and context_frame_idx + num_max_future_frames
            target_frame_idxs = np.arange(
                context_frame_idx,
                context_frame_idx + num_max_future_frames,
            )

        if not return_all:
            target_frame_idxs = np.random.choice(target_frame_idxs, self.num_target_timesteps, replace=False)
        target_frame_idxs = [min(idx, num_timesteps - 1) for idx in target_frame_idxs]
        target_frame_idxs = sorted(target_frame_idxs)

        # print('context_frame_idxs', context_frame_idxs)
        # print(' target_frame_idxs',  target_frame_idxs)
        # print(' scene_name', scene_json["scene_name"])

        # get context
        context_dict_list = []
        for ctx_id in context_frame_idxs:
            context_dict = self.get_frame(
                scene_json=scene_json,
                frame_idx=ctx_id,
                source_frame_idx=context_frame_idxs[0],
            )
            context_dict["time"] = torch.tensor(
                [time_in_seconds[ctx_id] - time_in_seconds[context_frame_idxs[0]]]
                * self.num_max_cams
            )
            context_dict_list.append(context_dict)
        if self.return_context_as_target:
            target_frame_idxs = context_frame_idxs
        target_dict_list = []
        for target_id in target_frame_idxs:
            target_dict = self.get_frame(
                scene_json=scene_json,
                frame_idx=target_id,
                source_frame_idx=context_frame_idxs[0],
            )
            target_dict["time"] = torch.tensor(
                [time_in_seconds[target_id] - time_in_seconds[context_frame_idxs[0]]]
                * self.num_max_cams
            )
            target_dict_list.append(target_dict)
        context_dict = default_collate(context_dict_list)
        target_dict = default_collate(target_dict_list)

        for k, v in context_dict.items():
            if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                context_dict[k] = torch.cat([d for d in v], dim=0)
        for k, v in target_dict.items():
            if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                target_dict[k] = torch.cat([d for d in v], dim=0)
        sample = {
            "context": context_dict,
            "target": target_dict,
            "dataset_name": scene_json['dataset'],
            "scene_id": scene_id,
            "scene_name": scene_json["scene_name"],
            "width": self.target_size[1],
            "height": self.target_size[0],
            "fps": fps,
            "timespan": self.timespan,
            "num_max_cams": self.num_max_cams,
        }
        return to_float_tensor(sample)

    def __getitem__(
        self, index: int, context_frame_idx: int = -1, return_all=False
    ) -> Dict[str, Any]:
        try:
            return self.get_segment(index, context_frame_idx, return_all)
        except Exception as e:
            scene_json = self.annotations[index % len(self.annotations)]
            scene_id = scene_json["scene_id"]
            logger.info(
                f"Error in scene_id: {scene_id}, context_frame_idx: {context_frame_idx}, scene_name: {scene_json['scene_name']}"
            )
            logger.info(e)
            try:
                return self.get_segment(index + 1, 0, return_all=return_all)
            except Exception as e:
                logger.info(e)
                return self.get_segment(index + 1, return_all=return_all)

    def get_one_scene(
        self, index: int, start_index=0, end_index=60, time_step: int = 5, return_all=True
    ) -> Dict[str, Any]:
        try:
            scene_json = self.annotations[index % len(self.annotations)]
            scene_id = scene_json["scene_id"]
            num_timesteps = scene_json["num_timesteps"]
            fps = scene_json["fps"]
            time_in_seconds = scene_json["normalized_time"]
            end_index = min(end_index, num_timesteps - 1)

            context_frame_idxs = np.arange(start_index, end_index+1, time_step)
            if return_all:
                # return all frames between context_frame_idx and context_frame_idx + num_max_future_frames
                target_frame_idxs = np.arange(start_index, end_index+1) #   np.arange(0, 2)
            else:
                # randomly sample "num_target_timesteps" frames
                target_frame_idxs = np.random.choice(
                    np.arange(
                        context_frame_idxs[0],
                        context_frame_idxs[0] + int(fps),
                    ),
                    self.num_target_timesteps,
                    replace=False,
                )
            target_frame_idxs = sorted(target_frame_idxs)

            # get context
            context_dict_list = []
            for ctx_id in context_frame_idxs:
                context_dict = self.get_frame(
                    scene_json=scene_json,
                    frame_idx=ctx_id,
                    source_frame_idx=context_frame_idxs[0],
                )
                context_dict["time"] = torch.tensor(
                    [time_in_seconds[ctx_id] - time_in_seconds[context_frame_idxs[0]]]
                    * self.num_max_cams
                )
                context_dict_list.append(context_dict)
            if self.return_context_as_target:
                target_frame_idxs = context_frame_idxs
            target_dict_list = []
            for target_id in target_frame_idxs:
                target_dict = self.get_frame(
                    scene_json=scene_json,
                    frame_idx=target_id,
                    source_frame_idx=context_frame_idxs[0],
                )
                target_dict["time"] = torch.tensor(
                    [time_in_seconds[target_id] - time_in_seconds[context_frame_idxs[0]]]
                    * self.num_max_cams
                )
                target_dict_list.append(target_dict)
            context_dict = default_collate(context_dict_list)
            target_dict = default_collate(target_dict_list)

            for k, v in context_dict.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    context_dict[k] = torch.cat([d for d in v], dim=0)
            for k, v in target_dict.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    target_dict[k] = torch.cat([d for d in v], dim=0)
            sample = {
                "context": context_dict,
                "target": target_dict,
                "scene_id": scene_id,
                "scene_name": scene_json["scene_name"],
                "width": self.target_size[1],
                "height": self.target_size[0],
                "fps": fps,
                "timespan": self.timespan,
                "num_max_cams": self.num_max_cams,
            }
            return to_float_tensor(sample)
        except Exception as e:
            logger.info(
                f"Error in scene_id: {scene_id}, context_frame_idx: {context_frame_idxs}, scene_name: {scene_json['scene_name']}"
            )
            logger.info(e)


class PerceptualModelDatasetEval(PerceptualModelDataset):
    def __init__(
        self,
        scene_id_list: Optional[List[int]] = None,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)

        if scene_id_list is None:
            scene_id_list = list(range(len(self.annotations)))
        val_sample_list = []
        for scene_id in scene_id_list:
            interval = self.get_interval(self.annotations[scene_id]['fps'])
            for start_id in range(0, self.annotations[scene_id]['num_timesteps'], interval):
                if scene_id == 63 and start_id == 0:
                    continue
                # Handle the last segment in the scene that does not meet the interval (this has already been inferred in the previous few frames).
                if start_id + interval > self.annotations[scene_id]['num_timesteps']:
                    start_id = self.annotations[scene_id]['num_timesteps'] - interval
                val_sample_list.append((scene_id, start_id))
        self.val_sample_list = val_sample_list

    def __len__(self) -> int:
        return len(self.val_sample_list)

    def __getitem__(self, index: int):
        return super(PerceptualModelDatasetEval, self).__getitem__(
            self.val_sample_list[index][0],
            self.val_sample_list[index][1],
            return_all=True,
        )


class SingleSequenceDataset(PerceptualModelDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        scene_json = self.annotations[index % len(self.annotations)]
        num_timesteps = scene_json["num_timesteps"]
        if end_index < 0 or end_index > num_timesteps:
            end_index = num_timesteps
        segment_data = []
        cam_to_world = scene_json["camera_to_world"]
        ref_camera_name = DATASET_DICT[scene_json["dataset"]]["ref_camera"]

        # world_to_canonical
        world_to_canonical = np.linalg.inv(torch.tensor(cam_to_world[ref_camera_name][start_index]))  # dtype('float32')

        clip_start_id = None
        interval = self.get_interval(scene_json['fps'])
        for start_id in trange(start_index, end_index, interval):
            # Handle the last segment in the scene that does not meet the interval (this has already been inferred in the previous few frames).
            if start_id + interval > num_timesteps:
                clip_start_id = start_id
                start_id = num_timesteps - interval
                print(f' ---- Total num_timesteps is {num_timesteps}, the last segment start at {start_id}, will clip at {clip_start_id}. ---- ')

            segment_dict = self.get_segment(
                index=index, context_frame_idx=start_id, return_all=True
            )
            # compute the relative transformation from a start_id to the first frame
            current_world = cam_to_world[ref_camera_name][start_id]

            # canonical (flu)
            segment_to_ref = (DATASETS[scene_json['dataset']]["canonical_to_flu"] @ world_to_canonical) @ \
                                (current_world @ np.linalg.inv(DATASETS[scene_json['dataset']]["canonical_to_flu"]))
            # worlds (flu)
            # segment_to_ref = DATASETS[scene_json['dataset']]["canonical_to_flu"] @ \
            #                     current_world @ np.linalg.inv(DATASETS[scene_json['dataset']]["canonical_to_flu"])

            # worlds (opencv)
            # opencv2waymo = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            # segment_to_ref = np.linalg.inv(opencv2waymo) @ DATASETS[scene_json['dataset']]["canonical_to_flu"] @ \
            #                     current_world @ np.linalg.inv(DATASETS[scene_json['dataset']]["canonical_to_flu"])
            # print('segment_to_ref: Point cloud in world coordinate system!')

            segment_dict["segment_to_ref"] = to_float_tensor(segment_to_ref)
            if clip_start_id is not None:
                segment_dict["relative_clip_start_id"] = clip_start_id - start_id
            segment_data.append(segment_dict)
        return segment_data

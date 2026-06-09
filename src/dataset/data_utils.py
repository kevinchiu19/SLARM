import os
import json
from typing import List, Tuple, Union

import numpy as np
import einops
import torch
import torch.nn.functional as F
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from src.dataset.constants import SEMANTIC_LABEL_LIST
if os.getenv("FEAT_DIST"):
    from tools.feats_tools import get_text_label_feats


def depth2xyz(depth, fxfycxcy, cam2world=None, return_pixel=False):
    h, w = depth.shape

    '''
    focal_length_px, orig_W=1920, orig_H=1280
    fxfycxcy = np.array(
        [focal_length_px / orig_W,
         focal_length_px / orig_H,
         0.5,
         0.5,]
        )  # intrinsic = np.loadtxt('/home/share/qiuzhicheng/S3Gaussian/data/waymo/processed_flow_seg/training/114/intrinsics/0.txt')
    '''

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = (x + 0.5) / w
    y = (y + 0.5) / h
    x = (x - fxfycxcy[2:3]) / fxfycxcy[0:1]
    y = (y - fxfycxcy[3:4]) / fxfycxcy[1:2]
    z = np.ones_like(x)
    ray_d = np.stack([x, y, z], axis=2)  # [b*v, h*w, 3]

    # debug
    # cam2world = np.eye(4)
    if cam2world is not None:
        ray_d = np.matmul(ray_d, cam2world[:3, :3].T)  # [b*v, h*w, 3]  a @ b.T
        ray_o = cam2world[:3, 3]

    # ray_d = ray_d / np.linalg.norm(ray_d, axis=2, keepdims=True)  # Convert to Ray coordinate system

    xyz = ray_d * depth[..., None]

    if cam2world is not None:
        xyz = ray_o + ray_d * depth[..., None]

    if return_pixel:
        return xyz
    else:
        return xyz.reshape(-1, 3)


def xyz2depth(points, xyz2img_rt, H, W, rgb=None):
    """
    cam_intrinsic = torch.eye(4)
    cam_intrinsic[:3, :3] = Ks_batched[0, 0]
    xyz2img_rt = (cam_intrinsic @ camtoworlds_batched.float()[0, 0].inverse().to(cam_intrinsic.device))
    depth_map = xyz2depth(means_batched[0].to(cam_intrinsic.device), xyz2img_rt, H=tgt_h, W=tgt_w)
    """
    pts_4d = torch.concat([points[:, :3], torch.ones((points.shape[0], 1))], dim=-1)
    pts_2d = pts_4d @ xyz2img_rt.T

    # NOTE: No need for 4 dimensions after matrix computation
    pts_2d = pts_2d[:, :3]

    # select points in front of the camera
    if rgb is not None:
        rgb = rgb[pts_2d[:, 2]>0]
    pts_2d = pts_2d[pts_2d[:, 2]>0]

    # normalize pixel points : (u,v,1)
    img_pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    # filter out points outside the image
    fov_inds = ((img_pts_2d[:, 0] < W)
                & (img_pts_2d[:, 0] >= 0)
                & (img_pts_2d[:, 1] < H)
                & (img_pts_2d[:, 1] >= 0))
    imgfov_pts_2d = img_pts_2d[fov_inds]
    pts_2d = pts_2d[fov_inds]
    if rgb is not None:
        rgb = rgb[fov_inds]
    # compute depth map
    depth_map = torch.zeros((H, W))
    depth_map[imgfov_pts_2d[:, 1].to(torch.int32), imgfov_pts_2d[:, 0].to(torch.int32)] = pts_2d[:, 2]

    if rgb is not None:
        new_rgb = torch.zeros((H, W, 3))
        new_rgb[imgfov_pts_2d[:, 1].to(torch.int32), imgfov_pts_2d[:, 0].to(torch.int32)] = rgb
        return depth_map, new_rgb
    else:
        return depth_map


def forward_pose(pose: torch.Tensor, transform: torch.Tensor, inv=False) -> torch.Tensor:
    rotation = pose[..., :3, :3]
    translation = pose[..., :3, 3]
    if inv:
        return (torch.transpose(rotation, -1, -2) * (transform - translation).unsqueeze(-2)).sum(-1)
    else:
        return (rotation * transform.unsqueeze(-2)).sum(-1) + translation


def get_path_front_left_lift_then_spiral_forward(
    pose_ref: np.ndarray,  # [N0,4,4], Original ego pose; will generate trajectories along this
    num_frames: int,  # Number of frames / waypoints to generate
    duration_frames: int = 100,  # Number of frames per round / cycle
    # Spiral configs
    up_max: float = 0.3,
    up_min: float = -0.3,
    left_max: float = 2.0,
    left_min: float = -2.0,
    elongation: float = 1.0,
    first_with_forward=False,
    # Frontal direction vector of OpenCV
    forward_vec: np.ndarray = np.array([0.0, 0.0, 1.0]),
    # Frontal direction vector of OpenCV
    up_vec: np.ndarray = np.array([0.0, -1.0, 0.0]),
    # Frontal direction vector of OpenCV
    left_vec: np.ndarray = np.array([-1.0, 0.0, 0.0]),
) -> np.ndarray:
    """
    First lift ego in the front left direction, then do spiral forward
    """
    pose_ref = torch.tensor(pose_ref, dtype=torch.float32, device="cpu")
    n_rots = num_frames / duration_frames
    track_ref = pose_ref[..., :3, 3]
    rot_ref = pose_ref[..., :3, :3]
    # NOTE: Convert from observer's coords dir to world coords dir;
    #       If the pose_ref is the pose of camera, `forward/up/left_vec` can be used as-is;
    #           (since the cameras are already OpenCV cameras if dataio's xxx_dataset.py is correctly implemented)
    #       otherwise, you should specify the coords dir vector of your given pose_ref.
    forward_vecs_ref = (
        forward_pose(pose_ref, torch.tensor(forward_vec, dtype=torch.float32)) - track_ref
    )
    up_vecs_ref = forward_pose(pose_ref, torch.tensor(up_vec, dtype=torch.float32)) - track_ref
    left_vecs_ref = forward_pose(pose_ref, torch.tensor(left_vec, dtype=torch.float32)) - track_ref

    forward_vecs_ref = forward_vecs_ref.numpy()
    up_vecs_ref = up_vecs_ref.numpy()
    left_vecs_ref = left_vecs_ref.numpy()
    track_ref = track_ref.numpy()
    rot_ref = rot_ref.numpy()

    nvs_seqs = []

    verti_radius = (up_max - up_min) / 2.0
    up_offset = (up_max + up_min) / 2.0
    horiz_radius = (left_max - left_min) / 2.0
    left_offset = (left_max + left_min) / 2.0
    assert (verti_radius >= -1e-5) and (horiz_radius >= -1e-5)

    # ----------------------------------------
    # ---- First: lift up & left
    # ----------------------------------------
    first_frames = int(0.1 * duration_frames) + 1
    remain_frames = num_frames - first_frames
    pace = np.linalg.norm(track_ref[0] - track_ref[-1]) * elongation
    forward_1st = ((first_frames / num_frames) * pace) if first_with_forward else 0

    track = (
        track_ref[0]
        + np.linspace(0, verti_radius + up_offset, first_frames)[..., None] * up_vecs_ref[0]
        + np.linspace(0, horiz_radius + left_offset, first_frames)[..., None] * left_vecs_ref[0]
        + np.linspace(0, forward_1st, first_frames)[..., None] * forward_vecs_ref[0]
    )
    pose = np.eye(4)[None, ...].repeat(first_frames, 0)
    pose[:, :3, 3] = track
    pose[:, :3, :3] = rot_ref[0]
    nvs_seqs.append(pose)

    # ----------------------------------------
    # ---- Then:   Sprial forward
    # ----------------------------------------
    w = np.linspace(0, 1, remain_frames)  # [0->1], data key time
    t = np.arange(len(track_ref)) / (
        len(track_ref) - 1
    )  # [0->1], render key time (could be extended)
    track_interp = interpolate.interp1d(t, track_ref, axis=0, fill_value="extrapolate")
    up_vec_interp = interpolate.interp1d(t, up_vecs_ref, axis=0, fill_value="extrapolate")
    left_vec_interp = interpolate.interp1d(t, left_vecs_ref, axis=0, fill_value="extrapolate")

    up_vecs_all = up_vec_interp(w * elongation)
    left_vecs_all = left_vec_interp(w * elongation)
    # ---- Base: left * [1], up * [1]
    track_base_all = (
        track_interp(w * elongation)
        + (up_offset + verti_radius) * up_vecs_all
        + (left_offset + horiz_radius) * left_vecs_all
        + forward_1st * forward_vecs_ref[None, 0]
    )

    # up_vecs_all = np.percentile(up_vecs_ref, w*100, 0)
    # left_vecs_all = np.percentile(left_vecs_ref, w*100, 0)
    # track_base_all = np.percentile(track_ref, w*100, 0) + (up_max+up_offset) * up_vecs_all + (left_max+left_offset) * left_vecs_all

    key_rots = R.from_matrix(rot_ref)
    rot_slerp = Slerp(t, key_rots)
    if elongation > 1:
        mask = (w * elongation) < 1.0
        rot_base_all = np.eye(3)[None, ...].repeat(remain_frames, 0)
        rot_base_all[mask] = rot_slerp(w[mask]).as_matrix()
        rot_base_all[~mask] = rot_ref[-1]
    else:
        rot_base_all = rot_slerp(w).as_matrix()

    rads = np.linspace(0, remain_frames / duration_frames * np.pi * 2.0, remain_frames)
    # ---- Spiral:
    # left: [0, -1, -2, -1, 0] + base: [1] -> [1, 0, -1, 0, 1]
    # up: [0, 1, 0, -1, 0] + base [1] -> [1, 2, 1, 0, 1]
    track = (
        track_base_all
        + (np.cos(rads) - 1)[..., None] * horiz_radius * left_vecs_all
        + (np.sin(rads))[..., None] * verti_radius * up_vecs_all
    )
    pose = np.eye(4)[None, ...].repeat(remain_frames, 0)
    pose[:, :3, 3] = track
    pose[:, :3, :3] = rot_base_all
    nvs_seqs.append(pose)

    render_pose_all = np.concatenate(nvs_seqs, 0)

    return render_pose_all


def to_tensor(x: Union[np.ndarray, List, Tuple]) -> torch.Tensor:
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    return x


def to_float_tensor(d):
    if isinstance(d, dict):
        return {k: to_float_tensor(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_float_tensor(v) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.float()
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d).float()
    else:
        return d


def to_batch_tensor(d):
    if isinstance(d, dict):
        return {k: to_batch_tensor(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_batch_tensor(v) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.unsqueeze(0)
    else:
        return d


def resize_depth(depth, target_size):
    height, width = depth.shape[-2:]
    if (height, width) == target_size:
        return depth
    if len(depth.shape) == 2:
        depth = depth[None, None, ...]
    elif len(depth.shape) == 3:
        depth = depth[None, ...]
    target_height, target_width = target_size
    kernel_size_h = height // target_height
    kernel_size_w = width // target_width

    if kernel_size_h > 0 and kernel_size_w > 0:
        depth = F.max_pool2d(
            depth,
            kernel_size=(kernel_size_h, kernel_size_w),
        )
    depth = F.interpolate(depth, size=target_size, mode="nearest")
    return depth.squeeze()


def resize_flow(flow, target_size):
    height, width = flow.shape[-3:-1]
    if (height, width) == target_size:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None, ...]
    target_height, target_width = target_size
    kernel_size_h = height // target_height
    kernel_size_w = width // target_width
    # flows have direction, so we can't just use max_pool2d.
    # otherwise the direction will be wrong, e.g., max_pool2d([0, -1]) = [0]
    # previous_valid_flow = (torch.norm(flow, p=2, dim=-1) > 0.5).float().sum()
    flow[torch.norm(flow, p=2, dim=-1) < 0.5] = -100000
    if kernel_size_h > 0 and kernel_size_w > 0:
        flow = F.max_pool2d(
            flow.permute(0, 3, 1, 2),
            kernel_size=(kernel_size_h, kernel_size_w),
        )
        flow = F.interpolate(flow, size=target_size, mode="nearest")
    else:
        flow = F.interpolate(flow.permute(0, 3, 1, 2), size=target_size, mode="nearest")
    flow = flow.permute(0, 2, 3, 1)
    flow[torch.norm(flow, p=2, dim=-1) > 1000] = 0
    # new_valid_flow = (torch.norm(flow, p=2, dim=-1) > 0.5).float().sum()
    return flow.squeeze()


def prepare_inputs_and_targets(data_dict, device=torch.device("cuda"), v=3, timespan=2.0, feat_extractor=None, is_vis=False):
    assert data_dict["context"]["image"].dim() == 5, "need to be b, tv, c, h, w"
    b, tv, c, h, w = data_dict["context"]["image"].shape
    context_t = tv // v
    target_t = data_dict["target"]["image"].shape[1] // v
    input_dict = {
        "context_image": data_dict["context"]["image"].reshape(b, context_t, v, c, h, w),
        # targets to render
        "context_camtoworlds": data_dict["context"]["camtoworld"].reshape(b, context_t, v, 4, 4),
        "context_intrinsics": data_dict["context"]["intrinsics"].reshape(b, context_t, v, 3, 3),
        "target_camtoworlds": data_dict["target"]["camtoworld"].reshape(b, target_t, v, 4, 4),
        "target_intrinsics": data_dict["target"]["intrinsics"].reshape(b, target_t, v, 3, 3),
    }
    if "depth" in data_dict["context"]:
        depth_h, depth_w = data_dict["context"]["depth"].shape[-2:]
        input_dict["context_depth"] = data_dict["context"]["depth"].reshape(
            b, context_t, v, depth_h, depth_w
        )
    if "pts3d" in data_dict["context"]:
        pts3d_h, pts3d_w = data_dict["context"]["pts3d"].shape[-3:-1]
        input_dict['context_pts3d'] = data_dict["context"]['pts3d'].reshape(
            b, context_t, v, pts3d_h, pts3d_w, 3
        )
    if "valid_masks" in data_dict["context"]:
        valid_masks_h, valid_masks_w = data_dict["context"]["valid_masks"].shape[-2:]
        input_dict['context_valid_masks'] = data_dict["context"]['valid_masks'].reshape(
            b, context_t, v, valid_masks_h, valid_masks_w
        )
    if "flow" in data_dict["context"]:
        flow_h, flow_w = data_dict["context"]["flow"].shape[-3:-1]
        input_dict["context_flow"] = data_dict["context"]["flow"].reshape(
            b, context_t, v, flow_h, flow_w, 3
        )
    if "time" in data_dict["context"]:
        input_dict["context_time"] = (
            data_dict["context"]["time"].reshape(b, context_t, v) / timespan
        )
    if "time" in data_dict["target"]:
        input_dict["target_time"] = data_dict["target"]["time"].reshape(b, target_t, v) / timespan
    if "sky_masks" in data_dict["context"]:
        input_dict["context_sky_masks"] = data_dict["context"]["sky_masks"].reshape(
            b, context_t, v, h, w
        )
    if "pseudo_depth" in data_dict["context"]:
        pseudo_depth_h, pseudo_depth_w = data_dict["context"]["pseudo_depth"].shape[-2:]
        input_dict["context_pseudo_depth"] = data_dict["context"]["pseudo_depth"].reshape(
            b, context_t, v, pseudo_depth_h, pseudo_depth_w
        )
    if "pseudo_depth_conf" in data_dict["context"]:
        pseudo_depth_conf_h, pseudo_depth_conf_w = data_dict["context"]["pseudo_depth_conf"].shape[-2:]
        input_dict['context_pseudo_depth_conf'] = data_dict["context"]['pseudo_depth_conf'].reshape(
            b, context_t, v, pseudo_depth_conf_h, pseudo_depth_conf_w
        )
    if os.getenv("CONTEXT_FEAT") and "semantic_labels" in data_dict["context"]:
        context_semantic_labels = data_dict["context"]["semantic_labels"].reshape(b, context_t, v, h, w).to(torch.int32)
        context_semantic_labels_mask = data_dict["context"]["semantic_labels_mask"].reshape(b, context_t, v)
        if context_semantic_labels.max() > 0:  # Filtering none semantic label data
            input_dict["context_semantic_labels"] = context_semantic_labels
            input_dict["context_semantic_labels_mask"] = context_semantic_labels_mask
        if is_vis:
            # for visualizing
            semantic_feats = torch.zeros(context_semantic_labels.shape + (get_text_label_feats(SEMANTIC_LABEL_LIST).shape[-1],))
            semantic_feats[:] = get_text_label_feats(SEMANTIC_LABEL_LIST)[context_semantic_labels]
            semantic_feats = einops.rearrange(semantic_feats, 'b t v h w c -> b t v c h w', b=b, v=v)
            input_dict["context_feat"] = semantic_feats  # this will be covered by args.online_feat
    # support multiple offline features
    # context feature
    # feat_idx  = 0
    # while True:
    #     if f"feat_{feat_idx}" in data_dict["context"]:
    #         feat = data_dict["context"][f"feat_{feat_idx}"]
    #         feat_dim, feat_h, feat_w = feat.shape[-3:]
    #         input_dict[f"context_feat_{feat_idx}"] = feat.reshape(b, context_t, v, feat_dim, feat_h, feat_w)
    #         feat_idx += 1
    #     else:
    #         break
    # only lseg feature supports online feature currently
    # hard code using 'context_feat'
    if os.getenv("CONTEXT_FEAT") and "frame_images_to_extract_feat" in data_dict["context"]:
        assert feat_extractor is not None
        images_to_extract_feat = data_dict["context"]["frame_images_to_extract_feat"]
        images_to_extract_feat = einops.rearrange(images_to_extract_feat, 'b tv c h w -> (b tv) c h w')
        with torch.no_grad():
            feat = feat_extractor.extract_lseg_feat_by_chunk(images_to_extract_feat.to(device), (h, w))
            # feat = feat_extractor.extract_lseg_feat_one_by_one(images_to_extract_feat.to(device), (h, w))
        feat = einops.rearrange(feat, '(b t v) c h w -> b t v c h w', b=b, v=v)
        input_dict["context_feat"] = feat

    target_dict = {
        "target_image": data_dict["target"]["image"].reshape(b, target_t, v, 3, h, w),
    }

    if "depth" in data_dict["target"]:
        target_dict["target_depth"] = data_dict["target"]["depth"].reshape(
            b, target_t, v, depth_h, depth_w
        )
    if "flow" in data_dict["target"]:
        target_dict["target_flow"] = data_dict["target"]["flow"].reshape(
            b, target_t, v, depth_h, depth_w, 3
        )
        target_dict["context_flow"] = data_dict["context"]["flow"].reshape(b, context_t, v, h, w, 3)
    if "flow" in data_dict["context"]:
        target_dict["context_flow"] = data_dict["context"]["flow"].reshape(
            b, context_t, v, depth_h, depth_w, 3
        )
    if "sky_masks" in data_dict["target"]:
        target_dict["target_sky_masks"] = data_dict["target"]["sky_masks"].reshape(
            b, target_t, v, h, w
        )
    if "sky_masks" in data_dict["context"]:
        target_dict["context_sky_masks"] = data_dict["context"]["sky_masks"].reshape(
            b, context_t, v, h, w
        )
    if "dynamic_masks" in data_dict["target"]:
        target_dict["target_dynamic_masks"] = data_dict["target"]["dynamic_masks"].reshape(
            b, target_t, v, h, w
        )
    if "ground_masks" in data_dict["target"]:
        target_dict["target_ground_masks"] = data_dict["target"]["ground_masks"].reshape(
            b, target_t, v, h, w
        )
    if "pseudo_depth" in data_dict["target"]:
        target_dict["target_pseudo_depth"] = data_dict["target"]["pseudo_depth"].reshape(
            b, target_t, v, pseudo_depth_h, pseudo_depth_w
        )
    if "pseudo_depth_conf" in data_dict["target"]:
        target_dict['target_pseudo_depth_conf'] = data_dict["target"]['pseudo_depth_conf'].reshape(
            b, target_t, v, pseudo_depth_conf_h, pseudo_depth_conf_w
        )

    if "semantic_labels" in data_dict["target"]:
        target_semantic_labels = data_dict["target"]["semantic_labels"].reshape(b, target_t, v, h, w).to(torch.int32)
        target_semantic_labels_mask = data_dict["target"]["semantic_labels_mask"].reshape(b, target_t, v)
        if target_semantic_labels.max() > 0:  # Filtering none semantic label data
            target_dict["target_semantic_labels"] = target_semantic_labels
            target_dict["target_semantic_labels_mask"] = target_semantic_labels_mask
        if is_vis:
            # for visualizing
            semantic_feats = torch.zeros(target_semantic_labels.shape + (get_text_label_feats(SEMANTIC_LABEL_LIST).shape[-1],))
            semantic_feats[:] = get_text_label_feats(SEMANTIC_LABEL_LIST)[target_semantic_labels]
            semantic_feats = einops.rearrange(semantic_feats, 'b t v h w c -> b t v c h w', b=b, v=v)
            target_dict["target_feat"] = semantic_feats  # this will be covered by args.online_feat
    # support multiple offline features
    # target feature offline feature
    # if "feat" in data_dict["target"]:
    #     feat = data_dict["target"]["feat"]
    #     feat_dim, feat_h, feat_w = feat.shape[-3:]
    #     target_dict["target_feat"] = feat.reshape(b, target_t, v, feat_dim, feat_h, feat_w)
    # else:
    #     target_dict["target_feat"] = None
    # only lseg feature supports online feature currently
    if not os.getenv("CONTEXT_FEAT") and "frame_images_to_extract_feat" in data_dict["target"]:
        assert feat_extractor is not None
        images_to_extract_feat = data_dict["target"]["frame_images_to_extract_feat"]
        images_to_extract_feat = einops.rearrange(images_to_extract_feat, 'b tv c h w -> (b tv) c h w')
        with torch.no_grad():
            feat = feat_extractor.extract_lseg_feat_by_chunk(images_to_extract_feat.to(device), (h, w))
            # feat = feat_extractor.extract_lseg_feat_one_by_one(images_to_extract_feat.to(device), (h, w))
        feat = einops.rearrange(feat, '(b t v) c h w -> b t v c h w', b=b, v=v)
        target_dict["target_feat"] = feat

    input_dict["context_frame_idx"] = data_dict["context"]["frame_idx"]
    input_dict["target_frame_idx"] = data_dict["target"]["frame_idx"]
    target_dict["target_frame_idx"] = data_dict["target"]["frame_idx"]
    input_dict = {k: v.to(device) for k, v in input_dict.items()}
    target_dict = {k: v.to(device) for k, v in target_dict.items()}
    input_dict["timespan"] = timespan
    input_dict["fps"] = data_dict["fps"]
    input_dict["scene_id"] = data_dict["scene_id"]
    input_dict["scene_name"] = data_dict["scene_name"]
    input_dict["height"], input_dict["width"] = h, w
    if 'segment_to_ref' in data_dict:
        input_dict['segment_to_ref'] = data_dict['segment_to_ref']
    return input_dict, target_dict


def prepare_inputs_and_targets_novel_view(data_dict, device=torch.device("cpu")):
    raise NotImplementedError("Legacy code, not tested for a while. But you can use it as a reference.")

    assert data_dict["context"]["image"].dim() == 5, "need to be b, tv, c, h, w"
    b, tv, c, h, w = data_dict["context"]["image"].shape
    v = int(data_dict["context"]["num_views"].flatten()[0].item())
    context_t = tv // v
    target_t = data_dict["target"]["image"].shape[1] // v
    # move the camera to right by 1 meter:
    # freeze time
    # cam_to_worlds = data_dict["target"]["camtoworld"].view(b, target_t, v, 4, 4)
    # cam_to_worlds0 = cam_to_worlds[:, 0:1]
    # cam_to_worlds = cam_to_worlds0.expand(-1, target_t, -1, -1, -1)
    # data_dict["target"]["camtoworld"] = cam_to_worlds.reshape(b, target_t * v, 4, 4)
    ### novel view
    cam_to_worlds = data_dict["target"]["camtoworld"].view(b, target_t, v, 4, 4)
    left_cam_to_worlds = cam_to_worlds[:, :, 0].numpy()
    center_cam_to_worlds = cam_to_worlds[:, :, 1].numpy()
    right_cam_to_worlds = cam_to_worlds[:, :, 2].numpy()
    new_left_cam_to_worlds = []
    new_center_cam_to_worlds = []
    new_right_cam_to_worlds = []
    for bix in range(left_cam_to_worlds.shape[0]):
        new_path = get_path_front_left_lift_then_spiral_forward(
            left_cam_to_worlds[bix],
            num_frames=target_t,
            duration_frames=20,
        )
        new_left_cam_to_worlds.append(torch.from_numpy(new_path))
        new_path = get_path_front_left_lift_then_spiral_forward(
            center_cam_to_worlds[bix],
            num_frames=target_t,
            duration_frames=20,
        )
        new_center_cam_to_worlds.append(torch.from_numpy(new_path))
        new_path = get_path_front_left_lift_then_spiral_forward(
            right_cam_to_worlds[bix],
            num_frames=target_t,
            duration_frames=20,
        )
        new_right_cam_to_worlds.append(torch.from_numpy(new_path))
    new_left_cam_to_worlds = torch.stack(new_left_cam_to_worlds, dim=0)
    new_center_cam_to_worlds = torch.stack(new_center_cam_to_worlds, dim=0)
    new_right_cam_to_worlds = torch.stack(new_right_cam_to_worlds, dim=0)
    new_cam_to_worlds = torch.stack(
        [new_left_cam_to_worlds, new_center_cam_to_worlds, new_right_cam_to_worlds],
        dim=2,
    )
    data_dict["target"]["camtoworld"] = new_cam_to_worlds.reshape(b, target_t * v, 4, 4).to(
        data_dict["target"]["camtoworld"]
    )
    ### novel view
    input_dict = {
        "context_image": data_dict["context"]["image"].reshape(b, context_t, v, c, h, w),
        # targets to render
        "context_camtoworlds": data_dict["context"]["camtoworld"].reshape(b, context_t, v, 4, 4),
        "context_intrinsics": data_dict["context"]["intrinsics"].reshape(b, context_t, v, 3, 3),
        "target_camtoworlds": data_dict["target"]["camtoworld"].reshape(b, target_t, v, 4, 4),
        "target_intrinsics": data_dict["target"]["intrinsics"].reshape(b, target_t, v, 3, 3),
    }
    if "depth" in data_dict["context"]:
        input_dict["context_depth"] = data_dict["context"]["depth"].reshape(b, context_t, v, h, w)
    if "flow" in data_dict["context"]:
        input_dict["context_flow"] = data_dict["context"]["flow"].reshape(b, context_t, v, h, w, 3)
    if "time" in data_dict["context"]:
        input_dict["context_time"] = data_dict["context"]["time"].reshape(b, context_t)
    if "time" in data_dict["target"]:
        input_dict["target_time"] = data_dict["target"]["time"].reshape(b, target_t)
    if "sky_masks" in data_dict["context"]:
        input_dict["context_sky_masks"] = data_dict["context"]["sky_masks"].reshape(
            b, context_t, v, h, w
        )

    target_dict = {
        "target_image": data_dict["target"]["image"].reshape(b, target_t, v, 3, h, w),
    }

    if "depth" in data_dict["target"]:
        target_dict["target_depth"] = data_dict["target"]["depth"].reshape(b, target_t, v, h, w)
    if "depth" in data_dict["context"]:
        target_dict["context_depth"] = data_dict["context"]["depth"].reshape(b, context_t, v, h, w)
    if "flow" in data_dict["target"]:
        target_dict["target_flow"] = data_dict["target"]["flow"].reshape(b, target_t, v, h, w, 3)
    if "sky_masks" in data_dict["target"]:
        target_dict["target_sky_masks"] = data_dict["target"]["sky_masks"].reshape(
            b, target_t, v, h, w
        )
    if "sky_masks" in data_dict["context"]:
        target_dict["context_sky_masks"] = data_dict["context"]["sky_masks"].reshape(
            b, context_t, v, h, w
        )
    if "dynamic_masks" in data_dict["target"]:
        target_dict["target_dynamic_masks"] = data_dict["target"]["dynamic_masks"].reshape(
            b, target_t, v, h, w
        )
    if "ground_masks" in data_dict["target"]:
        target_dict["target_ground_masks"] = data_dict["target"]["ground_masks"].reshape(
            b, target_t, v, h, w
        )

    target_dict["target_frame_idx"] = data_dict["target"]["frame_idx"]
    input_dict = {k: v.to(device) for k, v in input_dict.items()}
    target_dict = {k: v.to(device) for k, v in target_dict.items()}
    try:
        input_dict["height"] = data_dict["height"][0].item()
        input_dict["width"] = data_dict["width"][0].item()
        if "timespan" in data_dict:
            input_dict["timespan"] = data_dict["timespan"].flatten()[0].item()
    except:
        input_dict["height"] = data_dict["height"]
        input_dict["width"] = data_dict["width"]
        if "timespan" in data_dict:
            input_dict["timespan"] = data_dict["timespan"]
    input_dict["context_frame_idx"] = data_dict["context"]["frame_idx"]
    input_dict["scene_id"] = data_dict["scene_id"]
    input_dict["scene_name"] = data_dict["scene_name"]
    return input_dict, target_dict

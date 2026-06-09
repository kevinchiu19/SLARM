import argparse
import copy
import datetime
import json
import logging
import math
import os
import time
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import src.utils.misc as misc
from engine_tools import build_model
from src.dataset.constants import DATASET_DICT
from src.dataset.data_utils import to_batch_tensor
from src.dataset.datasets import SingleSequenceDataset
from src.utils.logging import setup_logging
from src.visualization.video_maker import make_video
from tools.lseg_feat_extractor import LSegFeatureExtractor
from main_slarm import get_args_parser

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    global logger
    args.exp_name = args.model.replace("/", "-") if args.exp_name is None else args.exp_name
    log_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    video_dir = os.path.join(log_dir, "videos")
    args.log_dir, args.ckpt_dir, args.video_dir = log_dir, checkpoint_dir, video_dir

    device = torch.device(args.device)
    seed = args.seed
    misc.fix_random_seeds(seed)
    cudnn.benchmark = True

    # set up logging
    setup_logging(output=log_dir, level=logging.INFO)
    logger = logging.getLogger("PerceptualModel")
    logger.info(f"hostname: {os.uname().nodename}\n")
    logger.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    logger.info(f"Logging to {log_dir}")
    logger.info(json.dumps(args.__dict__, indent=4, sort_keys=True))
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    assert len(args.dataset) == 1, 'Only one dataset is supported per inference session.'
    dataset_meta = DATASET_DICT[args.dataset[0]]
    train_annotation = dataset_meta["annotation_txt_file_train"]
    val_annotation = dataset_meta["annotation_txt_file_val"]
    if train_annotation is not None:
        if "nuscenes" in args.dataset:
            train_annotation = f"data/dataset_scene_list/nuscenes_train.txt"
        else:
            train_annotation = f"{args.data_root}/{train_annotation}"
    if val_annotation is not None:
        if "nuscenes" in args.dataset:
            val_annotation = f"data/dataset_scene_list/nuscenes_val.txt"
        else:
            val_annotation = f"{args.data_root}/{val_annotation}"
        if not os.path.exists(val_annotation):
            val_annotation = None
    num_context_timesteps = dataset_meta["num_context_timesteps"]
    num_target_timesteps = dataset_meta["num_target_timesteps"]
    if args.overwrite_train_ctx_view_with is not None:
        num_context_timesteps = args.overwrite_train_ctx_view_with
    if args.overwrite_test_ctx_view_with is not None:
        num_context_timesteps = args.overwrite_test_ctx_view_with
    if args.overwrite_train_tgt_view_with is not None:
        num_target_timesteps = args.overwrite_train_tgt_view_with
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"annotation_txt_file_list_train: {train_annotation}")

    model = build_model(args)

    logger.info(f"Model = {str(model)}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Parameters: {n_params / 1e6:.2f}M ({n_params:,})")
    model.to(device)
    dataset = SingleSequenceDataset(
        data_root=args.data_root,
        annotation_txt_file_list=train_annotation,
        target_size=args.input_size,
        num_context_timesteps=num_context_timesteps,
        num_target_timesteps=num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=args.load_depth,
        load_flow=args.load_flow,
        load_semantic_label=args.load_semantic_label,
        # load_feat_types=args.load_feat_types.split(',') if args.load_feat_types else [],
        online_feat=args.online_feat,
        img_norm_for_online_feat=args.img_norm_for_online_feat,
    )

    logger.info(f"Dataset contains {len(dataset):,} sequences using {train_annotation}.")
    misc.load_model(args, model)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Trainable Parameters: {num_trainable_params / 1e6:.2f}M")
    model.eval()

    # Load feature extractor
    # if args.online_feat and args.load_feat_types is not None:
    if args.online_feat:
        logger.info("Using online feature, loading feature extractor.")
        feat_extractor = LSegFeatureExtractor(args.lseg_model_pretrained_path, args.lseg_model_scratch_path, dtype=torch.float16 if os.environ.get('DISABLE_BFLOAT') else torch.bfloat16)
        logger.info("Feature extractor loaded.")
    else:
        feat_extractor = None

    scene_id = args.scene_id
    scene_start_index = args.scene_start_index
    scene_end_index = args.scene_end_index
    logger.info(f"The id of the inference scene is {scene_id}, the starting frame is {scene_start_index}, and the ending frame is {scene_end_index}")

    # Find the index corresponding to the scene
    with open(train_annotation, 'r', encoding='utf-8') as f:
        lines_a = [line.strip().split('/')[-1].replace('.json', '') for line in f.readlines()]

    if 'waymo' in args.dataset:
        with open('data/dataset_scene_list/waymo_train_list.txt', 'r', encoding='utf-8') as f:
            lines_b = [line.strip().split('/')[-1] for line in f.readlines()]
        name_to_index_a = {name: idx for idx, name in enumerate(lines_a)}
        name = lines_b[scene_id]
        scene_id_idx = name_to_index_a[name]
    else:
        scene_id_idx = scene_id

    logger.info(f"Preparing data... (This may take a while)")
    data_dict_list = dataset.__getitem__(index=scene_id_idx, start_index=scene_start_index, end_index=scene_end_index)
    data_dict_list = to_batch_tensor(data_dict_list)
    logger.info(f"Done preparing data.")
    video_frames_all = []
    output_name = f"dataset_{args.dataset[0]}_scene_id_{scene_id}_{scene_start_index}-{scene_end_index}.mp4"
    for i in range(len(data_dict_list)):
        data_dict = data_dict_list[i]
        video_frames = make_video(
            dataset=None,
            model=model,
            device=device,
            output_filename=output_name,
            data_dict=data_dict,
            feat_extractor=feat_extractor,
            reverse_video=False,
            save_video=False,
        )
        video_frames_all.extend(video_frames)
    print(f"Saved video to {output_name}")
    imageio.mimsave(output_name, video_frames_all, fps=data_dict["fps"])

if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--scene_id", type=int, default=365)
    parser.add_argument("--scene_start_index", type=int, default=0)
    parser.add_argument("--scene_end_index", type=int, default=180)
    parser.add_argument("--overwrite_train_ctx_view_with", default=None, type=int)
    parser.add_argument("--overwrite_train_tgt_view_with", default=None, type=int)
    parser.add_argument("--overwrite_test_ctx_view_with", default=None, type=int)
    args = parser.parse_args()
    main(args)

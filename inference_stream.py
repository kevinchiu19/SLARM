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
from src.dataset.data_utils import to_batch_tensor, prepare_inputs_and_targets
from src.dataset.datasets import PerceptualModelDataset
from src.utils.logging import setup_logging
from src.visualization.video_maker import make_video, make_clean_video
from tools.lseg_feat_extractor import LSegFeatureExtractor
from main_slarm import get_args_parser
from src.models.stream_session import StreamSession

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

    dataset = PerceptualModelDataset(
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

    window_size = int(args.mode.split('_')[-1])
    session = StreamSession(model, mode=args.mode.split('_')[0], window_size=window_size)

    # Load feature extractor
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

    data_dict = dataset.get_one_scene(index=scene_id_idx, start_index=args.scene_start_index, end_index=args.scene_end_index, time_step=args.time_step)
    data_dict = to_batch_tensor(data_dict)
    if not isinstance(data_dict['num_max_cams'], int):
        num_max_cams = int(data_dict['num_max_cams'][0])
    else:
        num_max_cams = data_dict['num_max_cams']
    input_dict, target_dict = prepare_inputs_and_targets(data_dict, device, v=num_max_cams, feat_extractor=feat_extractor)

    model = model.eval()  # model is needed only pred_dict is None
    dtype = torch.float16 if os.environ.get('DISABLE_BFLOAT') else torch.bfloat16

    if args.use_rotate_cam:
        rotate_input_dict = copy.deepcopy(input_dict)
        # rotate target camera
        # model.apply_novelview_rt(input_dict, degree_x=-70, degree_y=70, degree_z=0, trans_x=50, trans_y=50, trans_z=100)
        model.apply_novelview_rt(rotate_input_dict, degree_x=-80, degree_y=0, degree_z=0, trans_x=-5, trans_y=-1, trans_z=50)
        # TODO: context_time and target_time need further normalization
        rotate_input_dict_list = get_one_frame(rotate_input_dict)
        for one_frame_dict in rotate_input_dict_list:
            predictions = session.forward_stream(one_frame_dict, device, dtype)
        session.clear()
        torch.cuda.empty_cache()

        with torch.no_grad():
            rotate_pred_dict = model.post_processing(rotate_input_dict, predictions['gs_params'],
                                                    time_step=args.time_step,
                                                    pred_feat=predictions['pred_feat'],
                                                    sky_token=predictions['sky_token'],
                                                    affine_tokens=predictions['affine_tokens'],
                                                    pose_enc_list=predictions['pred_context_camera_enc_list'],
                                                    pred_context_depth=predictions['pred_context_depth'],
                                                    pred_context_depth_conf=predictions['pred_context_depth_conf'],
                                                    pred_context_pts3d=predictions['pred_context_pts3d'],
                                                    pred_context_pts3d_conf=predictions['pred_context_pts3d_conf'])
        rotate_render_results = copy.deepcopy(rotate_pred_dict["render_results"])

        del rotate_input_dict, rotate_input_dict_list, predictions, rotate_pred_dict
        torch.cuda.empty_cache()
    else:
        rotate_render_results = None

    input_dict_list = get_one_frame(input_dict, step=args.time_step)

    for one_frame_dict in input_dict_list:
        predictions = session.forward_stream(one_frame_dict, device, dtype)


    clip_input_list = get_input_clip(input_dict, step=args.time_step)
    clip_target_list = get_target_clip(target_dict, step=args.time_step)
    clip_gs_params_list = get_gs_clip(predictions['gs_params'])
    video_frames_all = []
    os.makedirs(f"output/window_{window_size}/", exist_ok=True)
    output_name = f"output/window_{window_size}/{args.dataset[0]}_{scene_id}_frame_{args.scene_start_index}_{args.scene_end_index}.mp4"
    with torch.no_grad():
        for i in range(len(clip_input_list)):
            clip_input = clip_input_list[i]
            clip_target = clip_target_list[i]
            clip_gs_params = clip_gs_params_list[i]

            with torch.autocast(device_type=device.type, dtype=dtype):
                pred_dict = model.post_processing(clip_input, clip_gs_params,
                                                time_step=args.time_step,
                                                pred_feat=predictions['pred_feat'][:, i:i+2],
                                                sky_token=predictions['sky_token'],
                                                affine_tokens=predictions['affine_tokens'],
                                                pose_enc_list=predictions['pred_context_camera_enc_list'],
                                                pred_context_depth=predictions['pred_context_depth'],
                                                pred_context_depth_conf=predictions['pred_context_depth_conf'],
                                                pred_context_pts3d=predictions['pred_context_pts3d'],
                                                pred_context_pts3d_conf=predictions['pred_context_pts3d_conf'],
                                                static_render=False
                                                )
            if args.use_clean_video:
                make_clean_video(
                    dataset=None,
                    model=model,
                    device=device,
                    output_filename=output_name,
                    data_dict=data_dict,
                    pred_dict=pred_dict,
                    rotate_render_results=rotate_render_results,
                    feat_extractor=feat_extractor,
                    skip_depth=True,
                    skip_gt_rgb=True,
                    skip_gt_feat=True,
                    font_size=10,
                )
            else:
                video_frames = make_video(
                    dataset=None,
                    model=model,
                    device=device,
                    data_dict=data_dict,
                    pred_dict=pred_dict,
                    feat_extractor=feat_extractor,
                    input_dict=clip_input,
                    target_dict=clip_target,
                    reverse_video=False,
                    save_video=False,
                )
                video_frames_all.extend(video_frames)
        imageio.mimsave(output_name, video_frames_all, fps=data_dict["fps"])
        print(f"Saved video to {output_name}")

    session.clear()
    del data_dict, input_dict, target_dict, predictions, pred_dict
    torch.cuda.empty_cache()

def get_one_frame(input_dict, step=5):
    images = input_dict['context_image']
    b, t, v, c, h, w = images.size()
    output_dict_list = []

    for i in range(t):
        output_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor) and 'target' in key:
                output_dict[key] = value[:, i*step:(i+1)*step, ...]
            elif isinstance(value, torch.Tensor) and 'context' in key:
                output_dict[key] = value[:, i:i+1, ...]
            else:
                output_dict[key] = value
        output_dict_list.append(output_dict)

    return output_dict_list

def get_target_clip(target_dict, step=5):
    context_flow = target_dict['context_flow']
    b, t, v, h, w, c = context_flow.size()
    output_dict_list = []

    for i in range(t-1):
        output_dict = {}
        for key, value in target_dict.items():
            if isinstance(value, torch.Tensor) and 'target' in key:
                output_dict[key] = value[:, i*step+1:(i+1)*step+1, ...]
            # elif isinstance(value, torch.Tensor) and 'context' in key:
            #     output_dict[key] = value[:, i:i+1, ...]
            # else:
            #     output_dict[key] = value
        output_dict_list.append(output_dict)

    return output_dict_list

def get_input_clip(input_dict, step=5):
    images = input_dict['context_image']
    b, t, v, c, h, w = images.size()
    output_dict_list = []

    for i in range(t-1):
        output_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor) and 'target' in key:
                output_dict[key] = value[:, i*step+1:(i+1)*step+1, ...] # (1-5) (6,10), ...
            elif isinstance(value, torch.Tensor) and 'context' in key:
                output_dict[key] = value[:, i:i+2, ...] # (0, 1), (1, 2), (2, 3), ...
            else:
                output_dict[key] = value
        output_dict_list.append(output_dict)

    return output_dict_list

def get_gs_clip(gs_params):
    means = gs_params['means']
    b, t, v, h, w, _ = means.size()
    output_dict_list = []

    for i in range(t-1):
        output_dict = {}
        for key, value in gs_params.items():
            if isinstance(value, torch.Tensor) and int(value.shape[1]) == t:
                output_dict[key] = value[:, i:i+2, ...] # (0, 1), (1, 2), (2, 3), ...
            # elif isinstance(value, torch.Tensor) and int(value.shape[1]) != t:
            #     output_dict[key] = value[:, i*step+1:(i+1)*step+1, ...] # (1-5) (6,10), ...
            else:
                output_dict[key] = value # affine, target_sky, images_without_affine
        output_dict_list.append(output_dict)

    return output_dict_list


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--scene_id", type=int, default=365)
    parser.add_argument("--scene_start_index", type=int, default=0)
    parser.add_argument("--scene_end_index", type=int, default=60)
    parser.add_argument("--overwrite_train_ctx_view_with", default=None, type=int)
    parser.add_argument("--overwrite_train_tgt_view_with", default=None, type=int)
    parser.add_argument("--overwrite_test_ctx_view_with", default=None, type=int)
    parser.add_argument("--time_step", type=int, default=5)
    parser.add_argument("--use_clean_video", action="store_true")
    parser.add_argument("--use_rotate_cam", action="store_true")
    args = parser.parse_args()
    main(args)

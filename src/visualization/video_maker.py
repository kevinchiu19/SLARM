import os
import logging
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from matplotlib import cm

from src.dataset.constants import MEAN, STD
from src.dataset.data_utils import (
    prepare_inputs_and_targets,
    prepare_inputs_and_targets_novel_view,
    to_batch_tensor,
)

from .annotation import add_label
from .layout import add_border, hcat, prep_image, vcat
from .visualization_tools import depth_visualizer, scene_flow_to_rgb, feat_visualizer, get_global_pac_linear

logger = logging.getLogger("PerceptualModel")


def get_pca_map(x):
    # x: (...., c) channel last
    x_shape = x.shape
    x = x.view(-1, x.shape[-1])
    x = x @ torch.pca_lowrank(x, q=3, niter=20)[2]
    x = (x - x.min(dim=0)[0]) / (x.max(dim=0)[0] - x.min(dim=0)[0])
    return x.view(*x_shape[:-1], 3)


@torch.no_grad()
def make_video(
    dataset=None,
    model=None,
    device=None,
    output_filename=None,
    scene_id=None,
    skip_plot_gt_depth_and_flow: bool = False,
    data_dict=None,
    input_dict=None,
    target_dict=None,
    pred_dict=None,
    feat_extractor=None,
    reverse_video=True,
    save_video=True,
):
    if save_video:
        assert output_filename is not None  # legacy code compatibility

    if data_dict is None:
        assert dataset is not None
        if scene_id is None:
            scene_id = np.random.randint(0, len(dataset))
        data_dict = dataset.__getitem__(scene_id, np.random.randint(10, 100), return_all=True)
        data_dict = to_batch_tensor(data_dict)
    if not isinstance(data_dict['num_max_cams'], int):
        num_max_cams = int(data_dict['num_max_cams'][0])
    else:
        num_max_cams = data_dict['num_max_cams']
    if num_max_cams > 1:
        font_size = 24
    else:
        font_size = 8
    if input_dict is None or target_dict is None:
        assert device is not None
        input_dict, target_dict = prepare_inputs_and_targets(data_dict, device, v=num_max_cams,
                                                             feat_extractor=feat_extractor,
                                                             is_vis=True)

    if pred_dict is None:
        assert model is not None
        assert device is not None
        model = model.eval()  # model is needed only pred_dict is None
        dtype = torch.float16 if os.environ.get('DISABLE_BFLOAT') else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=dtype):
            # start_time = time.perf_counter()
            # gs_params = model.get_gs_params(input_dict)
            # end_time = time.perf_counter()
            # logger.info(f"Time taken to get gs_params: {end_time - start_time} seconds")
            # start_time = time.perf_counter()
            # pred_dict = model.from_gs_params_to_output(gs_params, input_dict)
            # end_time = time.perf_counter()
            # logger.info(f"Time taken to get rendered results: {end_time - start_time} seconds")
            start_time = time.perf_counter()
            pred_dict = model(input_dict)
            end_time = time.perf_counter()
            logger.info(f"Time taken to forward rendered results: {end_time - start_time} seconds")
    B, context_t, context_v, _, H, W = input_dict["context_image"].shape
    _, target_t, target_v, _, H_tgt, W_tgt = target_dict["target_image"].shape

    device = input_dict["context_image"].device
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)

    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        # x = (x * std + mean).clamp(0.0, 1.0)  # NOTE: rgb normalization
        return rearrange(x, "t v h w c -> t v c h w")

    # t, v, c, h, w
    context_images = input_dict["context_image"][0]
    context_images = denormalize(context_images)

    if context_v == 1:
        n_ctx_per_row = 4  # better visualize for 1 view
    elif context_v <= 3:
        n_ctx_per_row = 2
    else:
        n_ctx_per_row = 1
    # concate context images horizontally
    context_frames = []
    for t in range(context_t):
        current_frame_idx = int(input_dict["context_frame_idx"][0][t].item())
        row = add_label(
            hcat(*[context_images[t][v_id] for v_id in range(context_v)]),
            f"Context RGB (t={current_frame_idx})",
            font_size=font_size,
            align="center",
        )
        context_frames.append(row)
    num_rows = max(1, len(context_frames) // n_ctx_per_row)
    context_frames = vcat(
        *[
            hcat(
                *context_frames[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                gap=24,
            )
            for row in range(num_rows)
        ]
    )
    vis_context_item = [context_frames]

    # concate rendered context images horizontally
    rendered_context_frames = []
    if "rendered_context_image" in pred_dict.keys():
        rendered_context_images = pred_dict["rendered_context_image"][0].permute(0, 1, 4, 2, 3)
        rendered_context_images = denormalize(rendered_context_images)
        for t in range(context_t):
            current_frame_idx = int(input_dict["context_frame_idx"][0][t].item())
            row = add_label(
                hcat(*[rendered_context_images[t][v_id] for v_id in range(context_v)]),
                f"Rendered Context RGB (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            rendered_context_frames.append(row)
        num_rows = max(1, len(rendered_context_frames) // n_ctx_per_row)
        rendered_context_frames = vcat(
            *[
                hcat(
                    *rendered_context_frames[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                    gap=24,
                )
                for row in range(num_rows)
            ]
        )
        vis_context_item.append(rendered_context_frames)

    rendered_context_depths_frames = []
    if "rendered_context_depth" in pred_dict.keys():
        # concate rendered context depth horizontally
        # alpha_image = rearrange(pred_dict["rendered_context_alpha"][0], 't v h w -> (t v) h w').detach().cpu().numpy()
        depth_image = rearrange(pred_dict["rendered_context_depth"][0], 't v h w -> (t v) h w').detach().cpu().numpy()
        alpha_image = None
        depth_image = depth_visualizer(depth_image, alpha_image)
        depth_image = torch.from_numpy(depth_image)
        rendered_context_depths = rearrange(depth_image, "(t v) ... c -> t v c ...", v=context_v)
        rendered_context_depths_frames = []
        for t in range(context_t):
            current_frame_idx = int(input_dict["context_frame_idx"][0][t].item())
            row = add_label(
                hcat(*[rendered_context_depths[t][v_id] for v_id in range(context_v)]),
                f"Rendered Context Depth (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            rendered_context_depths_frames.append(row)
        num_rows = max(1, len(rendered_context_depths_frames) // n_ctx_per_row)
        rendered_context_depths_frames = vcat(
            *[
                hcat(
                    *rendered_context_depths_frames[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                    gap=24,
                )
                for row in range(num_rows)
            ]
        )
        vis_context_item.append(rendered_context_depths_frames)

    rendered_context_alphas_frames = []
    if "rendered_context_alpha" in pred_dict.keys():
        # concate rendered context alphas horizontally
        rendered_context_alphas = pred_dict["rendered_context_alpha"][0].unsqueeze(2).repeat(1, 1, 3, 1, 1)
        for t in range(context_t):
            current_frame_idx = int(input_dict["context_frame_idx"][0][t].item())
            row = add_label(
                hcat(*[rendered_context_alphas[t][v_id] for v_id in range(context_v)]),
                f"Rendered Context Alpha (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            rendered_context_alphas_frames.append(row)
        num_rows = max(1, len(rendered_context_alphas_frames) // n_ctx_per_row)
        rendered_context_alphas_frames = vcat(
            *[
                hcat(
                    *rendered_context_alphas_frames[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                    gap=24,
                )
                for row in range(num_rows)
            ]
        )
        vis_context_item.append(rendered_context_alphas_frames)

    target_images = target_dict["target_image"][0]
    target_images = denormalize(target_images)
    render_results = pred_dict["render_results"]

    pred_images = render_results[render_results["rgb_key"]][0]
    pred_images = denormalize(pred_images, already_channel_last=True)

    pred_context_image_list = []
    pred_context_depth_list = []
    pred_context_alpha_list = []
    pred_context_flow_list = []
    for t in range(context_t):
        if f"context_{t}_rendered_image" in render_results.keys():
            pred_context_image_list.append(denormalize(render_results[f'context_{t}_rendered_image'][0], already_channel_last=True))
        if f"context_{t}_rendered_depth" in render_results.keys():
            pred_context_depth_list.append(rearrange(torch.from_numpy(depth_visualizer(
                                            rearrange(render_results[f'context_{t}_rendered_depth'][0], "t v ... -> (t v) ...")
                                            .detach().cpu().numpy(), None)), "(t v) h w c -> t v c h w", t=target_t))
        if f"context_{t}_rendered_alpha" in render_results.keys():
            pred_context_alpha_list.append(render_results[f'context_{t}_rendered_alpha'][0].unsqueeze(2).repeat(1, 1, 3, 1, 1))
        if f"context_{t}_rendered_flow" in render_results.keys():
            pred_context_flow_list.append(rearrange(
                        scene_flow_to_rgb(rearrange(render_results[f'context_{t}_rendered_flow'][0],  "t v ... -> (t v) ..."), flow_max_radius=15),
                    "(t v) h w c -> t v c h w", t=target_t))

    if "rendered_motion_seg" in render_results:
        # Get the max index (clusters) from the rendered results
        max_idx = render_results["rendered_motion_seg"][0]

        # Identify unique clusters
        unique_clusters = torch.unique(max_idx)
        try:
            velocities = pred_dict["gs_params"]["motion_bases"][0][unique_clusters]
        except:
            velocities = pred_dict["gs_params"]["motion_bases"][0].mean(dim=0)[unique_clusters]
        velocity_norm = torch.norm(velocities, dim=-1)
        # Sort the unique clusters according to velocity norm (lowest first)
        sorted_indices = torch.argsort(velocity_norm)
        sorted_clusters = unique_clusters[sorted_indices]

        # Number of unique clusters
        num_unique_clusters = len(unique_clusters)

        # Create a new colormap based on the unique clusters
        cmap = cm.get_cmap("rainbow", num_unique_clusters)

        # Map sorted unique clusters to new colors
        cluster_to_color_map = torch.tensor([cmap(i) for i in range(num_unique_clusters)])[
            :, :3
        ].to(max_idx.device)

        # Create a mapping from original clusters to the reassigned clusters
        cluster_mapping = torch.zeros_like(max_idx)

        # Map each pixel in max_idx to the new cluster index based on sorted clusters
        for new_cluster_idx, original_cluster in enumerate(sorted_clusters):
            cluster_mapping[max_idx == original_cluster] = new_cluster_idx

        # Assign the new colors to the cluster image
        cluster_image = cluster_to_color_map[cluster_mapping]
        if cluster_image.shape[-3] != H_tgt or cluster_image.shape[-2] != W_tgt:
            cluster_image = F.interpolate(
                rearrange(cluster_image, "t v h w c -> (t v) c h w"),
                size=(H_tgt, W_tgt),
                mode="nearest",
            )
            cluster_image = rearrange(
                cluster_image, "(t v) c h w -> t v h w c", t=target_t, v=target_v
            )
    else:
        cluster_image = None

    # Get Global PCA
    if not os.getenv("CONTEXT_FEAT") and "target_feat" in target_dict.keys():
        pca_linear = get_global_pac_linear(target_dict["target_feat"])
    if os.getenv("CONTEXT_FEAT") and "context_feat" in input_dict.keys():
        pca_linear = get_global_pac_linear(input_dict["context_feat"])

    video_frames = []
    if os.getenv("CONTEXT_FEAT"):
        c_idx_list = (input_dict['context_frame_idx'][0] - target_dict['target_frame_idx'][0][0]).int().tolist()
        c_idx = 0  # init
    for t in range(target_t):
        frame_list = []
        current_frame_idx = int(target_dict["target_frame_idx"][0][t].item())

        for c_t in range(context_t):
            if len(pred_context_image_list) > 0:
                frame_list.append(add_label(
                    hcat(*[pred_context_image_list[c_t][t][v_id] for v_id in range(target_v)]),
                    f"Predicted Context {c_t} RGB (t={current_frame_idx})",
                    font_size=font_size,
                    align="center",
                ))
            if len(pred_context_depth_list) > 0:
                frame_list.append(add_label(
                    hcat(*[pred_context_depth_list[c_t][t][v_id] for v_id in range(target_v)]),
                    f"Predicted Context {c_t} Depth (t={current_frame_idx})",
                    font_size=font_size,
                    align="center",
                ))
            if len(pred_context_alpha_list) > 0:
                frame_list.append(add_label(
                    hcat(*[pred_context_alpha_list[c_t][t][v_id] for v_id in range(target_v)]),
                    f"Predicted Context {c_t} Opacity (t={current_frame_idx})",
                    font_size=font_size,
                    align="center",
                ))
            if len(pred_context_flow_list) > 0:
                frame_list.append(add_label(
                    hcat(*[pred_context_flow_list[c_t][t][v_id] for v_id in range(target_v)]),
                    f"Predicted Context {c_t} Flow (t={current_frame_idx})",
                    font_size=font_size,
                    align="center",
                ))

        pred_rgb = add_label(
            hcat(*[pred_images[t][v_id] for v_id in range(target_v)]),
            f"Predicted RGB (t={current_frame_idx})",
            font_size=font_size,
            align="center",
        )
        frame_list.append(pred_rgb)

        gt_rgb = add_label(
            hcat(*[target_images[t][v_id] for v_id in range(target_v)]),
            f"Target GT RGB (t={current_frame_idx})",
            font_size=font_size,
            align="center",
        )
        frame_list.append(gt_rgb)

        if render_results["decoder_depth_key"] is not None:
            # this is a decoder depth map
            depth_image = render_results[render_results["decoder_depth_key"]][0][t]
            alpha_image = None
            depth_image = depth_image.detach().cpu().numpy()

            depth_image = depth_visualizer(depth_image, alpha_image)
            depth_image = torch.from_numpy(depth_image)
            depth_image = rearrange(depth_image, "v h w c -> v c h w")
            pred_depth = add_label(
                hcat(*depth_image),
                f"Predicted Decoder Depth (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(pred_depth)

        if render_results["depth_key"] is not None:
            # actual gs depth map
            depth_image = render_results[render_results["depth_key"]][0][t]
            alpha_image = render_results[render_results["alpha_key"]][0][t]
            if depth_image.shape[-2] != H_tgt or depth_image.shape[-1] != W_tgt:
                depth_image = F.interpolate(
                    depth_image.unsqueeze(-3),
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(-3)
                alpha_image = F.interpolate(
                    alpha_image.unsqueeze(-3),
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(-3)
            depth_image = depth_image.detach().cpu().numpy()
            alpha_image = alpha_image.detach().cpu().numpy()
            depth_image = depth_visualizer(depth_image, alpha_image)
            depth_image = torch.from_numpy(depth_image)
            depth_image = rearrange(depth_image, "v h w c -> v c h w")
            pred_depth = add_label(
                hcat(*depth_image),
                f"Predicted Depth (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(pred_depth)

        if "target_depth" in target_dict.keys():
            gt_depth = target_dict["target_depth"][0][t]
            gt_depth = gt_depth.detach().cpu().numpy()
            gt_depth = depth_visualizer(gt_depth, gt_depth > 0)
            gt_depth = torch.from_numpy(gt_depth)
            gt_depth = rearrange(gt_depth, "v h w c -> v c h w")
            gt_depth = add_label(
                hcat(*gt_depth),
                f"Target GT Depth (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(gt_depth)
        else:
            if not skip_plot_gt_depth_and_flow:
                gt_depth = torch.full((target_v, 3, H_tgt, W_tgt), 0.5)
                gt_depth = add_label(
                    hcat(*gt_depth),
                    f"Target GT Depth (t={current_frame_idx})",
                    font_size=font_size,
                    align="center",
                )
                frame_list.append(gt_depth)

        if "rendered_feat" in render_results:
            if os.getenv("CONTEXT_FEAT"):
                assert "context_feat" in input_dict.keys()
                gt_feat = input_dict["context_feat"]
            else:
                assert "target_feat" in target_dict.keys()
                gt_feat = target_dict["target_feat"]
            if os.getenv("CONTEXT_FEAT") and c_idx < len(c_idx_list) - 1 and t == c_idx_list[c_idx + 1]:
                c_idx += 1
            t_idx = c_idx if os.getenv("CONTEXT_FEAT") else t
            feat_image = render_results["rendered_feat"][0][t_idx]  # v, h, w, c
            feat_image = rearrange(feat_image, "v h w c -> v c h w")
            # feat_image = feat_image.detach().cpu().numpy()  # for local PCA
            # feat_image = feat_visualizer(feat_image)  # local PCA
            feat_image = feat_visualizer(feat_image, pca_linear)  # global PCA
            # feat_image = torch.from_numpy(feat_image)  # for local PCA
            pred_feat = add_label(
                hcat(*feat_image),
                f"Predicted Feat (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(pred_feat)

            # t_idx = c_idx_list[c_idx] if os.getenv("CONTEXT_FEAT") else t
            # t_idx = c_idx if os.getenv("CONTEXT_FEAT") else t
            gt_feat = gt_feat[0][t_idx]
            # gt_feat = gt_feat.detach().cpu().numpy()  # for local PCA
            # gt_feat = feat_visualizer(gt_feat)  # local PCA
            gt_feat = feat_visualizer(gt_feat, pca_linear)
            # gt_feat = torch.from_numpy(gt_feat)  # for local PCA
            gt_feat = add_label(
                hcat(*gt_feat),
                f"Target GT Feat (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(gt_feat)

        if render_results["flow_key"] is not None:
            flow_image = render_results[render_results["flow_key"]][0][t]
            flow_image = scene_flow_to_rgb(flow_image, flow_max_radius=15)
            flow_image = rearrange(flow_image, "v h w c -> v c h w")
            if flow_image.shape[-2] != H_tgt or flow_image.shape[-1] != W_tgt:
                flow_image = F.interpolate(
                    flow_image,
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=False,
                )
            flow_image = add_label(
                hcat(*flow_image),
                f"Predicted Flow (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(flow_image)

            if "target_flow" in target_dict.keys():
                gt_flow = target_dict["target_flow"][0][t]
                gt_flow = scene_flow_to_rgb(gt_flow, flow_max_radius=15)
                gt_flow = rearrange(gt_flow, "v h w c -> v c h w")
                gt_flow = add_label(
                    hcat(*gt_flow),
                    f"Target GT Flow (t={current_frame_idx})",
                    font_size=font_size,
                    align="center",
                )
                frame_list.append(gt_flow)
            if render_results["depth_key"] is not None:
                alpha_image = torch.from_numpy(alpha_image).unsqueeze(1)
                alpha_image = alpha_image.repeat(1, 3, 1, 1)
                alpha_image = add_label(
                    hcat(*alpha_image),
                    f"Predicted Opacity (t={current_frame_idx})",
                    font_size=font_size,
                    align="center",
                )
                frame_list.append(alpha_image)
            if "target_sky_masks" in target_dict.keys():
                sky_mask = target_dict["target_sky_masks"][0][t].unsqueeze(1)
                sky_mask = sky_mask.repeat(1, 3, 1, 1)
                sky_mask = add_label(
                    hcat(*sky_mask),
                    f"GT Sky Mask (t={current_frame_idx})",
                    font_size=font_size,
                    align="center",
                )
                frame_list.append(sky_mask)
            if cluster_image is not None:
                cluster_image_t = cluster_image[t]
                cluster_image_t = rearrange(cluster_image_t, "v h w c -> v c h w")
                cluster_image_t = add_label(
                    hcat(*cluster_image_t),
                    f"Motion Segmentation (t={current_frame_idx})",
                    font_size=font_size,
                    align="center",
                )
                frame_list.append(cluster_image_t)

        num_rows = len(frame_list) // n_ctx_per_row
        frame = vcat(
            *vis_context_item,
            vcat(
                *[
                    hcat(
                        *frame_list[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                        gap=24,
                    )
                    for row in range(num_rows)
                ]
            ),
        )
        # if there's a residual, we add it to the end
        if len(frame_list) % n_ctx_per_row != 0:
            frame = vcat(
                frame,
                hcat(
                    *frame_list[num_rows * n_ctx_per_row :],
                    gap=24,
                ),
            )

        frame = add_border(
            add_label(
                frame,
                f"Scene{input_dict['scene_id']:03d}-{input_dict['scene_name'][:15]}",
                font_size=24,
                align="center",
            )
        )
        video_frames.append(prep_image(frame))
    if data_dict is not None and 'relative_clip_start_id' in data_dict:
        video_frames = video_frames[data_dict['relative_clip_start_id']:]
        print(f" ---- Video relative_clip_start_id {data_dict['relative_clip_start_id']}. ---- ")
    if reverse_video:
        video_frame_reversed = video_frames[::-1][1:-1]
        video_frames.extend(video_frame_reversed)
    if save_video:
        imageio.mimsave(output_filename, video_frames, fps=data_dict["fps"])
    return video_frames

@torch.no_grad()
def make_clean_video(
    dataset=None,
    model=None,
    device=None,
    output_filename=None,
    scene_id=None,
    skip_plot_gt_depth_and_flow: bool = False,
    data_dict=None,
    input_dict=None,
    target_dict=None,
    rotate_render_results=None,
    pred_dict=None,
    feat_extractor=None,
    skip_depth=False,
    skip_gt_rgb=False,
    skip_gt_feat=False,
    font_size=None,
):
    assert output_filename is not None  # legacy code compatibility

    if data_dict is None:
        assert dataset is not None
        if scene_id is None:
            scene_id = np.random.randint(0, len(dataset))
        data_dict = dataset.__getitem__(scene_id, np.random.randint(10, 100), return_all=True)
        data_dict = to_batch_tensor(data_dict)
    if not isinstance(data_dict['num_max_cams'], int):
        num_max_cams = int(data_dict['num_max_cams'][0])
    else:
        num_max_cams = data_dict['num_max_cams']
    if font_size is None:
        if num_max_cams > 1:
            font_size = 24
        else:
            font_size = 8
    if input_dict is None or target_dict is None:
        assert device is not None
        input_dict, target_dict = prepare_inputs_and_targets(data_dict, device, v=num_max_cams,
                                                             feat_extractor=feat_extractor)

    if pred_dict is None:
        assert model is not None
        assert device is not None
        model = model.eval()  # model is needed only pred_dict is None
        dtype = torch.float16 if os.environ.get('DISABLE_BFLOAT') else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=dtype):
            start_time = time.perf_counter()
            pred_dict = model(input_dict)
            end_time = time.perf_counter()
            logger.info(f"Time taken to forward rendered results: {end_time - start_time} seconds")

    B, context_t, context_v, _, H, W = input_dict["context_image"].shape
    _, target_t, target_v, _, H_tgt, W_tgt = target_dict["target_image"].shape

    device = input_dict["context_image"].device
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)

    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        # x = (x * std + mean).clamp(0.0, 1.0)  # NOTE: rgb normalization
        return rearrange(x, "t v h w c -> t v c h w")

    # t, v, c, h, w
    context_images = input_dict["context_image"][0]
    context_images = denormalize(context_images)

    if context_v == 1:
        n_ctx_per_row = 4  # better visualize for 1 view
    elif context_v <= 3:
        n_ctx_per_row = 2
    else:
        n_ctx_per_row = 1
        # concate context images horizontally
    if skip_depth and skip_gt_rgb:
        n_ctx_per_row = 1

    target_images = target_dict["target_image"][0]
    target_images = denormalize(target_images)
    render_results = pred_dict["render_results"]

    pred_images = render_results[render_results["rgb_key"]][0]
    pred_images = denormalize(pred_images, already_channel_last=True)

    if rotate_render_results is not None:
        rotate_pred_images = rotate_render_results[render_results["rgb_key"]][0]
        rotate_pred_images = denormalize(rotate_pred_images, already_channel_last=True)
    else:
        rotate_pred_images = None

    if "rendered_motion_seg" in render_results:
        # Get the max index (clusters) from the rendered results
        max_idx = render_results["rendered_motion_seg"][0]

        # Identify unique clusters
        unique_clusters = torch.unique(max_idx)
        try:
            velocities = pred_dict["gs_params"]["motion_bases"][0][unique_clusters]
        except:
            velocities = pred_dict["gs_params"]["motion_bases"][0].mean(dim=0)[unique_clusters]
        velocity_norm = torch.norm(velocities, dim=-1)
        # Sort the unique clusters according to velocity norm (lowest first)
        sorted_indices = torch.argsort(velocity_norm)
        sorted_clusters = unique_clusters[sorted_indices]

        # Number of unique clusters
        num_unique_clusters = len(unique_clusters)

        # Create a new colormap based on the unique clusters
        cmap = cm.get_cmap("rainbow", num_unique_clusters)

        # Map sorted unique clusters to new colors
        cluster_to_color_map = torch.tensor([cmap(i) for i in range(num_unique_clusters)])[
            :, :3
        ].to(max_idx.device)

        # Create a mapping from original clusters to the reassigned clusters
        cluster_mapping = torch.zeros_like(max_idx)

        # Map each pixel in max_idx to the new cluster index based on sorted clusters
        for new_cluster_idx, original_cluster in enumerate(sorted_clusters):
            cluster_mapping[max_idx == original_cluster] = new_cluster_idx

        # Assign the new colors to the cluster image
        cluster_image = cluster_to_color_map[cluster_mapping]
        if cluster_image.shape[-3] != H_tgt or cluster_image.shape[-2] != W_tgt:
            cluster_image = F.interpolate(
                rearrange(cluster_image, "t v h w c -> (t v) c h w"),
                size=(H_tgt, W_tgt),
                mode="nearest",
            )
            cluster_image = rearrange(
                cluster_image, "(t v) c h w -> t v h w c", t=target_t, v=target_v
            )
    else:
        cluster_image = None

    # Get Global PCA
    if not os.getenv("CONTEXT_FEAT") and "target_feat" in target_dict.keys():
        pca_linear = get_global_pac_linear(target_dict["target_feat"])
    if os.getenv("CONTEXT_FEAT") and "context_feat" in input_dict.keys():
        pca_linear = get_global_pac_linear(input_dict["context_feat"])

    if os.getenv("CONTEXT_FEAT"):
        c_idx_list = (input_dict['context_frame_idx'][0] - target_dict['target_frame_idx'][0][0]).int().tolist()
        c_idx = 0  # init

    video_frames = []
    for t in range(0, int(target_t)):
        frame_list = []
        current_frame_idx = int(target_dict["target_frame_idx"][0][t].item())
        pred_rgb = add_label(
            hcat(*[pred_images[t][v_id] for v_id in range(target_v)]),
            f"Predicted RGB (t={current_frame_idx})",
            font_size=font_size,
            align="center",
        )
        frame_list.append(pred_rgb)

        if rotate_pred_images is not None:
            rotate_pred_rgb = add_label(
                hcat(*[rotate_pred_images[t][v_id] for v_id in range(target_v)]),
                f"Predicted RGB (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(rotate_pred_rgb)

        if not skip_gt_rgb:
            gt_rgb = add_label(
                hcat(*[target_images[t][v_id] for v_id in range(target_v)]),
                f"Target GT RGB (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(gt_rgb)

        if render_results["decoder_depth_key"] is not None and not skip_depth:
            # this is a decoder depth map
            depth_image = render_results[render_results["decoder_depth_key"]][0][t]
            alpha_image = None
            depth_image = depth_image.detach().cpu().numpy()

            depth_image = depth_visualizer(depth_image, alpha_image)
            depth_image = torch.from_numpy(depth_image)
            depth_image = rearrange(depth_image, "v h w c -> v c h w")
            pred_depth = add_label(
                hcat(*depth_image),
                f"Predicted Decoder Depth (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(pred_depth)

        if render_results["depth_key"] is not None and not skip_depth:
            # actual gs depth map
            depth_image = render_results[render_results["depth_key"]][0][t]
            alpha_image = render_results[render_results["alpha_key"]][0][t]
            if depth_image.shape[-2] != H_tgt or depth_image.shape[-1] != W_tgt:
                depth_image = F.interpolate(
                    depth_image.unsqueeze(-3),
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(-3)
                alpha_image = F.interpolate(
                    alpha_image.unsqueeze(-3),
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(-3)
            depth_image = depth_image.detach().cpu().numpy()
            alpha_image = alpha_image.detach().cpu().numpy()
            depth_image = depth_visualizer(depth_image, alpha_image)
            depth_image = torch.from_numpy(depth_image)
            depth_image = rearrange(depth_image, "v h w c -> v c h w")
            pred_depth = add_label(
                hcat(*depth_image),
                f"Predicted Depth (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(pred_depth)

        if "target_depth" in target_dict.keys() and not skip_depth:
            gt_depth = target_dict["target_depth"][0][t]
            gt_depth = gt_depth.detach().cpu().numpy()
            gt_depth = depth_visualizer(gt_depth, gt_depth > 0)
            gt_depth = torch.from_numpy(gt_depth)
            gt_depth = rearrange(gt_depth, "v h w c -> v c h w")
            gt_depth = add_label(
                hcat(*gt_depth),
                f"Target GT Depth (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(gt_depth)
        else:
            if not skip_plot_gt_depth_and_flow and not skip_depth:
                gt_depth = torch.full((target_v, 3, H_tgt, W_tgt), 0.5)
                gt_depth = add_label(
                    hcat(*gt_depth),
                    f"Target GT Depth (t={current_frame_idx})",
                    font_size=font_size,
                    align="center",
                )
                frame_list.append(gt_depth)

        if "rendered_feat" in render_results:
            if os.getenv("CONTEXT_FEAT"):
                assert "context_feat" in input_dict.keys()
                gt_feat = input_dict["context_feat"]
            else:
                assert "target_feat" in target_dict.keys()
                gt_feat = target_dict["target_feat"]
            if os.getenv("CONTEXT_FEAT") and c_idx < len(c_idx_list) - 1 and t == c_idx_list[c_idx + 1]:
                c_idx += 1
            t_idx = c_idx if os.getenv("CONTEXT_FEAT") else t
            feat_image = render_results["rendered_feat"][0][t_idx]  # v, h, w, c
            feat_image = rearrange(feat_image, "v h w c -> v c h w")
            # feat_image = feat_image.detach().cpu().numpy()  # for local PCA
            # feat_image = feat_visualizer(feat_image)  # local PCA
            feat_image = feat_visualizer(feat_image, pca_linear)  # global PCA
            # feat_image = torch.from_numpy(feat_image)  # for local PCA
            pred_feat = add_label(
                hcat(*feat_image),
                f"Predicted Feat (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            )
            frame_list.append(pred_feat)

            if not skip_gt_feat:
                # t_idx = c_idx_list[c_idx] if os.getenv("CONTEXT_FEAT") else t
                # t_idx = c_idx if os.getenv("CONTEXT_FEAT") else t
                gt_feat = gt_feat[0][t_idx]
                # gt_feat = gt_feat.detach().cpu().numpy()  # for local PCA
                # gt_feat = feat_visualizer(gt_feat)  # local PCA
                gt_feat = feat_visualizer(gt_feat, pca_linear)
                # gt_feat = torch.from_numpy(gt_feat)  # for local PCA
                gt_feat = add_label(
                    hcat(*gt_feat),
                    f"Target GT Feat (t={current_frame_idx})",
                    font_size=font_size,
                    align="center",
                )
                frame_list.append(gt_feat)

        num_rows = len(frame_list) // n_ctx_per_row
        frame_rows = [
            hcat(
                *frame_list[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                gap=24,
            )
            for row in range(num_rows)
        ]
        # if len(frame_rows) > 0:
        frame = vcat(*frame_rows)
        # if there's a residual, we add it to the end
        if len(frame_list) % n_ctx_per_row != 0:
            frame = vcat(
                frame,
                hcat(
                    *frame_list[num_rows * n_ctx_per_row :],
                    gap=24,
                ),
            )
        video_frames.append(prep_image(frame))
    imageio.mimsave(output_filename, video_frames, fps=data_dict["fps"])
    return video_frames

@torch.no_grad()
def make_video_vis(
    dataset,
    model,
    device,
    output_filename,
    scene_id=None,
    data_dict=None,
    args=None,
    input_dict=None,
    target_dict=None,
    pred_dict=None,
    time_step=10,
):
    data_dict = dataset.__getitem__(scene_id, time_step, return_all=True)
    data_dict = to_batch_tensor(data_dict)
    input_dict, target_dict = prepare_inputs_and_targets(
        # input_dict, target_dict = prepare_inputs_and_targets_novel_view(
        data_dict,
        device,
    )
    model = model.eval()
    pred_dict = model(input_dict)
    B, context_t, context_v, _, H, W = input_dict["context_image"].shape
    _, target_t, target_v, _, H_tgt, W_tgt = target_dict["target_image"].shape

    device = input_dict["context_image"].device
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)

    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        # x = (x * std + mean).clamp(0.0, 1.0)
        return rearrange(x, "t v h w c -> t v c h w")

    # t, v, c, h, w
    context_images = input_dict["context_image"][0]
    context_images = denormalize(context_images)

    if context_v <= 3:
        n_ctx_per_row = 2
    else:
        n_ctx_per_row = 1
        # concate context images horizontally
    context_frames = []
    for t in range(context_t):
        current_frame_idx = int(input_dict["context_frame_idx"][0][t].item())
        row = add_label(
            hcat(*[context_images[t][v_id] for v_id in range(context_v)]),
            f"Context RGB (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        context_frames.append(row)
    num_rows = max(1, len(context_frames) // n_ctx_per_row)
    context_frames = vcat(
        *[
            hcat(
                *context_frames[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                gap=24,
            )
            for row in range(num_rows)
        ]
    )

    target_images = target_dict["target_image"][0]
    target_images = denormalize(target_images)
    render_results = pred_dict["render_results"]

    pred_images = render_results[render_results["rgb_key"]][0]
    pred_images = denormalize(pred_images, already_channel_last=True)
    if "rendered_motion_seg" in render_results:
        # Get the max index (clusters) from the rendered results
        max_idx = render_results["rendered_motion_seg"][0]

        # Identify unique clusters
        unique_clusters = torch.unique(max_idx)
        velocities = pred_dict["gs_params"]["motion_bases"][0][unique_clusters]
        velocity_norm = torch.norm(velocities, dim=-1)
        # Sort the unique clusters according to velocity norm (lowest first)
        sorted_indices = torch.argsort(velocity_norm)
        sorted_clusters = unique_clusters[sorted_indices]

        # Number of unique clusters
        num_unique_clusters = len(sorted_clusters)

        # Create a new colormap based on the sorted unique clusters
        cmap = cm.get_cmap("rainbow", num_unique_clusters)

        # Map sorted unique clusters to new colors
        cluster_to_color_map = torch.tensor([cmap(i) for i in range(num_unique_clusters)])[
            :, :3
        ].to(max_idx.device)

        # Create a mapping from original clusters to the reassigned clusters
        cluster_mapping = torch.zeros_like(max_idx)

        # Map each pixel in max_idx to the new cluster index based on sorted clusters
        for new_cluster_idx, original_cluster in enumerate(sorted_clusters):
            cluster_mapping[max_idx == original_cluster] = new_cluster_idx

        # Assign the new colors to the cluster image
        cluster_image = cluster_to_color_map[cluster_mapping]
        if cluster_image.shape[-3] != H_tgt or cluster_image.shape[-2] != W_tgt:
            cluster_image = F.interpolate(
                rearrange(cluster_image, "t v h w c -> (t v) c h w"),
                size=(H_tgt, W_tgt),
                mode="nearest",
            )
            cluster_image = rearrange(
                cluster_image, "(t v) c h w -> t v h w c", t=target_t, v=target_v
            )
    else:
        cluster_image = None

    video_frames = []
    for t in range(target_t):
        frame_list = []
        current_frame_idx = int(target_dict["target_frame_idx"][0][t].item())
        pred_rgb = add_label(
            hcat(*[pred_images[t][v_id] for v_id in range(target_v)]),
            f"Predicted RGB (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        frame_list.append(pred_rgb)

        gt_rgb = add_label(
            hcat(*[target_images[t][v_id] for v_id in range(target_v)]),
            f"Target GT RGB (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        frame_list.append(gt_rgb)

        if render_results["decoder_depth_key"] is not None:
            # this is a decoder depth map
            depth_image = render_results[render_results["decoder_depth_key"]][0][t]
            alpha_image = None
            depth_image = depth_image.detach().cpu().numpy()

            depth_image = depth_visualizer(depth_image, alpha_image)
            depth_image = torch.from_numpy(depth_image)
            depth_image = rearrange(depth_image, "v h w c -> v c h w")
            pred_depth = add_label(
                hcat(*depth_image),
                f"Predicted Decoder Depth (t={current_frame_idx})",
                font_size=24,
                align="center",
            )
            frame_list.append(pred_depth)

        if render_results["depth_key"] is not None:
            # actual gs depth map
            depth_image = render_results[render_results["depth_key"]][0][t]
            alpha_image = render_results[render_results["alpha_key"]][0][t]
            if depth_image.shape[-2] != H_tgt or depth_image.shape[-1] != W_tgt:
                depth_image = F.interpolate(
                    depth_image.unsqueeze(-3),
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(-3)
                alpha_image = F.interpolate(
                    alpha_image.unsqueeze(-3),
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(-3)
            depth_image = depth_image.detach().cpu().numpy()
            alpha_image = alpha_image.detach().cpu().numpy()
            depth_image = depth_visualizer(depth_image, alpha_image)
            depth_image = torch.from_numpy(depth_image)
            depth_image = rearrange(depth_image, "v h w c -> v c h w")
            pred_depth = add_label(
                hcat(*depth_image),
                f"Predicted Depth (t={current_frame_idx})",
                font_size=24,
                align="center",
            )
            frame_list.append(pred_depth)

            if render_results["depth_key"] is not None:
                alpha_image = torch.from_numpy(alpha_image).unsqueeze(1)
                alpha_image = alpha_image.repeat(1, 3, 1, 1)
                alpha_image = add_label(
                    hcat(*alpha_image),
                    f"Predicted Opacity (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(alpha_image)

        if render_results["flow_key"] is not None:
            flow_image = render_results[render_results["flow_key"]][0][t]
            flow_image = scene_flow_to_rgb(flow_image, flow_max_radius=15)
            flow_image = rearrange(flow_image, "v h w c -> v c h w")
            if flow_image.shape[-2] != H_tgt or flow_image.shape[-1] != W_tgt:
                flow_image = F.interpolate(
                    flow_image,
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=False,
                )
            flow_image = add_label(
                hcat(*flow_image),
                f"Predicted Flow (t={current_frame_idx})",
                font_size=24,
                align="center",
            )
            frame_list.append(flow_image)

            if "target_flow" in target_dict.keys():
                gt_flow = target_dict["target_flow"][0][t]
                gt_flow = scene_flow_to_rgb(gt_flow, flow_max_radius=15)
                gt_flow = rearrange(gt_flow, "v h w c -> v c h w")
                gt_flow = add_label(
                    hcat(*gt_flow),
                    f"GT Flow (t={current_frame_idx}) (Not used as supervision)",
                    font_size=24,
                    align="center",
                )
                frame_list.append(gt_flow)

            if cluster_image is not None:
                cluster_image_t = cluster_image[t]
                cluster_image_t = rearrange(cluster_image_t, "v h w c -> v c h w")
                cluster_image_t = add_label(
                    hcat(*cluster_image_t),
                    f"Motion Segmentation (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(cluster_image_t)

        num_rows = len(frame_list) // n_ctx_per_row
        frame = vcat(
            context_frames,
            vcat(
                *[
                    hcat(
                        *frame_list[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                        gap=24,
                    )
                    for row in range(num_rows)
                ]
            ),
        )
        # if there's a residual, we add it to the end
        if len(frame_list) % n_ctx_per_row != 0:
            frame = vcat(
                frame,
                hcat(
                    *frame_list[num_rows * n_ctx_per_row :],
                    gap=24,
                ),
            )

        frame = add_border(
            add_label(
                frame,
                f"Scene{input_dict['scene_id']:03d}-{input_dict['scene_name'][:15]}",
                font_size=24,
                align="center",
            )
        )
        video_frames.append(prep_image(frame))
    video_frame_reversed = video_frames[::-1][1:-1]
    video_frames.extend(video_frame_reversed)
    imageio.mimsave(output_filename, video_frames, fps=data_dict["fps"])


@torch.no_grad()
def make_video_av2(
    dataset,
    model,
    device,
    output_filename,
    scene_id=None,
    skip_plot_gt_depth_and_flow: bool = False,
):
    if scene_id is None:
        scene_id = np.random.randint(0, len(dataset))
    data_dict = dataset.__getitem__(scene_id, 10, return_all=True)
    data_dict = to_batch_tensor(data_dict)
    input_dict, target_dict = prepare_inputs_and_targets(
        data_dict,
        device,
    )

    with torch.no_grad():
        pred_dict = model(input_dict)
    B, context_t, context_v, _, H, W = input_dict["context_image"].shape
    _, target_t, target_v, _, H_tgt, W_tgt = target_dict["target_image"].shape

    device = input_dict["context_image"].device
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)

    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        # x = (x * std + mean).clamp(0.0, 1.0)
        return rearrange(x, "t v h w c -> t v c h w")

    # t, v, c, h, w
    context_images = input_dict["context_image"][0]
    context_images = denormalize(context_images)

    def resize(input, size, mode="bilinear"):
        if len(input.shape) == 3:
            input = input.unsqueeze(0)

        elif len(input.shape) == 2:
            input = input.unsqueeze(0).unsqueeze(0)
        output = F.interpolate(input, size=size, mode=mode, align_corners=False)
        return output.squeeze()

    reduct_mat = None
    if context_v <= 3:
        n_ctx_per_row = 2
    else:
        n_ctx_per_row = 1
        # concate context images horizontally
    context_frames = []
    for t in range(context_t):
        current_frame_idx = int(input_dict["context_frame_idx"][0][t].item())
        row = add_label(
            hcat(
                *[
                    (
                        context_images[t][v_id]
                        if v_id != context_v // 2
                        else resize(context_images[t][v_id], (W, H))
                    )
                    for v_id in range(context_v)
                ],
                align="bottom",
            ),
            f"Context RGB (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        context_frames.append(row)
    num_rows = max(1, len(context_frames) // n_ctx_per_row)
    context_frames = vcat(
        *[
            hcat(
                *context_frames[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                gap=24,
            )
            for row in range(num_rows)
        ]
    )

    target_images = target_dict["target_image"][0]
    target_images = denormalize(target_images)
    render_results = pred_dict["render_results"]

    pred_images = render_results[render_results["rgb_key"]][0]
    pred_images = denormalize(pred_images, already_channel_last=True)
    video_frames = []
    for t in range(target_t):
        frame_list = []
        current_frame_idx = int(target_dict["target_frame_idx"][0][t].item())
        pred_rgb = add_label(
            hcat(
                *[
                    (
                        pred_images[t][v_id]
                        if v_id != target_v // 2
                        else resize(pred_images[t][v_id], (W, H))
                    )
                    for v_id in range(target_v)
                ],
                align="bottom",
            ),
            # hcat(*[pred_images[t][v_id] for v_id in range(target_v)]),
            f"Predicted RGB (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        frame_list.append(pred_rgb)

        gt_rgb = add_label(
            hcat(
                *[
                    (
                        target_images[t][v_id]
                        if v_id != target_v // 2
                        else resize(target_images[t][v_id], (W, H))
                    )
                    for v_id in range(target_v)
                ],
                align="bottom",
            ),
            # hcat(*[target_images[t][v_id] for v_id in range(target_v)]),
            f"Target GT RGB (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        frame_list.append(gt_rgb)

        if render_results["decoder_depth_key"] is not None:
            # this is a decoder depth map
            depth_image = render_results[render_results["decoder_depth_key"]][0][t]
            alpha_image = None
            depth_image = depth_image.detach().cpu().numpy()

            depth_image = depth_visualizer(depth_image, alpha_image)
            depth_image = torch.from_numpy(depth_image)
            depth_image = rearrange(depth_image, "v h w c -> v c h w")
            pred_depth = add_label(
                hcat(
                    *[
                        (
                            depth_image[v_id]
                            if v_id != target_v // 2
                            else resize(depth_image[v_id], (W, H))
                        )
                        for v_id in range(target_v)
                    ],
                    align="bottom",
                ),
                # hcat(*depth_image),
                f"Predicted Decoder Depth (t={current_frame_idx})",
                font_size=24,
                align="center",
            )
            frame_list.append(pred_depth)

        if render_results["depth_key"] is not None:
            # actual gs depth map
            depth_image = render_results[render_results["depth_key"]][0][t]
            alpha_image = render_results[render_results["alpha_key"]][0][t]
            if depth_image.shape[-2] != H_tgt or depth_image.shape[-1] != W_tgt:
                depth_image = F.interpolate(
                    depth_image.unsqueeze(-3),
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(-3)
                alpha_image = F.interpolate(
                    alpha_image.unsqueeze(-3),
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(-3)
            depth_image = depth_image.detach().cpu().numpy()
            alpha_image = alpha_image.detach().cpu().numpy()
            depth_image = depth_visualizer(depth_image, alpha_image)
            depth_image = torch.from_numpy(depth_image)
            depth_image = rearrange(depth_image, "v h w c -> v c h w")
            pred_depth = add_label(
                hcat(
                    *[
                        (
                            depth_image[v_id]
                            if v_id != target_v // 2
                            else resize(depth_image[v_id], (W, H))
                        )
                        for v_id in range(target_v)
                    ],
                    align="bottom",
                ),
                # hcat(*depth_image),
                f"Predicted Depth (t={current_frame_idx})",
                font_size=24,
                align="center",
            )
            frame_list.append(pred_depth)

        if "target_depth" in target_dict.keys():
            gt_depth = target_dict["target_depth"][0][t]
            gt_depth = gt_depth.detach().cpu().numpy()
            gt_depth = depth_visualizer(gt_depth, gt_depth > 0)
            gt_depth = torch.from_numpy(gt_depth)
            gt_depth = rearrange(gt_depth, "v h w c -> v c h w")
            gt_depth = add_label(
                hcat(
                    *[
                        (
                            gt_depth[v_id]
                            if v_id != target_v // 2
                            else resize(gt_depth[v_id], (W, H))
                        )
                        for v_id in range(target_v)
                    ],
                    align="bottom",
                ),
                # hcat(*gt_depth),
                f"Target GT Depth (t={current_frame_idx})",
                font_size=24,
                align="center",
            )
            frame_list.append(gt_depth)
        else:
            if not skip_plot_gt_depth_and_flow:
                gt_depth = torch.full((target_v, 3, H_tgt, W_tgt), 0.5)
                gt_depth = add_label(
                    hcat(
                        *[
                            (
                                gt_depth[v_id]
                                if v_id != target_v // 2
                                else resize(gt_depth[v_id], (W, H))
                            )
                            for v_id in range(target_v)
                        ],
                        align="bottom",
                    ),
                    # hcat(*gt_depth),
                    f"Target GT Depth (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(gt_depth)

        if render_results["flow_key"] is not None:
            flow_image = render_results[render_results["flow_key"]][0][t]
            flow_image = scene_flow_to_rgb(flow_image, flow_max_radius=15)
            flow_image = rearrange(flow_image, "v h w c -> v c h w")
            if flow_image.shape[-2] != H_tgt or flow_image.shape[-1] != W_tgt:
                flow_image = F.interpolate(
                    flow_image,
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=False,
                )
            flow_image = add_label(
                hcat(
                    *[
                        (
                            flow_image[v_id]
                            if v_id != target_v // 2
                            else resize(flow_image[v_id], (W, H))
                        )
                        for v_id in range(target_v)
                    ],
                    align="bottom",
                ),
                # hcat(*flow_image),
                f"Predicted Flow (t={current_frame_idx})",
                font_size=24,
                align="center",
            )
            frame_list.append(flow_image)

            if "target_flow" in target_dict.keys():
                gt_flow = target_dict["target_flow"][0][t]
                gt_flow = scene_flow_to_rgb(gt_flow, flow_max_radius=15)
                gt_flow = rearrange(gt_flow, "v h w c -> v c h w")
                gt_flow = add_label(
                    hcat(
                        *[
                            (
                                gt_flow[v_id]
                                if v_id != target_v // 2
                                else resize(gt_flow[v_id], (W, H))
                            )
                            for v_id in range(target_v)
                        ],
                        align="bottom",
                    ),
                    # hcat(*gt_flow),
                    f"Target GT Flow (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(gt_flow)
            if render_results["depth_key"] is not None:
                alpha_image = torch.from_numpy(alpha_image).unsqueeze(1)
                alpha_image = alpha_image.repeat(1, 3, 1, 1)
                alpha_image = add_label(
                    hcat(
                        *[
                            (
                                alpha_image[v_id]
                                if v_id != target_v // 2
                                else resize(alpha_image[v_id], (W, H))
                            )
                            for v_id in range(target_v)
                        ],
                        align="bottom",
                    ),
                    # hcat(*alpha_image),
                    f"Predicted Opacity (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(alpha_image)
            if "target_sky_masks" in target_dict.keys():
                sky_mask = target_dict["target_sky_masks"][0][t].unsqueeze(1)
                sky_mask = sky_mask.repeat(1, 3, 1, 1)
                sky_mask = add_label(
                    hcat(
                        *[
                            (
                                sky_mask[v_id]
                                if v_id != target_v // 2
                                else resize(sky_mask[v_id], (W, H))
                            )
                            for v_id in range(target_v)
                        ],
                        align="bottom",
                    ),
                    # hcat(*sky_mask),
                    f"GT Sky&/Road Mask (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(sky_mask)
        num_rows = len(frame_list) // n_ctx_per_row
        frame = vcat(
            context_frames,
            vcat(
                *[
                    hcat(
                        *frame_list[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                        gap=24,
                    )
                    for row in range(num_rows)
                ]
            ),
        )
        # if there's a residual, we add it to the end
        if len(frame_list) % n_ctx_per_row != 0:
            frame = vcat(
                frame,
                hcat(
                    *frame_list[num_rows * n_ctx_per_row :],
                    gap=24,
                ),
            )

        frame = add_border(
            add_label(
                frame,
                f"Scene{input_dict['scene_id']:03d}-{input_dict['scene_name'][:15]}",
                font_size=24,
                align="center",
            )
        )
        video_frames.append(prep_image(frame))
    video_frame_reversed = video_frames[::-1][1:-1]
    video_frames.extend(video_frame_reversed)
    imageio.mimsave(output_filename, video_frames, fps=data_dict["fps"])
    return output_filename

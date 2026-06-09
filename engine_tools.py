import datetime
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import einops

import src.models as models
import src.utils.distributed as distributed
from src.dataset.constants import MEAN, STD, SEMANTIC_LABEL_LIST
from src.dataset.data_utils import prepare_inputs_and_targets
from src.utils.losses import compute_scene_flow_metrics, compute_semantic_metrics
from src.visualization.video_maker import make_video
if os.getenv("FEAT_DIST"):
    from tools.feats_tools import get_text_label_feats, feat2class


logger = logging.getLogger("PerceptualModel")


def build_model(args):
    if args.model in models.STORM_models:
        model = models.STORM_models[args.model](
            img_size=args.input_size,
            gs_dim=args.gs_dim,
            decoder_type=args.decoder_type,
            grad_checkpointing=not args.disable_grad_checkpointing,
            use_sky_token=args.use_sky_token,
            use_affine_token=args.use_affine_token,
            num_motion_tokens=args.num_motion_tokens,
            num_cams=args.num_max_cameras,
            sigmoid_rgb=args.sigmoid_rgb,
            gs_marbles=args.gs_marbles,
            max_scale=args.max_scale,
            render_context_view=args.render_context_view,
            render_context_frame_contribution=args.render_context_frame_contribution,
            pred_gs_conf=args.pred_gs_conf,
            voxelize=args.voxelize,
            voxel_size=args.voxel_size,
            use_ms3_motion=args.use_ms3_motion,
            add_angular_velocity=args.add_angular_velocity,
            use_render_novel_view=args.use_render_novel_view,
        )
    elif args.model == 'slarm':
        model = models.SLARM(
            img_size=args.input_size,
            gs_dim=args.gs_dim,
            decoder_type=args.decoder_type,
            embed_dim=args.embed_dim,
            patch_size=args.patch_size,
            depth=args.depth,
            patch_embed=args.patch_embed,
            grad_checkpointing=not args.disable_grad_checkpointing,
            vggt_pretrained_weight_filepath=args.vggt_pretrained_weight_filepath,
            enable_depth_head=args.enable_depth_head,
            enable_camera_head=args.enable_camera_head,
            enable_point_head=args.enable_point_head,
            shortcut_rgb=args.shortcut_rgb,
            use_ms3_motion=args.use_ms3_motion,
            gs_marbles=args.gs_marbles,
            render_context_view=args.render_context_view,
            render_context_frame_contribution=args.render_context_frame_contribution,
            use_2dgs=args.use_2dgs,
            pesudo_3dgs=args.pesudo_3dgs,
            save_gaussian=args.save_gaussian,
            gaussian_save_path=args.gaussian_save_path,
            save_rendered_pc=args.save_rendered_pc,
            rendered_pc_save_path=args.rendered_pc_save_path,
            use_time_token=args.use_time_token,
            use_sky_token=args.use_sky_token,
            use_affine_token=args.use_affine_token,
            num_motion_tokens=args.num_motion_tokens,
            pred_gs_conf=args.pred_gs_conf,
            enable_lifespan=args.enable_lifespan,
            voxelize=args.voxelize,
            voxel_size=args.voxel_size,
            num_cams=args.num_max_cameras,
            sigmoid_rgb=args.sigmoid_rgb,
            use_pred_camera_pose=args.use_pred_camera_pose,
            use_pred_depth=args.use_pred_depth,
            add_patch_plucker_embed=args.add_patch_plucker_embed,
            add_camera_embed=args.add_camera_embed,
            concat_plucker_embed=args.concat_plucker_embed,
            use_last_token=args.use_last_token,
            use_render_novel_view=args.use_render_novel_view,
            add_angular_velocity=args.add_angular_velocity,
            # target_feat_types=args.load_feat_types.split(',') if args.load_feat_types else [],
            with_feat=not args.without_feat,
            similarity_probs_threshold=args.similarity_probs_threshold,
            mode=args.mode,
            )
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    return model


@torch.no_grad()
def visualize(args, model, dset_train, step, train_vis_id, device,
    dset_val=None, val_vis_id=None, log_writer=None, feat_extractor=None):
    model.eval()
    global_rank = distributed.get_global_rank()
    split = "train"
    for vis_id, dataset in zip([train_vis_id, val_vis_id], [dset_train, dset_val]):
        if vis_id is None or dataset is None:  # sometimes there is no validation set
            continue

        sample_id = global_rank * 80 + vis_id
        out_pth = f"{args.video_dir}/step{step}-rank{global_rank}-sample{sample_id}-{split}.mp4"

        logger.info(f"saving video to {out_pth}")
        video = make_video(
            dataset,
            model,
            device,
            output_filename=out_pth,
            scene_id=sample_id,
            skip_plot_gt_depth_and_flow=False,
            feat_extractor=feat_extractor,
            # args=args,
        )
        video = np.array(video)
        video = einops.rearrange(video, '(b t) h w c -> b t c h w', b=1) # batch size 1 in visualize
        if log_writer is not None:
            log_writer.add_video(f"video/{split}", video)

        logger.info(f"saved video to {out_pth}")
        split = "val"

    torch.cuda.empty_cache()
    return train_vis_id + 1, val_vis_id + 1 if val_vis_id is not None else None


@torch.no_grad()
def evaluate(dataloader, model, args, name_str=None):
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    mean = torch.tensor(MEAN).to(device)
    std = torch.tensor(STD).to(device)

    eval_result_dir = os.path.join(args.log_dir, "eval_results")
    os.makedirs(eval_result_dir, exist_ok=True)
    logger.info(f"Saving evaluation results to {eval_result_dir}")
    # use yr-mo-dy-hr-min
    if name_str is None:
        name_str = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    def get_numpy(tensor):
        return tensor.squeeze().detach().cpu().numpy()

    # Initialize running sums and counts
    total_samples, total_dynamic_samples, total_valid_dynamic_depth_samples = 0, 0, 0
    total_psnr, total_ssim, total_depth_rmse = 0.0, 0.0, 0.0
    total_occupied_psnr, total_occupied_ssim = 0.0, 0.0
    total_dynamic_psnr, total_dynamic_ssim, total_dynamic_rmse = 0.0, 0.0, 0.0

    total_depth_rmse_0_100, total_depth_rmse_100_200 = 0.0, 0.0
    total_valid_depth_samples_0_100, total_valid_depth_samples_100_200 = 0, 0
    total_dynamic_depth_rmse_0_100, total_dynamic_depth_rmse_100_200 = 0.0, 0.0
    total_valid_dynamic_depth_samples_0_100, total_valid_dynamic_depth_samples_100_200 = 0, 0

    # Safely compute mean, return None when no valid samples
    def safe_mean(sum_val, count):
        if count > 0:
            return sum_val / count
        return None

    # Progress bar display formatting
    def format_value(val, is_numeric=True):
        if val is None:
            return "none"
        if is_numeric:
            return f"{val:.4f}"
        return val

    printed = False
    pbar = tqdm(dataloader, desc="Evaluating")
    for data_dict in pbar:
        input_dict, target_dict = prepare_inputs_and_targets(data_dict, device, v=args.num_max_cameras)
        input_indices = input_dict["context_frame_idx"][0].cpu().numpy().tolist()
        input_indice_start = input_indices[0]
        input_indices = [idx - input_indice_start for idx in input_indices]
        # target indices include all frames between t+0 to t+19, including the input frames
        target_indices = target_dict["target_frame_idx"][0].cpu().numpy().tolist()
        # remove the input frames from target indices to get test indices
        target_indices = [idx - input_indice_start for idx in target_indices]
        test_indices = [idx for idx in target_indices if idx not in input_indices]
        if not printed:
            logger.info(f"Input indices: {input_indices}")
            logger.info(f"Test indices: {test_indices}")
            printed = True
        pred_dict = model(input_dict)
        # evaluate on real target images:
        # b, t, v, c, h, w
        gt_rgb = target_dict["target_image"][:, test_indices]
        # b, t, v, h, w, c
        # gt_rgb = gt_rgb.permute(0, 1, 2, 4, 5, 3) * std + mean  # transform from [-1, 1] to [0, 1]
        gt_rgb = gt_rgb.permute(0, 1, 2, 4, 5, 3)

        height, width = gt_rgb.shape[-3], gt_rgb.shape[-2]
        # btv, h, w, c
        gt_rgb = gt_rgb.reshape(-1, height, width, 3)
        gt_depth = target_dict["target_depth"][:, test_indices].view(-1, height, width)
        gt_sky_mask = target_dict["target_sky_masks"][:, test_indices].view(-1, height, width)
        occupied_mask = (gt_sky_mask == 0).bool()
        if "target_dynamic_masks" in target_dict:
            gt_dynamic_mask = target_dict["target_dynamic_masks"][:, test_indices]
            gt_dynamic_mask = gt_dynamic_mask.view(-1, height, width)
            dynamic_mask = gt_dynamic_mask.bool()
        else:
            dynamic_mask = torch.ones_like(occupied_mask)
        valid_depth_mask = (gt_depth > 0.01) & (gt_depth < 200)

        valid_depth_mask_0_100 = valid_depth_mask & (gt_depth <= 100.0)  # (0.01,100]
        valid_depth_mask_100_200 = valid_depth_mask & (gt_depth > 100.0)  # (100,200)

        rendered_results = pred_dict["render_results"]
        # pred_rgb = rendered_results[rendered_results["rgb_key"]][:, test_indices] * std + mean
        pred_rgb = rendered_results[rendered_results["rgb_key"]][:, test_indices]
        pred_rgb = pred_rgb.reshape(-1, height, width, 3).detach()
        pred_rgb = torch.clamp(pred_rgb, 0, 1)
        if rendered_results["decoder_depth_key"] is None:
            pred_depth = rendered_results[rendered_results["depth_key"]][:, test_indices].view(
                -1, height, width
            )
        else:
            pred_depth = rendered_results[rendered_results["decoder_depth_key"]][
                :, test_indices
            ].view(-1, height, width)
        psnrs, ssim_scores, depth_rmses = [], [], []
        occupied_ssims, occupied_psnrs = [], []
        dynamic_ssims, dynamic_psnrs, dynamic_depth_rmses = [], [], []
        depth_rmses_0_100, depth_rmses_100_200 = [], []
        dynamic_depth_rmses_0_100, dynamic_depth_rmses_100_200 = [], []
        for i in range(len(gt_rgb)):
            ssim_score = ssim(
                get_numpy(pred_rgb[i]),
                get_numpy(gt_rgb[i]),
                data_range=1.0,
                channel_axis=-1,
            )
            ssim_scores.append(ssim_score)
            occupied_ssims.append(
                ssim(
                    get_numpy(pred_rgb[i]),
                    get_numpy(gt_rgb[i]),
                    data_range=1.0,
                    channel_axis=-1,
                    full=True,
                )[1][get_numpy(occupied_mask[i])].mean()
            )
            psnrs.append(
                -10
                * torch.log10(
                    F.mse_loss(
                        pred_rgb[i],
                        gt_rgb[i],
                    )
                ).item()
            )
            occupied_psnrs.append(
                -10
                * torch.log10(
                    F.mse_loss(
                        pred_rgb[i][occupied_mask[i]],
                        gt_rgb[i][occupied_mask[i]],
                    )
                ).item()
            )
            depth_rms = torch.sqrt(
                F.mse_loss(
                    pred_depth[i][valid_depth_mask[i]],
                    gt_depth[i][valid_depth_mask[i]],
                )
            ).item()
            depth_rmses.append(depth_rms)

            # Compute 0-100m depth RMSE (only add when valid samples exist)
            if valid_depth_mask_0_100[i].sum() > 0:
                depth_rms_0_100 = torch.sqrt(
                    F.mse_loss(
                        pred_depth[i][valid_depth_mask_0_100[i]],
                        gt_depth[i][valid_depth_mask_0_100[i]],
                    )
                ).item()
                depth_rmses_0_100.append(depth_rms_0_100)

            # Compute 100-200m depth RMSE (only add when valid samples exist)
            if valid_depth_mask_100_200[i].sum() > 0:
                depth_rms_100_200 = torch.sqrt(
                    F.mse_loss(
                        pred_depth[i][valid_depth_mask_100_200[i]],
                        gt_depth[i][valid_depth_mask_100_200[i]],
                    )
                ).item()
                depth_rmses_100_200.append(depth_rms_100_200)

            # Dynamic region metrics computation
            if dynamic_mask[i].sum() == 0:
                continue
            dynamic_ssims.append(
                ssim(
                    get_numpy(pred_rgb[i]),
                    get_numpy(gt_rgb[i]),
                    data_range=1.0,
                    channel_axis=-1,
                    full=True,
                )[1][get_numpy(dynamic_mask[i])].mean()
            )
            dynamic_psnrs.append(
                -10
                * torch.log10(
                    F.mse_loss(
                        pred_rgb[i][dynamic_mask[i]],
                        gt_rgb[i][dynamic_mask[i]],
                    )
                ).item()
            )

            total_dynamic_samples += 1
            _valid_depth_mask = dynamic_mask[i] & valid_depth_mask[i]
            if _valid_depth_mask.sum() == 0:
                continue
            dynamic_depth_rms = torch.sqrt(
                F.mse_loss(
                    pred_depth[i][dynamic_mask[i] & valid_depth_mask[i]],
                    gt_depth[i][dynamic_mask[i] & valid_depth_mask[i]],
                )
            ).item()
            dynamic_depth_rmses.append(dynamic_depth_rms)
            total_valid_dynamic_depth_samples += 1

            # Dynamic region 0-100m depth RMSE
            _valid_depth_mask_0_100 = dynamic_mask[i] & valid_depth_mask_0_100[i]
            if _valid_depth_mask_0_100.sum() > 0:
                dynamic_depth_rms_0_100 = torch.sqrt(
                    F.mse_loss(
                        pred_depth[i][_valid_depth_mask_0_100],
                        gt_depth[i][_valid_depth_mask_0_100],
                    )
                ).item()
                dynamic_depth_rmses_0_100.append(dynamic_depth_rms_0_100)
                total_valid_dynamic_depth_samples_0_100 += 1

            # Dynamic region 100-200m depth RMSE
            _valid_depth_mask_100_200 = dynamic_mask[i] & valid_depth_mask_100_200[i]
            if _valid_depth_mask_100_200.sum() > 0:
                dynamic_depth_rms_100_200 = torch.sqrt(
                    F.mse_loss(
                        pred_depth[i][_valid_depth_mask_100_200],
                        gt_depth[i][_valid_depth_mask_100_200],
                    )
                ).item()
                dynamic_depth_rmses_100_200.append(dynamic_depth_rms_100_200)
                total_valid_dynamic_depth_samples_100_200 += 1

        psnr_sum = np.sum(psnrs)
        ssim_sum = np.sum(ssim_scores)
        depth_rmse_sum = np.sum(depth_rmses)
        occupied_ssim_sum = np.sum(occupied_ssims)
        occupied_psnr_sum = np.sum(occupied_psnrs)
        dynamic_ssim_sum = np.sum(dynamic_ssims)
        dynamic_psnr_sum = np.sum(dynamic_psnrs)
        dynamic_depth_rmse_sum = np.sum(dynamic_depth_rmses)

        depth_rmse_sum_0_100 = np.sum(depth_rmses_0_100)
        depth_rmse_sum_100_200 = np.sum(depth_rmses_100_200)
        dynamic_depth_rmse_sum_0_100 = np.sum(dynamic_depth_rmses_0_100)
        dynamic_depth_rmse_sum_100_200 = np.sum(dynamic_depth_rmses_100_200)

        valid_samples_0_100 = len(depth_rmses_0_100)
        valid_samples_100_200 = len(depth_rmses_100_200)

        batch_size = len(gt_rgb)
        # Update running sums and counts
        total_psnr += psnr_sum
        total_ssim += ssim_sum
        total_depth_rmse += depth_rmse_sum
        total_occupied_psnr += occupied_psnr_sum
        total_occupied_ssim += occupied_ssim_sum
        total_dynamic_psnr += dynamic_psnr_sum
        total_dynamic_ssim += dynamic_ssim_sum
        total_dynamic_rmse += dynamic_depth_rmse_sum

        total_depth_rmse_0_100 += depth_rmse_sum_0_100
        total_depth_rmse_100_200 += depth_rmse_sum_100_200
        total_valid_depth_samples_0_100 += valid_samples_0_100
        total_valid_depth_samples_100_200 += valid_samples_100_200
        total_dynamic_depth_rmse_0_100 += dynamic_depth_rmse_sum_0_100
        total_dynamic_depth_rmse_100_200 += dynamic_depth_rmse_sum_100_200

        # Compute batch average (return None if no valid samples)
        avg_depth_rmse_0_100 = safe_mean(depth_rmse_sum_0_100, valid_samples_0_100)
        avg_depth_rmse_100_200 = safe_mean(depth_rmse_sum_100_200, valid_samples_100_200)

        # Compute global average (return None if no valid samples)
        global_avg_0_100 = safe_mean(total_depth_rmse_0_100, total_valid_depth_samples_0_100)
        global_avg_100_200 = safe_mean(total_depth_rmse_100_200, total_valid_depth_samples_100_200)

        # Dynamic region global average (return None if no valid samples)
        dyn_global_avg_0_100 = safe_mean(total_dynamic_depth_rmse_0_100, total_valid_dynamic_depth_samples_0_100)
        dyn_global_avg_100_200 = safe_mean(total_dynamic_depth_rmse_100_200, total_valid_dynamic_depth_samples_100_200)

        total_samples += batch_size

        pbar.set_postfix(
            # psnr=psnr_sum / batch_size,
            # ssim=ssim_sum / batch_size,
            # depth_rmse_0_100=format_value(avg_depth_rmse_0_100),
            # depth_rmse_100_200=format_value(avg_depth_rmse_100_200),
            # depth_rmse=depth_rmse_sum / batch_size,
            avg_psnr=total_psnr / total_samples,
            # avg_depth_rmse_0_100=format_value(global_avg_0_100),
            # avg_depth_rmse_100_200=format_value(global_avg_100_200),
            avg_depth_rmse=total_depth_rmse / total_samples,
            avg_dynamic_psnr=total_dynamic_psnr / total_dynamic_samples,
            # avg_dynamic_depth_rmse_0_100=format_value(dyn_global_avg_0_100),
            # avg_dynamic_depth_rmse_100_200=format_value(dyn_global_avg_100_200),
            avg_dynamic_depth_rmse=total_dynamic_rmse / total_valid_dynamic_depth_samples,
        )

    # Create tensors for sums and counts
    total_psnr_tensor = torch.tensor(total_psnr, dtype=torch.float32, device=device)
    total_ssim_tensor = torch.tensor(total_ssim, dtype=torch.float32, device=device)
    total_depth_rmse_tensor = torch.tensor(total_depth_rmse, dtype=torch.float32, device=device)
    total_occupied_psnr_tensor = torch.tensor(total_occupied_psnr, dtype=torch.float32, device=device)
    total_occupied_ssim_tensor = torch.tensor(total_occupied_ssim, dtype=torch.float32, device=device)
    total_dynamic_psnr_tensor = torch.tensor(total_dynamic_psnr, dtype=torch.float32, device=device)
    total_dynamic_ssim_tensor = torch.tensor(total_dynamic_ssim, dtype=torch.float32, device=device)
    total_dynamic_rmse_tensor = torch.tensor(total_dynamic_rmse, dtype=torch.float32, device=device)

    total_depth_rmse_0_100_tensor = torch.tensor(total_depth_rmse_0_100, dtype=torch.float32, device=device)
    total_depth_rmse_100_200_tensor = torch.tensor(total_depth_rmse_100_200, dtype=torch.float32, device=device)
    total_valid_depth_samples_0_100_tensor = torch.tensor(total_valid_depth_samples_0_100, device=device)
    total_valid_depth_samples_100_200_tensor = torch.tensor(total_valid_depth_samples_100_200, device=device)
    total_dynamic_depth_rmse_0_100_tensor = torch.tensor(total_dynamic_depth_rmse_0_100, dtype=torch.float32, device=device)
    total_dynamic_depth_rmse_100_200_tensor = torch.tensor(total_dynamic_depth_rmse_100_200, dtype=torch.float32, device=device)
    total_valid_dynamic_depth_samples_0_100_tensor = torch.tensor(total_valid_dynamic_depth_samples_0_100, device=device)
    total_valid_dynamic_depth_samples_100_200_tensor = torch.tensor(total_valid_dynamic_depth_samples_100_200, device=device)

    total_samples_tensor = torch.tensor(total_samples, device=device)  # dtype int64
    total_dynamic_samples_tensor = torch.tensor(total_dynamic_samples, device=device)  # dtype int64
    total_valid_dynamic_depth_samples_tensor = torch.tensor(
        total_valid_dynamic_depth_samples, device=device
    )  # dtype int64

    torch.cuda.synchronize()

    if distributed.is_enabled():
        # Aggregate sums across all processes
        torch.distributed.all_reduce(total_psnr_tensor)
        torch.distributed.all_reduce(total_ssim_tensor)
        torch.distributed.all_reduce(total_depth_rmse_tensor)
        torch.distributed.all_reduce(total_occupied_psnr_tensor)
        torch.distributed.all_reduce(total_occupied_ssim_tensor)
        torch.distributed.all_reduce(total_dynamic_psnr_tensor)
        torch.distributed.all_reduce(total_dynamic_ssim_tensor)
        torch.distributed.all_reduce(total_dynamic_rmse_tensor)
        torch.distributed.all_reduce(total_depth_rmse_0_100_tensor)
        torch.distributed.all_reduce(total_depth_rmse_100_200_tensor)
        torch.distributed.all_reduce(total_valid_depth_samples_0_100_tensor)
        torch.distributed.all_reduce(total_valid_depth_samples_100_200_tensor)
        torch.distributed.all_reduce(total_dynamic_depth_rmse_0_100_tensor)
        torch.distributed.all_reduce(total_dynamic_depth_rmse_100_200_tensor)
        torch.distributed.all_reduce(total_valid_dynamic_depth_samples_0_100_tensor)
        torch.distributed.all_reduce(total_valid_dynamic_depth_samples_100_200_tensor)
        torch.distributed.all_reduce(total_samples_tensor)
        torch.distributed.all_reduce(total_dynamic_samples_tensor)
        torch.distributed.all_reduce(total_valid_dynamic_depth_samples_tensor)

    result = None
    if distributed.is_main_process():
        avg_psnr = total_psnr_tensor.item() / total_samples_tensor.item()
        avg_ssim = total_ssim_tensor.item() / total_samples_tensor.item()
        avg_depth_rmse = total_depth_rmse_tensor.item() / total_samples_tensor.item()
        avg_occupied_psnr = total_occupied_psnr_tensor.item() / total_samples_tensor.item()
        avg_occupied_ssim = total_occupied_ssim_tensor.item() / total_samples_tensor.item()
        avg_dynamic_psnr = total_dynamic_psnr_tensor.item() / total_dynamic_samples_tensor.item()
        avg_dynamic_ssim = total_dynamic_ssim_tensor.item() / total_dynamic_samples_tensor.item()
        avg_dynamic_rmse = (
            total_dynamic_rmse_tensor.item() / total_valid_dynamic_depth_samples_tensor.item()
        )
        avg_depth_rmse_0_100 = (
            total_depth_rmse_0_100_tensor.item() / total_valid_depth_samples_0_100_tensor.item()
            if total_valid_depth_samples_0_100_tensor.item() > 0 else -1.0
        )
        avg_depth_rmse_100_200 = (
            total_depth_rmse_100_200_tensor.item() / total_valid_depth_samples_100_200_tensor.item()
            if total_valid_depth_samples_100_200_tensor.item() > 0 else -1.0
        )
        avg_dynamic_depth_rmse_0_100 = (
            total_dynamic_depth_rmse_0_100_tensor.item() / total_valid_dynamic_depth_samples_0_100_tensor.item()
            if total_valid_dynamic_depth_samples_0_100_tensor.item() > 0 else -1.0
        )
        avg_dynamic_depth_rmse_100_200 = (
            total_dynamic_depth_rmse_100_200_tensor.item() / total_valid_dynamic_depth_samples_100_200_tensor.item()
            if total_valid_dynamic_depth_samples_100_200_tensor.item() > 0 else -1.0
        )

        # Metric formatting helper function
        def format_metric(val):
            if isinstance(val, (int, float)):
                return f"{val:.4f}"
            return val

        with open(os.path.join(eval_result_dir, f"eval_{name_str}.txt"), "w") as f:
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Average Depth RMSE (0.01-100m): {format_metric(avg_depth_rmse_0_100)}\n")
            f.write(f"Average Depth RMSE (100-200m): {format_metric(avg_depth_rmse_100_200)}\n")
            f.write(f"Average Depth RMSE (0.01-200m): {avg_depth_rmse:.4f}\n")
            f.write(f"Average Occupied PSNR: {avg_occupied_psnr:.4f}\n")
            f.write(f"Average Occupied SSIM: {avg_occupied_ssim:.4f}\n")
            f.write(f"Average Dynamic PSNR: {avg_dynamic_psnr:.4f}\n")
            f.write(f"Average Dynamic SSIM: {avg_dynamic_ssim:.4f}\n")
            f.write(f"Average Dynamic Depth RMSE (0.01-100m): {format_metric(avg_dynamic_depth_rmse_0_100)}\n")
            f.write(f"Average Dynamic Depth RMSE (100-200m): {format_metric(avg_dynamic_depth_rmse_100_200)}\n")
            f.write(f"Average Dynamic Depth RMSE (0.01-200m): {avg_dynamic_rmse:.4f}\n")
            f.write(f"Evaluated on {total_samples_tensor.item()} samples.\n")
            f.write(f"Valid depth samples (0.01-100m): {total_valid_depth_samples_0_100_tensor.item()}\n")
            f.write(f"Valid depth samples (100-200m): {total_valid_depth_samples_100_200_tensor.item()}\n")
            f.write(f"Valid dynamic depth samples (0.01-100m): {total_valid_dynamic_depth_samples_0_100_tensor.item()}\n")
            f.write(f"Valid dynamic depth samples (100-200m): {total_valid_dynamic_depth_samples_100_200_tensor.item()}\n")

        logger.info("Evaluation results saved.")
        logger.info(f"Evaluated on {total_samples_tensor.item()} samples.")
        logger.info(
            f"Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}"
        )
        logger.info(
            f"Average Depth RMSE (0.01-100m): {format_metric(avg_depth_rmse_0_100)}, "
            f"Average Depth RMSE (100-200m): {format_metric(avg_depth_rmse_100_200)}"
        )
        logger.info(
            f"Average Depth RMSE (0.01-200m): {avg_depth_rmse:.4f}"
        )
        logger.info(
            f"Average Occupied PSNR: {avg_occupied_psnr:.4f}, Average Occupied SSIM: {avg_occupied_ssim:.4f}"
        )
        logger.info(
            f"Average Dynamic PSNR: {avg_dynamic_psnr:.4f}, Average Dynamic SSIM: {avg_dynamic_ssim:.4f}"
        )
        logger.info(
            f"Average Dynamic Depth RMSE (0.01-100m): {format_metric(avg_dynamic_depth_rmse_0_100)}, "
            f"Average Dynamic Depth RMSE (100-200m): {format_metric(avg_dynamic_depth_rmse_100_200)}"
        )
        logger.info(
            f"Average Dynamic Depth RMSE (0.01-200m): {avg_dynamic_rmse:.4f}"
        )

        result = {
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "depth_rmse_0_100m": avg_depth_rmse_0_100,
            "depth_rmse_100_200m": avg_depth_rmse_100_200,
            "depth_rmse": avg_depth_rmse,
            "occupied_psnr": avg_occupied_psnr,
            "occupied_ssim": avg_occupied_ssim,
            "dynamic_psnr": avg_dynamic_psnr,
            "dynamic_ssim": avg_dynamic_ssim,
            "dynamic_depth_rmse_0_100m": avg_dynamic_depth_rmse_0_100,
            "dynamic_depth_rmse_100_200m": avg_dynamic_depth_rmse_100_200,
            "dynamic_depth_rmse": avg_dynamic_rmse,
            "valid_depth_samples_0_100m": total_valid_depth_samples_0_100_tensor.item(),
            "valid_depth_samples_100_200m": total_valid_depth_samples_100_200_tensor.item(),
            "valid_dynamic_depth_samples_0_100m": total_valid_dynamic_depth_samples_0_100_tensor.item(),
            "valid_dynamic_depth_samples_100_200m": total_valid_dynamic_depth_samples_100_200_tensor.item(),
        }

    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def evaluate_flow(dataloader, model, args, name_str=None):
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    eval_result_dir = os.path.join(args.log_dir, "eval_results")
    os.makedirs(eval_result_dir, exist_ok=True)
    logger.info(f"Saving evaluation results to {eval_result_dir}")
    # use yr-mo-dy-hr-min
    if name_str is None:
        name_str = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    (
        total_flow_epes,
        total_flow_accs_strict,
        total_flow_accs_relax,
        total_flow_angles,
        total_flow_rmse,
        total_numb_flow_samples,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pbar = tqdm(dataloader, desc="Evaluating Flow")
    for data_dict in pbar:
        input_dict, target_dict = prepare_inputs_and_targets(data_dict, device, v=args.num_max_cameras)
        pred_dict = model(input_dict)
        # evaluate on real target images:
        # b, t, v, c, h, w
        b, t, v, height, width = target_dict["target_depth"].shape
        gt_depth = target_dict["target_depth"].view(b * t, -1, height, width)
        num_imgs = gt_depth.shape[0]
        valid_depth_mask = (gt_depth > 0.01) & (gt_depth < 200)
        rendered_results = pred_dict["render_results"]
        if args.load_ground and "target_ground_masks" in target_dict:  # TODO: driving sim: ground mask
            gt_ground_mask = target_dict["target_ground_masks"].view(b * t, -1, height, width)
            gt_ground_mask = gt_ground_mask.bool()
        num_valid_samples = 0
        eval_flow = (
            "rendered_flow" in rendered_results
            and args.decoder_type == "dummy"
            and args.load_flow
            and "target_flow" in target_dict
        )
        if eval_flow:
            gt_flow = target_dict["target_flow"].view(b * t, -1, height, width, 3)
            pred_flow = rendered_results["rendered_flow"].view(b * t, -1, height, width, 3)
            # pred_flow = gt_flow.clone() + torch.rand_like(gt_flow) * 0.05
            flow_epes, flow_accs_strict, flow_accs_relax, flow_angles = [], [], [], []
            flow_rmse = []

            for i in range(1, num_imgs - 1):  # num_imgs - 1: do not evaluate the last frame (it is a future frame prediction)
                if torch.max(gt_flow[i].norm(dim=-1)) > 1.0:
                    if args.load_ground and "target_ground_masks" in target_dict:  # TODO: driving sim: ground mask
                        non_ground_gt_flow = gt_flow[i][~gt_ground_mask[i] & valid_depth_mask[i]]
                        non_ground_pred_flow = pred_flow[i][
                            ~gt_ground_mask[i] & valid_depth_mask[i]
                        ]
                    else:
                        non_ground_gt_flow = gt_flow[i][valid_depth_mask[i]]
                        non_ground_pred_flow = pred_flow[i][valid_depth_mask[i]]
                    flow_metrics = compute_scene_flow_metrics(
                        non_ground_pred_flow, non_ground_gt_flow
                    )
                    flow_epes.append(flow_metrics["EPE3D"])
                    flow_accs_strict.append(flow_metrics["acc3d_strict"] * 100)
                    flow_accs_relax.append(flow_metrics["acc3d_relax"] * 100)
                    flow_angles.append(flow_metrics["angle_error"])
                    flow_rmse.append(
                        torch.sqrt(
                            F.mse_loss(
                                pred_flow[i][valid_depth_mask[i]],
                                gt_flow[i][valid_depth_mask[i]],
                            )
                        ).item()
                    )
                    num_valid_samples += 1

            flow_epe_sum = np.sum(flow_epes)
            flow_acc_strict_sum = np.sum(flow_accs_strict)
            flow_acc_relax_sum = np.sum(flow_accs_relax)
            flow_angle_sum = np.sum(flow_angles)
            flow_rmse_sum = np.sum(flow_rmse)
            valid_flow_samples = num_valid_samples

            # Update running sums and counts
            total_flow_epes += flow_epe_sum
            total_flow_accs_strict += flow_acc_strict_sum
            total_flow_accs_relax += flow_acc_relax_sum
            total_flow_angles += flow_angle_sum
            total_flow_rmse += flow_rmse_sum
            total_numb_flow_samples += valid_flow_samples

        pbar.set_postfix(
            avg_flow_epe=total_flow_epes / total_numb_flow_samples,
            avg_flow_acc_relax=total_flow_accs_relax / total_numb_flow_samples,
            avg_flow_acc_strict=total_flow_accs_strict / total_numb_flow_samples,
            avg_flow_angle=total_flow_angles / total_numb_flow_samples,
            avg_flow_rmse=total_flow_rmse / total_numb_flow_samples,
        )


    # Create tensors for sums and counts
    result = None
    total_flow_epes_tensor = torch.tensor(total_flow_epes, dtype=torch.float32, device=device)
    total_flow_accs_strict_tensor = torch.tensor(total_flow_accs_strict, dtype=torch.float32, device=device)
    total_flow_accs_relax_tensor = torch.tensor(total_flow_accs_relax, dtype=torch.float32, device=device)
    total_flow_angles_tensor = torch.tensor(total_flow_angles, dtype=torch.float32, device=device)
    total_flow_rmse_tensor = torch.tensor(total_flow_rmse, dtype=torch.float32, device=device)
    total_numb_flow_samples_tensor = torch.tensor(total_numb_flow_samples, device=device)  # dtype float32

    torch.cuda.synchronize()

    if distributed.is_enabled():
        # Aggregate sums across all processes
        torch.distributed.all_reduce(total_flow_epes_tensor)
        torch.distributed.all_reduce(total_flow_accs_strict_tensor)
        torch.distributed.all_reduce(total_flow_accs_relax_tensor)
        torch.distributed.all_reduce(total_flow_angles_tensor)
        torch.distributed.all_reduce(total_flow_rmse_tensor)
        torch.distributed.all_reduce(total_numb_flow_samples_tensor)
    if distributed.is_main_process() and total_numb_flow_samples_tensor.item() > 0:
        avg_flow_epe = total_flow_epes_tensor.item() / total_numb_flow_samples_tensor.item()
        avg_flow_acc_strict = (
            total_flow_accs_strict_tensor.item() / total_numb_flow_samples_tensor.item()
        )
        avg_flow_acc_relax = (
            total_flow_accs_relax_tensor.item() / total_numb_flow_samples_tensor.item()
        )
        avg_flow_angle = total_flow_angles_tensor.item() / total_numb_flow_samples_tensor.item()
        avg_flow_rmse = total_flow_rmse_tensor.item() / total_numb_flow_samples_tensor.item()
        with open(os.path.join(eval_result_dir, f"eval_{name_str}_flow.txt"), "w") as f:
            f.write(f"Average Flow EPE: {avg_flow_epe:.4f}\n")
            f.write(f"Average Flow Acc Strict: {avg_flow_acc_strict:.4f}\n")
            f.write(f"Average Flow Acc Relax: {avg_flow_acc_relax:.4f}\n")
            f.write(f"Average Flow Angle: {avg_flow_angle:.4f}\n")
            f.write(f"Average Flow RMSE: {avg_flow_rmse:.4f}\n")
            f.write(f"Evaluated on {total_numb_flow_samples_tensor.item()} samples.\n")
        logger.info("Evaluation results saved.")
        logger.info(f"Evaluated on {total_numb_flow_samples_tensor.item()} samples.")
        logger.info(
            f"Average Flow EPE: {avg_flow_epe:.4f}, Average Flow Acc Strict: {avg_flow_acc_strict:.4f}, Average Flow Acc Relax: {avg_flow_acc_relax:.4f}, Average Flow Angle: {avg_flow_angle:.4f}"
        )
        logger.info(f"Average Flow RMSE: {avg_flow_rmse:.4f}")
        result = {
            "flow_epe": avg_flow_epe,
            "flow_acc_strict": avg_flow_acc_strict,
            "flow_acc_relax": avg_flow_acc_relax,
            "flow_angle": avg_flow_angle,
            "flow_rmse": avg_flow_rmse,
        }
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def evaluate_semantic(dataloader, model, args, name_str=None, feat_extractor=None):
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    eval_result_dir = os.path.join(args.log_dir, "eval_results")
    os.makedirs(eval_result_dir, exist_ok=True)
    logger.info(f"Saving evaluation results to {eval_result_dir}")

    # use yr-mo-dy-hr-min
    if name_str is None:
        name_str = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    (
        total_semantic_miou,
        total_semantic_acc,
        total_numb_semantic_samples,
    ) = (0.0, 0.0, 0.0)

    pbar = tqdm(dataloader, desc="Evaluating")
    for data_dict in pbar:
        input_dict, target_dict = prepare_inputs_and_targets(data_dict, device, v=args.num_max_cameras, feat_extractor=feat_extractor)
        # print(target_dict['target_frame_idx'])
        pred_dict = model(input_dict)
        # evaluate on real target images:
        # b, t, v, c, h, w
        b, t, v, height, width = target_dict["target_depth"].shape
        gt_depth = target_dict["target_depth"].view(b * t, -1, height, width)
        num_imgs = gt_depth.shape[0]

        rendered_results = pred_dict["render_results"]

        num_valid_samples = 0
        eval_semantic = (
            ("rendered_feat" in rendered_results if not os.getenv("CONTEXT_FEAT") else "rendered_semantic" in rendered_results)
            and "target_semantic_labels" in target_dict
        )
        if eval_semantic:
            gt_semantic = target_dict["target_semantic_labels"].view(b * t, -1, height, width).long()
            if not os.getenv("CONTEXT_FEAT"):
                pred_feat = rendered_results['rendered_feat']
                # 512feat -> semantic
                pred_feat = einops.rearrange(pred_feat, "b t v h w c -> (b t v h w) c")

                # Using the results of the distillation model, such as LSeg
                # pred_feat = einops.rearrange(target_dict['target_feat'], "b t v c h w -> (b t v h w) c").float()
                # print('Using Lseg')

                pred_semantic = feat2class(pred_feat, get_text_label_feats(SEMANTIC_LABEL_LIST), args.similarity_probs_threshold)
                pred_semantic = pred_semantic.view(b * t, -1, height, width)
            else:
                pred_semantic = rendered_results['rendered_semantic'].view(b * t, -1, height, width)

            semantic_miou, semantic_acc = [], []

            for i in range(num_imgs):
                if gt_semantic[i].max() > 0:
                    semantic_metrics = compute_semantic_metrics(
                        pred_semantic[i], gt_semantic[i],
                        # orig_img=target_dict['target_image'][0][i]
                    )
                    semantic_miou.append(semantic_metrics["MIOU"])
                    semantic_acc.append(semantic_metrics["ACC"])
                    num_valid_samples += 1
                    # print("['scene_id']", input_dict['scene_id'], target_dict['target_frame_idx'][0][i],
                    #       semantic_metrics["MIOU"], semantic_metrics["ACC"])

            semantic_miou_sum = np.sum(semantic_miou)
            semantic_acc_sum = np.sum(semantic_acc)
            valid_semantic_samples = num_valid_samples

            # Update running sums and counts
            total_semantic_miou += semantic_miou_sum
            total_semantic_acc += semantic_acc_sum
            total_numb_semantic_samples += valid_semantic_samples

        if total_numb_semantic_samples > 0:
            pbar.set_postfix(
                avg_semantic_miou=total_semantic_miou / total_numb_semantic_samples,
                avg_semantic_acc=total_semantic_acc / total_numb_semantic_samples,
            )

    # Create tensors for sums and counts
    result = None
    total_semantic_miou_tensor = torch.tensor(total_semantic_miou, dtype=torch.float32, device=device)
    total_semantic_acc_tensor = torch.tensor(total_semantic_acc, dtype=torch.float32, device=device)
    total_numb_semantic_samples_tensor = torch.tensor(total_numb_semantic_samples, device=device)  # dtype float32

    # print(f"from GPU {torch.cuda.current_device()}", total_numb_semantic_samples_tensor, total_semantic_miou_tensor, total_semantic_acc_tensor flush=True)
    torch.cuda.synchronize()

    if distributed.is_enabled():
        # Aggregate sums across all processes
        torch.distributed.all_reduce(total_semantic_miou_tensor)
        torch.distributed.all_reduce(total_semantic_acc_tensor)
        torch.distributed.all_reduce(total_numb_semantic_samples_tensor)
    if distributed.is_main_process() and total_numb_semantic_samples_tensor.item() > 0:
        avg_semantic_miou = total_semantic_miou_tensor.item() / total_numb_semantic_samples_tensor.item()
        avg_semantic_acc = total_semantic_acc_tensor.item() / total_numb_semantic_samples_tensor.item()
        with open(os.path.join(eval_result_dir, f"eval_{name_str}_semantic.txt"), "w") as f:
            f.write(f"Average Semantic mIOU: {avg_semantic_miou:.4f}\n")
            f.write(f"Average Semantic Accuracy: {avg_semantic_acc:.4f}\n")
            f.write(f"Evaluated on {total_numb_semantic_samples_tensor.item()} samples.\n")
        logger.info("Evaluation results saved.")
        logger.info(f"Evaluated on {total_numb_semantic_samples_tensor.item()} samples.")
        logger.info(
            f"Average Semantic mIOU: {avg_semantic_miou:.4f}, Average Semantic Accuracy: {avg_semantic_acc:.4f}"
        )
        result = {
            "semantic_miou": avg_semantic_miou,
            "semantic_acc": avg_semantic_acc
        }
    torch.cuda.empty_cache()
    return result

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import glob
import imageio
import numpy as np
import cv2
import torch
from typing import Optional
from depth_anything_3.api import DepthAnything3


def min_max_scale(depth_map):
    # Find the max and min values of the depth map
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)

    # Normalize depth map to 0-1
    if max_val > min_val:
        depth_map_normalized = (depth_map - min_val) / (max_val - min_val)
    else:
        # If max equals min, set to 0 to avoid division by zero
        depth_map_normalized = np.zeros_like(depth_map)

    return depth_map_normalized

def save_depth_image(depth_map, valid, output_path, need_reverse=True):
    depth_map_normalized = min_max_scale(depth_map)
    if need_reverse:
        depth_map_normalized = 1 - depth_map_normalized

    if valid is not None:
        depth_map_normalized[~valid] = 0.0

    # Convert to suitable format for saving, e.g., 8-bit unsigned integer
    depth_map_normalized_u8 = (255.0 * depth_map_normalized).astype(np.uint8)

    # Save normalized depth map using imageio
    imageio.imwrite(output_path, depth_map_normalized_u8)

    return depth_map_normalized

def align_relative_depth_ransac(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask_valid: Optional[torch.Tensor] = None,
    ransac_iter: int = 1000,
    inlier_threshold: float = 0.1,  # in meters (or same unit as gt)
    min_valid_points: int = 10,
    final_least_squares: bool = True,
):
    """
    Align a relative depth map to sparse GT depth using RANSAC.

    Args:
        pred_depth (H, W): predicted relative depth (>=0, arbitrary scale)
        gt_depth (H, W): ground truth depth (>0 where valid)
        mask_valid (H, W): optional boolean mask of valid pixels
        ransac_iter: number of RANSAC iterations
        inlier_threshold: residual threshold to count as inlier
        min_valid_points: skip if fewer valid points
        final_least_squares: if True, refit model using all inliers

    Returns:
        scale (float), shift (float), inlier_mask (H, W, bool)
    """
    device = pred_depth.device
    H, W = pred_depth.shape

    # Create valid mask
    if mask_valid is None:
        mask_valid = (gt_depth > 0) & (pred_depth > 0)
    else:
        mask_valid = mask_valid & (gt_depth > 0) & (pred_depth > 0)

    valid_indices = torch.nonzero(mask_valid, as_tuple=False)  # (N, 2)
    N = valid_indices.shape[0]

    if N < 2:
        raise ValueError("Not enough valid points for alignment.")
    if N < min_valid_points:
        # Fallback: use least squares without RANSAC
        pred_vals = pred_depth[mask_valid].cpu().numpy()
        gt_vals = gt_depth[mask_valid].cpu().numpy()
        A = np.vstack([pred_vals, np.ones_like(pred_vals)]).T
        try:
            sol, _, _, _ = np.linalg.lstsq(A, gt_vals, rcond=None)
            scale, shift = sol[0], sol[1]
            inlier_mask = mask_valid.clone()
            return float(scale), float(shift), inlier_mask
        except np.linalg.LinAlgError:
            return 1.0, 0.0, mask_valid

    pred_vals = pred_depth[mask_valid].cpu().numpy()  # (N,)
    gt_vals = gt_depth[mask_valid].cpu().numpy()      # (N,)

    best_inlier_count = -1
    best_scale, best_shift = 1.0, 0.0
    best_inlier_mask_flat = np.zeros(N, dtype=bool)

    for _ in range(ransac_iter):
        # Randomly pick 2 distinct indices
        idx = np.random.choice(N, size=2, replace=False)
        p1, p2 = pred_vals[idx]
        g1, g2 = gt_vals[idx]

        # Skip if pred values are too close (ill-conditioned)
        if abs(p2 - p1) < 1e-8:
            continue

        # Solve: g = s * p + t
        s = (g2 - g1) / (p2 - p1)
        t = g1 - s * p1

        # Compute residuals
        residuals = np.abs(gt_vals - (s * pred_vals + t))
        inliers = residuals < inlier_threshold
        inlier_count = np.sum(inliers)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_scale, best_shift = s, t
            best_inlier_mask_flat = inliers

    # Optional: refine with least squares on inliers
    if final_least_squares and best_inlier_count >= 2:
        inlier_pred = pred_vals[best_inlier_mask_flat]
        inlier_gt = gt_vals[best_inlier_mask_flat]
        A = np.vstack([inlier_pred, np.ones_like(inlier_pred)]).T
        try:
            sol, _, _, _ = np.linalg.lstsq(A, inlier_gt, rcond=None)
            best_scale, best_shift = sol[0], sol[1]
        except np.linalg.LinAlgError:
            pass  # keep RANSAC estimate

    # Build full inlier mask (H, W)
    inlier_mask = torch.zeros_like(mask_valid)
    flat_inlier_indices = valid_indices[best_inlier_mask_flat]
    if flat_inlier_indices.numel() > 0:
        inlier_mask[flat_inlier_indices[:, 0], flat_inlier_indices[:, 1]] = True

    return float(best_scale), float(best_shift), inlier_mask


if __name__ == "__main__":
    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE") # "depth-anything/da3mono-large"
    model = model.to(device=device)
    root_directory = "xxx/SLARM_data/datasets/waymo/training"
    for folder in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder)
        if not os.path.isdir(folder_path):
            continue
        example_path = os.path.join(root_directory, folder, "images")
        gt_depth_path = os.path.join(root_directory, folder, "depth_flows_4")

        images = sorted(glob.glob(os.path.join(example_path, "*.jpg")))
        gt_depths = sorted(glob.glob(os.path.join(gt_depth_path, "*.npy")))
        pseudo_path = os.path.join(root_directory, folder, "pseudo_depth_4")
        os.makedirs(pseudo_path, exist_ok=True)

        # # Ensure there is a place to store output results
        # output_dir = "depth_output_089"
        # os.makedirs(output_dir, exist_ok=True)

        # print(len(images))
        # print(len(gt_depths))

        for i, (image_path, gt_depth_path) in enumerate(zip(images, gt_depths)):
            img_name = os.path.splitext(os.path.basename(image_path))[0]
            depth_name = os.path.splitext(os.path.basename(gt_depth_path))[0]
            assert img_name == depth_name
            # load gt depth
            depth_and_flow = np.load(gt_depth_path)
            gt_depth = depth_and_flow[..., 0]
            valid = gt_depth > 0.0

            prediction = model.inference([image_path])
            depth_map = prediction.depth[0]
            conf_map = prediction.conf[0]

            # resize
            target_size = (gt_depth.shape[1], gt_depth.shape[0])
            depth_map = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_LINEAR)
            conf_map = cv2.resize(conf_map, target_size, interpolation=cv2.INTER_LINEAR)

            # print(conf_map.mean(), conf_map.min(), conf_map.max())

            if i < 10 and False:
                os.makedirs(os.path.join(output_dir, img_name), exist_ok=True)
                depth_map_normalized = save_depth_image(depth_map, None, os.path.join(output_dir, img_name, f"normalized_depth_{img_name}.png"))
                depth_map_normalized = save_depth_image(depth_map, valid, os.path.join(output_dir, img_name, f"normalized_depth_valid_{img_name}.png"))
                gt_depth_normalized = save_depth_image(gt_depth, valid, os.path.join(output_dir, img_name, f"normalized_gt_depth_{img_name}.png"))
                if conf_map is not None:
                    conf_map_normalized = save_depth_image(conf_map, None, os.path.join(output_dir, img_name, f"normalized_conf_map_{img_name}.png"), need_reverse=False)

                delta_map_normalized = abs(depth_map_normalized - gt_depth_normalized)
                delta_map_normalized_u8 = (255.0 * delta_map_normalized).astype(np.uint8)
                imageio.imwrite(os.path.join(output_dir, img_name, f"normalized_delta_depth_{img_name}.png"), delta_map_normalized_u8)

            # to tensor
            pred_depth = torch.from_numpy(depth_map).float().to(device)
            gt_depth = torch.from_numpy(gt_depth).float().to(device)

            # calculate scale and shift
            scale, shift, inlier_mask = align_relative_depth_ransac(
                pred_depth,
                gt_depth,
                ransac_iter=5000,
                inlier_threshold=0.5,  # 5cm tolerance
            )

            # print(scale, shift)

            # Apply alignment and save
            aligned_depth = scale * pred_depth + shift
            np.save(os.path.join(pseudo_path, f"{img_name}.npy"), aligned_depth.cpu().numpy())
            if conf_map is not None:
                conf_map_normalized = min_max_scale(conf_map)
                np.save(os.path.join(pseudo_path, f"{img_name}_conf.npy"), conf_map_normalized)

            check_save_valid = False
            if check_save_valid:
                new_align = np.load(os.path.join(pseudo_path, f"{img_name}.npy"))
                print(abs(aligned_depth.cpu().numpy() - new_align).mean())

            # Optional: evaluate only on inliers
            # print(valid.sum(), inlier_mask.sum())
            # valid_gt = gt_depth[inlier_mask]
            # valid_aligned = aligned_depth[inlier_mask]
            # rmse = torch.sqrt(((valid_gt - valid_aligned) ** 2).mean())
            # print(rmse)

            # aligned_depth_normalized = save_depth_image(aligned_depth.cpu().numpy(), valid, os.path.join(output_dir, f"normalized_align_depth_{i}.png"))
            # print((aligned_depth_normalized-depth_map_normalized).sum())

        print(f"Pseudo depth maps have been saved to {pseudo_path}")

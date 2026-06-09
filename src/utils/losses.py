import logging
import os
from math import ceil, floor

import torch
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

from src.dataset.constants import MEAN, STD, SEMANTIC_LABEL_LIST
from ..models.components.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
if os.getenv("FEAT_DIST"):
    from tools.feats_tools import get_text_label_feats, feat2class


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.

    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.

    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization

    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.

    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.

    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.

    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def depth_loss(depth, depth_conf, batch, gamma=1.0, alpha=0.2, loss_type="conf", predict_disparity=False, affine_inv=False, gradient_loss=None, valid_range=-1, disable_conf=False, all_mean=False, **kwargs):

    gt_depth = batch['context_depth'].clone()

    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")

    gt_depth = rearrange(gt_depth, 'b t v ... -> b (t v) ... 1')
    valid_mask = gt_depth > 0.0

    depth_conf = depth_conf[..., None]

    if loss_type == "conf":
        conf_loss_dict = conf_loss(depth, depth_conf, gt_depth, valid_mask,
                               batch, normalize_pred=False, normalize_gt=False,
                               gamma=gamma, alpha=alpha, affine_inv=affine_inv, gradient_loss=gradient_loss, valid_range=valid_range, postfix="_depth", disable_conf=disable_conf, all_mean=all_mean)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return conf_loss_dict


def point_loss(pts3d, pts3d_conf, batch, normalize_gt=True, normalize_pred=True, gamma=1.0, alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False, conf_loss_type="v1", **kwargs):
    """
    pts3d: B, S, H, W, 3
    pts3d_conf: B, S, H, W
    """
    # gt_pts3d: B, S, H, W, 3
    gt_pts3d = batch['context_pts3d'].clone()
    gt_pts3d = rearrange(gt_pts3d, 'b t v ... -> b (t v) ...')
    gt_pts3d = check_and_fix_inf_nan(gt_pts3d, "gt_pts3d")

    # valid_mask: B, S, H, W
    valid_mask = batch['context_valid_masks'].bool()
    valid_mask = rearrange(valid_mask, 'b t v ... -> b (t v) ...')

    if conf_loss_type == "v1":
        conf_loss_fn = conf_loss
    else:
        raise ValueError(f"Invalid conf loss type: {conf_loss_type}")

    conf_loss_dict = conf_loss_fn(pts3d, pts3d_conf, gt_pts3d, valid_mask,
                                batch, normalize_gt=normalize_gt, normalize_pred=normalize_pred, gamma=gamma, alpha=alpha, affine_inv=affine_inv,
                                gradient_loss=gradient_loss, valid_range=valid_range, camera_centric_reg=camera_centric_reg, disable_conf=disable_conf, all_mean=all_mean)


    return conf_loss_dict


def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.

    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside
                                  [-hard_max, hard_max] will be clamped. If None,
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor

    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.

    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.

    This helps remove outliers that could destabilize training.

    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss

    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.

    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss

    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering

    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # NOTE: Unnormalized scene values are too large, need to divide by max
    gt_max = gt.max()

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] / gt_max - pred[mask] / gt_max, dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")

    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def compute_depth_loss_with_conf(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute depth loss.

    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['pred_context_depth']
    pred_depth_conf = predictions['pred_context_depth_conf']

    gt_depth = batch['context_depth']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth_mask = batch['context_valid_masks'].clone()   # 3D points derived from depth map, so we use the same mask

    # rearrange
    pred_depth = rearrange(pred_depth, 'b t v ... -> b (t v) ... 1')
    gt_depth = rearrange(gt_depth, 'b t v ... -> b (t v) ... 1')
    pred_depth_conf = rearrange(pred_depth_conf, 'b t v ... -> b (t v) ...')
    gt_depth_mask = rearrange(gt_depth_mask, 'b t v ... -> b (t v) ...')
    gt_depth_mask = gt_depth_mask.to(torch.bool)

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict


def compute_point_loss_with_conf(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.

    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['pred_context_pts3d']
    pred_points_conf = predictions['pred_context_pts3d_conf']
    gt_points = batch['context_pts3d']
    gt_points_mask = batch['context_valid_masks']

    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")

    # rearrange
    pred_points = rearrange(pred_points, 'b t v ... -> b (t v) ...')
    gt_points = rearrange(gt_points, 'b t v ... -> b (t v) ...')
    pred_points_conf = rearrange(pred_points_conf, 'b t v ... -> b (t v) ...')
    gt_points_mask = rearrange(gt_points_mask, 'b t v ... -> b (t v) ...')
    gt_points_mask = gt_points_mask.to(torch.bool)

    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict

    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }

    return loss_dict


def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.

    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)

    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pred_context_camera_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = rearrange(batch_data['context_valid_masks'], 'b t v ... -> b (t v) ...')
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = rearrange(batch_data['context_camtoworlds'], 'b t v ... -> b (t v) ...').inverse()[..., :3, :4]  # extrinsic is world to camera [b, s, 3, 4], intrinsic (no normalization) [b, s, 3, 3]
    gt_intrinsics = rearrange(batch_data['context_intrinsics'], 'b t v ... -> b (t v) ...')
    image_hw = batch_data['context_image'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }


def compute_depth_loss(pred_depth, gt_depth, max_depth=None):
    pred_depth = pred_depth.squeeze()
    gt_depth = gt_depth.squeeze()
    if pred_depth.shape != gt_depth.shape:
        # resize pred_depth to match gt depth size
        try:
            b, v, h, w = pred_depth.shape
            gt_h, gt_w = gt_depth.shape[-2:]
            pred_depth = F.interpolate(
                rearrange(pred_depth, "b v h w -> (b v) 1 h w"),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            )
            pred_depth = rearrange(pred_depth, "(b v) 1 h w -> b v h w", b=b, v=v)
        except:
            b, t, v, h, w = pred_depth.shape
            gt_h, gt_w = gt_depth.shape[-2:]
            pred_depth = F.interpolate(
                rearrange(pred_depth, "b t v h w -> (b t v) 1 h w"),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            )
            pred_depth = rearrange(pred_depth, "(b t v) 1 h w -> b t v h w", b=b, t=t, v=v)

    valid_mask = (gt_depth > 0.01) & (gt_depth < 200)
    if max_depth is None:
        max_depth = gt_depth[valid_mask].max()
    pred_depth = pred_depth[valid_mask] / max_depth
    gt_depth = gt_depth[valid_mask] / max_depth
    return F.l1_loss(pred_depth, gt_depth)

def compute_pseudo_depth_loss(pred_depth, gt_depth, depth_conf):
    pred_depth = pred_depth.squeeze()
    gt_depth = gt_depth.squeeze()
    if pred_depth.shape != gt_depth.shape:
        # resize pred_depth to match gt depth size
        try:
            b, v, h, w = pred_depth.shape
            gt_h, gt_w = gt_depth.shape[-2:]
            pred_depth = F.interpolate(
                rearrange(pred_depth, "b v h w -> (b v) 1 h w"),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            )
            pred_depth = rearrange(pred_depth, "(b v) 1 h w -> b v h w", b=b, v=v)
        except:
            b, t, v, h, w = pred_depth.shape
            gt_h, gt_w = gt_depth.shape[-2:]
            pred_depth = F.interpolate(
                rearrange(pred_depth, "b t v h w -> (b t v) 1 h w"),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            )
            pred_depth = rearrange(pred_depth, "(b t v) 1 h w -> b t v h w", b=b, t=t, v=v)

    max_depth = gt_depth.max()
    pred_depth = pred_depth / max_depth
    gt_depth = gt_depth / max_depth

    valid_mask = (depth_conf > 0.5).float()
    l1_error = (pred_depth - gt_depth).abs()
    weighted_loss = valid_mask * depth_conf * l1_error
    loss = weighted_loss.sum() / (valid_mask.sum() + 1e-8)

    return loss

def adaptive_confidence(gt_depth, pseudo_depth, alpha=0.1, valid_min=0.01, high_conf=0.8, low_conf=0.2, no_gt_conf=1.0):
    """
    Adaptive confidence map generation.

    Args:
        gt_depth: Sparse ground-truth depth map (H, W) or (B, H, W)
        pseudo_depth: Dense pseudo depth map, same shape
        alpha: Relative error tolerance ratio, e.g., 0.1 means 10%
        valid_min: Minimum value to determine valid gt_depth
        high_conf / low_conf: High/low confidence values when ground truth is available
        no_gt_conf: Confidence value for regions without ground truth (adjust as needed)
    """
    delta_depth = (gt_depth - pseudo_depth).abs()

    # Find valid regions
    valid_mask = gt_depth > valid_min  # (..., H, W)

    if valid_mask.any():
        # Extract valid gt depth values
        valid_gt_vals = gt_depth[valid_mask]
        # Use median for robustness (can also use mean)
        median_depth = torch.median(valid_gt_vals)
        # Adaptive threshold
        threshold = alpha * median_depth.clamp(min=0.1)  # Prevent too small values (e.g., close to 0)
    else:
        # If no valid ground truth, fall back to fixed threshold or set all to no_gt_conf
        threshold = torch.tensor(2.0, device=gt_depth.device)

    # Assign confidence in valid regions based on adaptive threshold
    conf_in_valid = torch.where(delta_depth < threshold, high_conf, low_conf)

    # Merge: use adaptive confidence in valid regions, default value in invalid regions
    delta_conf = torch.where(valid_mask, conf_in_valid, no_gt_conf)

    return delta_conf

def compute_pseudo_depth_loss_v2(pred_depth, pseudo_depth, gt_depth, depth_conf):
    pred_depth = pred_depth.squeeze()
    pseudo_depth = pseudo_depth.squeeze()
    if pred_depth.shape != pseudo_depth.shape:
        # resize pred_depth to match gt depth size
        try:
            b, v, h, w = pred_depth.shape
            gt_h, gt_w = pseudo_depth.shape[-2:]
            pred_depth = F.interpolate(
                rearrange(pred_depth, "b v h w -> (b v) 1 h w"),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            )
            pred_depth = rearrange(pred_depth, "(b v) 1 h w -> b v h w", b=b, v=v)
        except:
            b, t, v, h, w = pred_depth.shape
            gt_h, gt_w = pseudo_depth.shape[-2:]
            pred_depth = F.interpolate(
                rearrange(pred_depth, "b t v h w -> (b t v) 1 h w"),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            )
            pred_depth = rearrange(pred_depth, "(b t v) 1 h w -> b t v h w", b=b, t=t, v=v)

    delta_conf = adaptive_confidence(gt_depth, pseudo_depth, alpha=0.15)

    max_depth = pseudo_depth.max()
    pred_depth = pred_depth / max_depth
    pseudo_depth = pseudo_depth / max_depth

    valid_mask = (depth_conf > 0.5).float()
    l1_error = (pred_depth - pseudo_depth).abs()
    weighted_loss = valid_mask * depth_conf * delta_conf * l1_error
    loss = weighted_loss.sum() / (valid_mask.sum() + 1e-8)

    return loss


def compute_sky_depth_loss(pred_depth, gt_sky_mask, sky_depth: float = 1e3, flow=None):
    pred_depth = pred_depth.squeeze()
    gt_sky_mask = gt_sky_mask.squeeze()
    gt_h, gt_w = gt_sky_mask.shape[-2:]
    if pred_depth.shape != gt_sky_mask.shape:
        # resize pred_depth to match gt depth size
        b, t, v, h, w = pred_depth.shape
        pred_depth = F.interpolate(
            rearrange(pred_depth, "b t v h w -> (b t v) 1 h w"),
            size=(gt_h, gt_w),
            mode="bilinear",
            align_corners=False,
        )
        pred_depth = rearrange(pred_depth, "(b t v) 1 h w -> b t v h w", b=b, t=t, v=v)
    if flow is not None and (flow.shape[-3] != gt_h or flow.shape[-2] != gt_w):
        flow = F.interpolate(
            rearrange(flow, "b t v h w c -> (b t v) c h w"),
            size=(gt_h, gt_w),
            mode="bilinear",
            align_corners=False,
        )
        flow = rearrange(flow, "(b t v) c h w -> b t v h w c", b=b, t=t, v=v)
        # penalize flow in sky region
        sky_flow = flow[gt_sky_mask > 0]
        sky_flow_reg_loss = F.mse_loss(sky_flow, torch.zeros_like(sky_flow))
    else:
        sky_flow_reg_loss = torch.tensor(0.0).to(pred_depth.device)

    sky_region = gt_sky_mask > 0
    pred_depth = pred_depth[sky_region]
    return (
        F.mse_loss(pred_depth / sky_depth, torch.ones_like(pred_depth)) * 0.01,
        sky_flow_reg_loss,
    )


def compute_loss(output_dict, input_dict, target_dict, args=None, lpips_loss=None, data_iter_step=None):
    gs_params, pred_dict = output_dict["gs_params"], output_dict["render_results"]
    device = pred_dict[pred_dict["rgb_key"]].device
    mean, std = torch.tensor(MEAN).to(device), torch.tensor(STD).to(device)
    # pred_rgb = pred_dict[pred_dict["rgb_key"]] * std + mean
    # target_rgb = rearrange(target_dict["target_image"], "b t v c h w -> b t v h w c") * std + mean
    # NOTE: No normalization
    pred_rgb = pred_dict[pred_dict["rgb_key"]]
    target_rgb = rearrange(target_dict["target_image"], "b t v c h w -> b t v h w c")

    loss_dict = {}

    # context rgb loss
    if args.enable_context_rgb_loss and "rendered_context_image" in output_dict:
        # pred_context_rgb = output_dict["rendered_context_image"] * std + mean
        # target_context_rgb = rearrange(input_dict["context_image"], "b t v c h w -> b t v h w c") * std + mean
        # NOTE: No normalization
        pred_context_rgb = output_dict["rendered_context_image"]
        target_context_rgb = rearrange(input_dict["context_image"], "b t v c h w -> b t v h w c")

        # rendering loss
        if lpips_loss is not None:
            context_lpips_loss = lpips_loss(pred_context_rgb, target_context_rgb)
            for k, v in context_lpips_loss.items():
                loss_dict[f'context_{k}'] = args.context_rgb_loss_coeff * v
        else:
            context_rgb_loss = F.mse_loss(pred_context_rgb, target_context_rgb)
            loss_dict["context_rgb_loss"] = args.context_rgb_loss_coeff * context_rgb_loss
        loss_dict["context_rgb_loss"] *= args.rgb_loss_coeff

    # context sky opacity loss
    if args.enable_context_sky_opacity_loss and "context_sky_masks" in input_dict and "rendered_context_alpha" in output_dict:
        context_opacity = output_dict["rendered_context_alpha"].squeeze(-1)
        b, t, v, h, w = context_opacity.shape
        gt_h, gt_w = input_dict["context_sky_masks"].shape[-2:]
        if h != gt_h or w != gt_w:
            context_opacity = F.interpolate(
                rearrange(context_opacity, "b t v h w -> (b t v) 1 h w"),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            )
            context_opacity = rearrange(context_opacity, "(b t v) 1 h w -> b t v h w", b=b, t=t, v=v)
        context_sky_opacity_loss = F.l1_loss(context_opacity, 1 - input_dict["context_sky_masks"])
        loss_dict["context_sky_opacity_loss"] = context_sky_opacity_loss * args.context_sky_opacity_loss_coeff

    # context depth loss
    if args.enable_context_depth_loss and "pred_context_depth" in output_dict and "context_depth" in input_dict:
        if args.context_depth_loss_with_conf:
            # conf loss
            context_depth_loss_dict = compute_depth_loss_with_conf(output_dict, input_dict, gradient_loss_fn='grad', valid_range=0.98)
            context_depth_loss = context_depth_loss_dict["loss_conf_depth"] + context_depth_loss_dict["loss_reg_depth"] + context_depth_loss_dict["loss_grad_depth"]
            # TODO: Consider adding weights, it's a bit large at the beginning
            # TODO: Loss can be negative, though it seems fine, it might affect rendering loss
            context_depth_loss = args.context_depth_loss_coeff * context_depth_loss
            loss_dict["context_depth_loss"] = context_depth_loss
        else:
            # only depth loss
            pred_context_depth = output_dict["pred_context_depth"]
            target_context_depth = input_dict["context_depth"]
            context_depth_loss = compute_depth_loss(pred_context_depth, target_context_depth)
            context_depth_loss = args.context_depth_loss_coeff * context_depth_loss
            loss_dict["context_depth_loss"] = context_depth_loss

    # context point loss
    if args.enable_context_point_loss and "pred_context_pts3d" in output_dict and "context_pts3d" in input_dict:
        if args.context_point_loss_with_conf:
            # conf loss
            context_point_loss_dict = compute_point_loss_with_conf(output_dict, input_dict, gradient_loss_fn='normal', valid_range=0.98)
            context_point_loss = context_point_loss_dict["loss_conf_point"] + context_point_loss_dict["loss_reg_point"] + context_point_loss_dict["loss_grad_point"]
            context_point_loss = args.context_point_loss_coeff * context_point_loss
            loss_dict["context_point_loss"] = context_point_loss
        else:
            # only point loss
            pred_context_pts3d = output_dict["pred_context_pts3d"]
            gt_context_pts3d = input_dict["context_pts3d"]
            gt_context_points_mask = input_dict['context_valid_masks'].to(torch.bool)
            context_point_loss = F.l1_loss(pred_context_pts3d[gt_context_points_mask], gt_context_pts3d[gt_context_points_mask])
            context_point_loss = args.context_point_loss_coeff * context_point_loss
            loss_dict["context_point_loss"] = context_point_loss

    # context feat loss
    if args.enable_context_feat_loss:
        pred_feat = pred_dict["rendered_feat"]
        target_context_feat = rearrange(input_dict["context_feat"], "b t v c h w -> b t v h w c") \
            if "context_feat" in input_dict else None
        if args.feat_loss_type == 'mse':
            context_feat_loss = F.mse_loss(pred_feat, target_context_feat)
        elif args.feat_loss_type == 'cos_dist':
            context_feat_loss = 1. - F.cosine_similarity(pred_feat, target_context_feat, dim=-1)  # in c dim
            context_feat_loss = context_feat_loss.mean()
        elif args.feat_loss_type == 'cls_prob':
            if "context_semantic_labels" in input_dict:
                semantic_labels_mask = input_dict["context_semantic_labels_mask"].to(torch.bool)
                # pred_feat[semantic_labels_mask]: b t v h w c -> n h w c
                valid_pred_feat = rearrange(pred_feat[semantic_labels_mask], "n h w c -> (n h w) c")
                similarity = valid_pred_feat @ get_text_label_feats(SEMANTIC_LABEL_LIST).T / 0.07  # temperature same as LSeg
                semantic_labels = input_dict["context_semantic_labels"]
                valid_semantic_labels = rearrange(semantic_labels[semantic_labels_mask], "n h w -> (n h w)")
                # context_feat_loss = F.cross_entropy(similarity, valid_semantic_labels.long())
                context_feat_loss = F.cross_entropy(similarity, valid_semantic_labels.long()) / valid_pred_feat.shape[0] * 1.0
            else:
                context_feat_loss = torch.tensor(0.).to(pred_feat)
        else:
            raise ValueError
        loss_dict["context_feat_loss"] = args.feat_loss_coeff * context_feat_loss  # TODO: Change args.feat_loss_coeff to a list to use different coefficients for different features

    # camera pose loss
    if args.enable_camera_loss and "pred_context_camera_enc_list" in output_dict and \
                                                    "context_camtoworlds" in input_dict and "context_intrinsics" in input_dict:
        camera_loss_dict = compute_camera_loss(output_dict, input_dict, loss_type="l1")
        loss_dict["camera_pose_loss"] = args.camera_loss_coeff * camera_loss_dict['loss_camera']

        '''
        if args.enable_context_depth_loss and "pred_context_depth" in output_dict and "context_depth" in input_dict:
            pred_context_depth = output_dict["pred_context_depth"]
            target_context_depth = input_dict["context_depth"]
            context_depth_loss = compute_depth_loss(pred_context_depth, target_context_depth)
            loss_dict["context_depth_loss"] = context_depth_loss
        '''

    # NOTE: Pre-train the 3D annotation head for the context part
    if data_iter_step < args.context_prediction_loss_warmup_steps:
        return loss_dict

    # rendering loss
    if lpips_loss is not None:
        loss_dict.update(lpips_loss(pred_rgb, target_rgb))
    else:
        rgb_loss = F.mse_loss(pred_rgb, target_rgb)
        loss_dict["rgb_loss"] = rgb_loss

    # NOTE: Also involved in lpips_loss, so written outside
    loss_dict["rgb_mse"] = loss_dict["rgb_loss"].clone()
    loss_dict["rgb_loss"] *= args.rgb_loss_coeff

    if args.enable_depth_loss and "target_depth" in target_dict:
        pred_depth, target_depth = pred_dict[pred_dict["depth_key"]], target_dict["target_depth"]
        depth_loss = compute_depth_loss(pred_depth, target_depth)
        loss_dict["depth_loss"] = depth_loss

        if pred_dict["decoder_depth_key"] is not None:
            pred_decoder_depth = pred_dict[pred_dict["decoder_depth_key"]]
            decoded_depth_loss = compute_depth_loss(pred_decoder_depth, target_depth)
            loss_dict["decoded_depth_loss"] = decoded_depth_loss
            if (
                args.enable_sky_depth_loss or args.enable_sky_opacity_loss
            ) and "target_sky_masks" in target_dict:
                sky_decoded_depth_loss, _ = compute_sky_depth_loss(
                    pred_decoder_depth,
                    target_dict["target_sky_masks"],
                    sky_depth=args.sky_depth,
                )
                loss_dict["sky_decodede_depth_loss"] = sky_decoded_depth_loss

    if args.enable_pseudo_depth_loss and "target_pseudo_depth" in target_dict:
        pred_depth, target_pseudo_depth, target_pseudo_depth_conf = pred_dict[pred_dict["depth_key"]], target_dict["target_pseudo_depth"], target_dict["target_pseudo_depth_conf"]
        target_depth = target_dict["target_depth"]
        # pseudo_depth_loss = compute_pseudo_depth_loss(pred_depth, target_pseudo_depth, target_pseudo_depth_conf)
        pseudo_depth_loss = compute_pseudo_depth_loss_v2(pred_depth, target_pseudo_depth, target_depth, target_pseudo_depth_conf)
        loss_dict["pseudo_depth_loss"] = args.pseudo_depth_coeff * pseudo_depth_loss

    # long lifespan regularization loss
    if args.enable_lifespan_reg_loss and "lifespans" in gs_params:
        lifespan_reg_loss = torch.abs(1 / (gs_params["lifespans"] + 1e-8)).mean()
        loss_dict["lifespan_reg_loss"] = args.lifespan_reg_coeff * lifespan_reg_loss

    if args.enable_flow_reg_loss:
        pred_flow = gs_params["forward_flow"]
        zero_flow = torch.zeros_like(pred_flow).to(device)
        forward_flow_reg = F.mse_loss(pred_flow, zero_flow, reduction="none")
        loss_dict["flow_reg_loss"] = args.flow_reg_coeff * forward_flow_reg.mean()

    if args.enable_flow_loss and pred_dict["flow_key"] is not None and data_iter_step > args.flow_loss_start_iter:
        # context frames
        # pred_flow = gs_params["forward_flow"]
        # target_flow = input_dict['context_flow']

        # target frames
        pred_flow = pred_dict['rendered_flow']
        target_flow = target_dict['target_flow']

        nonzero_valid_mask = target_flow.norm(dim=-1) > 0.01

        if nonzero_valid_mask.sum() > 0:
            max_flow = target_flow.norm(dim=-1).max()
            nonzero_pred_flow = pred_flow[nonzero_valid_mask] / max_flow
            nonzero_target_flow = target_flow[nonzero_valid_mask] / max_flow
            nonzero_forward_flow_loss = F.l1_loss(nonzero_pred_flow, nonzero_target_flow)

            zero_pred_flow = pred_flow[~nonzero_valid_mask]
            zero_target_flow = target_flow[~nonzero_valid_mask]
            zero_forward_flow_loss = F.l1_loss(zero_pred_flow, zero_target_flow)

            nonzero_weight = (nonzero_valid_mask.sum() / pred_flow.numel())

            forward_flow_loss = (1 - nonzero_weight) * nonzero_forward_flow_loss + nonzero_weight * zero_forward_flow_loss
        else:
            forward_flow_loss =  0.001 * F.l1_loss(pred_flow, target_flow, reduction="none")
        loss_dict["flow_loss"] = args.flow_coeff * forward_flow_loss.mean()

    if args.enable_sky_depth_loss and "target_sky_masks" in target_dict:
        # real gaussian depth
        sky_depth_loss, sky_flow_reg_loss = compute_sky_depth_loss(
            pred_dict[pred_dict["depth_key"]],
            target_dict["target_sky_masks"],
            sky_depth=args.sky_depth,
            flow=(pred_dict[pred_dict["flow_key"]] if pred_dict["flow_key"] is not None else None),
        )
        loss_dict["sky_depth_loss"] = sky_depth_loss
        loss_dict["sky_flow_reg_loss"] = sky_flow_reg_loss
        loss_dict["opacity_loss"] = 0.01 * F.mse_loss(
            pred_dict[pred_dict["alpha_key"]],
            torch.ones_like(pred_dict[pred_dict["alpha_key"]]),
        )
        if pred_dict["decoder_depth_key"] is not None:
            (sky_decoded_depth_loss, sky_decoded_flow_reg_loss,) = compute_sky_depth_loss(
                pred_dict[pred_dict["decoder_depth_key"]],
                target_dict["target_sky_masks"],
                sky_depth=args.sky_depth,
                flow=(
                    pred_dict[pred_dict["decoder_flow_key"]]
                    if pred_dict["decoder_flow_key"] is not None
                    else None
                ),
            )
            loss_dict["sky_decodede_depth_loss"] = sky_decoded_depth_loss
            loss_dict["sky_decoded_flow_reg_loss"] = sky_decoded_flow_reg_loss

    elif args.enable_sky_opacity_loss and "target_sky_masks" in target_dict:
        opacity = pred_dict[pred_dict["alpha_key"]].squeeze(-1)
        b, t, v, h, w = opacity.shape
        gt_h, gt_w = target_dict["target_sky_masks"].shape[-2:]
        if h != gt_h or w != gt_w:
            opacity = F.interpolate(
                rearrange(opacity, "b t v h w -> (b t v) 1 h w"),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            )
            opacity = rearrange(opacity, "(b t v) 1 h w -> b t v h w", b=b, t=t, v=v)
        sky_opacity_loss = F.l1_loss(opacity, 1 - target_dict["target_sky_masks"])
        loss_dict["sky_opacity_loss"] = sky_opacity_loss * args.sky_opacity_loss_coeff

    if args.enable_feat_loss:
        pred_feat = pred_dict["rendered_feat"]
        target_feat = rearrange(target_dict["target_feat"], "b t v c h w -> b t v h w c") \
            if "target_feat" in target_dict else None
        if args.feat_loss_type == 'mse':
            feat_loss = F.mse_loss(pred_feat, target_feat)
        elif args.feat_loss_type == 'cos_dist':
            feat_loss = 1. - F.cosine_similarity(pred_feat, target_feat, dim=-1)  # in c dim
            feat_loss = feat_loss.mean()
        elif args.feat_loss_type == 'cls_prob':
            if "target_semantic_labels" in target_dict:
                semantic_labels_mask = target_dict["target_semantic_labels_mask"].to(torch.bool)
                # pred_feat[semantic_labels_mask]: b t v h w c -> n h w c
                valid_pred_feat = rearrange(pred_feat[semantic_labels_mask], "n h w c -> (n h w) c")
                similarity = valid_pred_feat @ get_text_label_feats(SEMANTIC_LABEL_LIST).T / 0.07  # temperature same as LSeg
                semantic_labels = target_dict["target_semantic_labels"]
                valid_semantic_labels = rearrange(semantic_labels[semantic_labels_mask], "n h w -> (n h w)")
                # feat_loss = F.cross_entropy(similarity, valid_semantic_labels.long())
                feat_loss = F.cross_entropy(similarity, valid_semantic_labels.long()) / valid_pred_feat.shape[0] * 1.0
            else:
                feat_loss = torch.tensor(0.).to(pred_feat)
        else:
            raise ValueError
        loss_dict["feat_loss"] = args.feat_loss_coeff * feat_loss  # TODO: Change args.feat_loss_coeff to a list to use different coefficients for different features

    return loss_dict


def compute_scene_flow_metrics(pred, labels):
    """
    Computes the scene flow metrics between the predicted and target scene flow values.
    # modified from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/0e4f403c73cb3fcd5503294a7c461926a4cdd1ad/utils.py#L12

    Args:
        pred (Tensor): predicted scene flow values
        labels (Tensor): target scene flow values
    Returns:
        dict: scene flow metrics
    """
    l2_norm = torch.sqrt(torch.sum((pred - labels) ** 2, -1)).cpu()
    # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, -1)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: Acc_5
    error_lt_5 = torch.BoolTensor((l2_norm < 0.05))
    relative_err_lt_5 = torch.BoolTensor((relative_err < 0.05))
    acc3d_strict = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: Acc_10
    error_lt_10 = torch.BoolTensor((l2_norm < 0.1))
    relative_err_lt_10 = torch.BoolTensor((relative_err < 0.1))
    acc3d_relax = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    # NOTE: outliers
    l2_norm_gt_3 = torch.BoolTensor(l2_norm > 0.3)
    relative_err_gt_10 = torch.BoolTensor(relative_err > 0.1)
    outlier = torch.mean((l2_norm_gt_3 | relative_err_gt_10).float()).item()

    # NOTE: angle error
    unit_label = labels / (labels.norm(dim=-1, keepdim=True) + 1e-7)
    unit_pred = pred / (pred.norm(dim=-1, keepdim=True) + 1e-7)

    # it doesn't make sense to compute angle error on zero vectors
    # we use a threshold of 0.1 to avoid noisy gt flow
    non_zero_flow_mask = labels_norm > 0.1
    # Apply the mask to filter out zero vectors
    unit_label = unit_label[non_zero_flow_mask]
    unit_pred = unit_pred[non_zero_flow_mask]
    # Initialize angle_error
    angle_error = 0.0
    # Check if there are any valid vectors to compute the angle error
    if unit_label.numel() > 0:
        eps = 1e-7
        # Compute the dot product and clamp its values to avoid numerical issues with acos
        dot_product = (unit_label * unit_pred).sum(dim=-1).clamp(min=-1 + eps, max=1 - eps)

        # Optionally, handle any remaining NaNs in the dot product
        dot_product = torch.nan_to_num(dot_product, nan=0.0)

        # Compute the angle error in radians and take the mean
        angle_error = torch.acos(dot_product).mean().item()

    torch.cuda.empty_cache()
    return {
        "EPE3D": EPE3D,
        "acc3d_strict": acc3d_strict,
        "acc3d_relax": acc3d_relax,
        "outlier": outlier,
        "angle_error": angle_error,
    }


def remap_gt_to_interested(gt_label, interested_classes, ignore_value=255):
    """
    Remap classes in gt_label to 0, 1, 2, ... according to interested_classes order,
    and set others to ignore_value.

    Args:
        gt_label: Tensor of shape [1, H, W] or [H, W], dtype=torch.long
        interested_classes: list of original class IDs, e.g., [3, 5, 12]
        ignore_value: int, value for ignored pixels (default: 255)

    Returns:
        new_gt_label: same shape as input, dtype=torch.long
                     values: 0 to len(interested_classes)-1 for interested classes,
                             ignore_value otherwise.
    """
    device = gt_label.device
    orig_shape = gt_label.shape
    gt_flat = gt_label.view(-1)  # [N]

    # Create lookup table: size is max_label + 1
    max_label = gt_flat.max().item()
    # If the max original label is 28, create a lookup table of length 29
    lookup = torch.full((max_label + 1,), ignore_value, dtype=torch.long, device=device)

    # Fill mappings for interested classes: original class -> new index
    for new_idx, orig_cls in enumerate(interested_classes):
        if orig_cls <= max_label:  # Safety check
            lookup[orig_cls] = new_idx

    # Apply mapping
    new_flat = lookup[gt_flat]

    return new_flat.view(orig_shape)


def compute_semantic_metrics(pred, labels, interested_classes=[1, 2, 3], ignore_index=255, orig_img=None):
    # # merge
    # labels[labels == 10] = 6  # TYPE_BICYCLE and TYPE_CYCLIST
    # labels[labels == 11] = 7  # TYPE_MOTORCYCLE and TYPE_MOTORCYCLIST

    # interested_classes = [2, 3, 4, 6]

    # # remap: only compute certain classes
    # num_classes = len(interested_classes)
    # labels_remap = remap_gt_to_interested(labels, interested_classes=interested_classes)

    # mIoU: Use JaccardIndex (i.e., IoU)
    miou_metric = MulticlassJaccardIndex(
        # num_classes=num_classes+1,
        num_classes=len(SEMANTIC_LABEL_LIST) + 1,
        ignore_index=ignore_index,
        average="macro"  # Equivalent to mIoU (average over all classes)
    ).to(labels.device)

    # Pixel Accuracy
    acc_metric = MulticlassAccuracy(
        # num_classes=num_classes+1,
        num_classes=len(SEMANTIC_LABEL_LIST) + 1,
        ignore_index=ignore_index,
        average="micro"  # Micro average = total correct pixels / total pixels
    ).to(labels.device)

    # Compute mIoU: pred needs to be logits (or probabilities), target needs to be long labels
    # miou = miou_metric(pred, labels_remap)
    # acc = acc_metric(pred, labels_remap)
    miou = miou_metric(pred, labels)
    acc = acc_metric(pred, labels)

    if orig_img is not None:
        # The default camera is the front camera, but it can be changed to another one.
        if pred.shape[0] > 1:
            cam_idx = 1
        else:
            cam_idx = 0
        fig, axes = plt.subplots(2, len(SEMANTIC_LABEL_LIST) + 1, figsize=(27, 6))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        axes[0, 0].imshow(orig_img[cam_idx].permute(1, 2, 0).detach().cpu().numpy())
        axes[0, 0].set_title(f"Input")
        axes[1, 0].imshow(orig_img[cam_idx].permute(1, 2, 0).detach().cpu().numpy())
        axes[1, 0].set_title(f"Input")
        # Pred
        for i, (label_id, label_name) in enumerate(enumerate(SEMANTIC_LABEL_LIST)):
            ax = axes[0, i + 1]
            mask = (pred[cam_idx] == label_id).detach().cpu().numpy()
            ax.imshow(mask, cmap='gray')
            ax.set_title(f"Pred: {label_name}", fontsize=8)
            ax.axis('off')
        # GT
        for i, (label_id, label_name) in enumerate(enumerate(SEMANTIC_LABEL_LIST)):
            ax = axes[1, i + 1]
            mask = (labels[cam_idx] == label_id).detach().cpu().numpy()
            ax.imshow(mask, cmap='gray')
            ax.set_title(f"GT: {label_name}", fontsize=8)
            ax.axis('off')

        fig.suptitle(
            f'Semantic Segmentation Results - mIoU: {miou.item():.4f}, Acc: {acc.item():.4f}',
            fontsize=16, y=0.95
        )
        plt.savefig('test.png', bbox_inches='tight', dpi=150)
        plt.close(fig)

    return {
        "MIOU": miou.item(),
        "ACC": acc.item(),
    }

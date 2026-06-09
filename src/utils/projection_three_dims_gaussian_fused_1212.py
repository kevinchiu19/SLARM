from typing import Optional, Tuple

import torch
import torch_npu
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module
import torch.nn.functional as F
from typing_extensions import Literal

import meta_gauss_render._C


def decode_uint8_bitmask(filter_mask: torch.Tensor, high_order=True) -> torch.Tensor:
    """
    Decode uint8 single-bit mask (shape: 1,1,N/8) to bool mask (shape: 1,1,N)

    Args:
        filter_mask: uint8 Tensor, shape=(1,1,N/8)

    Returns:
        bool Tensor, shape=(1,1,N)
    """
    # 1. Squeeze redundant dimensions: (1,1,N/8) -> (N/8,)
    mask_squeezed = filter_mask.squeeze()  # shape: (N/8,)

    # 2. Generate bit masks: extract 8 bits from each uint8 (note bit order, default is high order first, adjust as needed)
    if high_order:
        # Mask example: [128, 64, 32, 16, 8, 4, 2, 1] -> corresponds to bits 7 to 0 (high -> low)
        bit_masks = (1 << torch.arange(7, -1, -1)).to(filter_mask.device, dtype=torch.uint8)  # shape: (8,)
    else:
        bit_masks = (1 << torch.arange(0, 8)).to(filter_mask.device, dtype=torch.uint8)  # shape: (8,)

    # 3. Extract 8 bits from each byte: broadcast computation (N/8,) x (8,) -> (N/8, 8)
    # bit_values: each element is 0 or 1, shape=(N/8,8)
    bit_values = (mask_squeezed.unsqueeze(1) & bit_masks) != 0

    # 4. Flatten and reshape to target shape: (N/8x8,) -> (1,1,N)
    bool_mask = bit_values.flatten().reshape(1, 1, -1)  # shape: (1,1,N)

    return bool_mask


class ProjectionThreeDimsGaussianFused(Function):
    @staticmethod
    def forward(ctx,
                means: torch.Tensor,
                colors: torch.Tensor,
                covars: torch.Tensor = None,
                quat: torch.Tensor = None,
                scales: torch.Tensor = None,
                opacities: torch.Tensor = None,
                viewmats: torch.Tensor = None,
                Ks: torch.Tensor = None,
                width: int = 0,
                height: int = 0,
                eps: float = 0.3,
                near_plane: float = 0.01,
                far_plane: float = 1e10,
                calc_compensations: bool = False,
                camera_model: str = "pinhole"):

        if quat is not None:
            if scales is None:
                raise ValueError("'quat' and 'scales' are required together.")
            if covars is not None:
                raise ValueError("Invalid parameter combination: 'covars' and ('quat', 'scales')pair are mutually exclusive.")
            quat = quat.permute((0, 2, 1)).contiguous()
            scales = scales.permute((0, 2, 1)).contiguous()
            covars = meta_gauss_render._C.quat_scales_to_covars(quat, scales)
        else:
            covars = covars.permute((0, 2, 3, 1)).contiguous()

        means = means.permute((0, 2, 1)).contiguous()

        means2d, depths, conics, compensations, det, radius, covars2d = meta_gauss_render._C.projection_three_dims_gaussian_forward(
                means,
                covars,
                opacities,
                viewmats,
                Ks,
                width,
                height,
                eps,
                calc_compensations,
                camera_model
        )

        det = det.squeeze(-2)
        depths = depths.squeeze(-2)
        if calc_compensations:
            compensations = compensations.squeeze(-2)
        else:
            compensations = None

        means_culling, colors_culling, means2d_culling, depths_culling, radius_culling, covars2d_culling, conics_culling, opacities_culling, filter, cnt = meta_gauss_render._C.gaussian_filter(
                means,
                colors,
                det,
                opacities,
                means2d,
                depths,
                radius,
                conics,
                covars2d,
                compensations,
                width,
                height,
                near_plane,
                far_plane)

        ctx.save_for_backward(means, conics, viewmats, quat, scales, Ks, filter, compensations)
        ctx.width = width
        ctx.height = height
        return means2d_culling, depths_culling, conics_culling, opacities_culling, radius_culling, covars2d_culling, colors_culling, cnt

    @staticmethod
    def backward(
        ctx, *v_args
    ):
        means, conics, viewmats, quats, scales, Ks, filter, compensations = ctx.saved_tensors

        width = ctx.width
        height = ctx.height
        v_means2d, v_depths, v_conics, v_opacities_culling, v_radii, v_covars2d, v_colors_culling, v_cnt = v_args
        v_pW, v_quats, v_scales, v_R, v_colors, v_opacities = meta_gauss_render._C.fully_fused_projection_bwd(
                means,
                quats,
                scales,
                conics,
                viewmats,
                Ks,
                v_means2d,
                v_depths,
                v_conics,
                v_colors_culling,
                v_opacities_culling,
                filter,
                compensations,
                width,
                height
        )

        filter = decode_uint8_bitmask(filter, high_order=False)

        v_pW = torch.where(~(filter.unsqueeze(-1)), torch.tensor(0.0, device=v_pW.device), v_pW)
        v_quats = torch.where(~(filter.unsqueeze(-1)), torch.tensor(0.0, device=v_quats.device), v_quats)
        v_scales = torch.where(~(filter.unsqueeze(-1)), torch.tensor(0.0, device=v_scales.device), v_scales)
        v_colors = torch.where(~(filter), torch.tensor(0.0, device=v_colors.device), v_colors)
        v_opacities = torch.where(~(filter.squeeze(1)), torch.tensor(0.0, device=v_opacities.device), v_opacities)

        return v_pW, v_colors, None, v_quats, v_scales, v_opacities, \
                None, None, None, None, None, None, None, None, None

projection_three_dims_gaussian_fused = ProjectionThreeDimsGaussianFused.apply
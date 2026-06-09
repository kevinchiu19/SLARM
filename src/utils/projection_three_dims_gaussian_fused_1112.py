import os
from typing import Optional, Tuple

import torch
import torch_npu
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module
import torch.nn.functional as F
from typing_extensions import Literal

import meta_gauss_render._C

def create_offset_tensor(device):
    offset_list = [[] for _ in range(18)]
    offset = []
    for block in range(8):
        base = block * 16
        for o in range(12):
            offset_list[o].extend([base + o])
        for value in range(6):
            ks_base = block * 9
            offset_list[value + 12].extend([ks_base + value])

    offset = [*offset_list[0], *offset_list[1], *offset_list[2], *offset_list[4],
              *offset_list[5], *offset_list[6], *offset_list[8], *offset_list[9],
              *offset_list[10], *offset_list[3], *offset_list[7], *offset_list[11],
              *offset_list[12], *offset_list[13], *offset_list[14], *offset_list[15],
              *offset_list[16], *offset_list[17]]

    offset_tensor = torch.tensor(offset, dtype=torch.int32, device=device) * 4
    return offset_tensor

# copy from meta_gauss_render/npu/tests/test_projection_3dgs_forward.py in https://codehub-y.huawei.com/LargeSpatialModel/MetaGaussianSplat/files/commit/492115ec6e1f68fe2468b10177fba98c046c9ade?ref=ascend_gauss_render_v0.3_optimize_1127
def _persp_proj(
    means: Tensor,  # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of perspective projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [..., C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [..., C, N, 3, 3].
        Ks: Camera intrinsics. [..., C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [..., C, N, 2].
        - **cov2d**: Projected covariances. [..., C, N, 2, 2].
    """
    batch_dims = means.shape[:-3]
    C, N = means.shape[-3:-1]
    assert means.shape == batch_dims + (C, N, 3), means.shape
    assert covars.shape == batch_dims + (C, N, 3, 3), covars.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [..., C, N]
    tz2 = tz**2  # [..., C, N]

    fx = Ks[..., 0, 0, None]  # [..., C, 1]
    fy = Ks[..., 1, 1, None]  # [..., C, 1]
    cx = Ks[..., 0, 2, None]  # [..., C, 1]
    cy = Ks[..., 1, 2, None]  # [..., C, 1]
    tan_fovx = 0.5 * width / fx  # [..., C, 1]
    tan_fovy = 0.5 * height / fy  # [..., C, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy

    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    O = torch.zeros(batch_dims + (C, N), device=means.device, dtype=means.dtype)
    J = torch.stack(
        [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(batch_dims + (C, N, 2, 3))

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum(
        "...ij,...nj->...ni", Ks[..., :2, :3], means
    )  # [..., C, N, 2]

    means2d = means2d / tz[..., None]  # [..., C, N, 2]
    return means2d, cov2d  # [..., C, N, 2], [..., C, N, 2, 2]

# copy from meta_gauss_render/npu/tests/test_projection_3dgs_forward.py in https://codehub-y.huawei.com/LargeSpatialModel/MetaGaussianSplat/files/commit/492115ec6e1f68fe2468b10177fba98c046c9ade?ref=ascend_gauss_render_v0.3_optimize_1127
def _fisheye_proj(
    means: Tensor,  # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of fisheye projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [..., C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [..., C, N, 3, 3].
        Ks: Camera intrinsics. [..., C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [..., C, N, 2].
        - **cov2d**: Projected covariances. [..., C, N, 2, 2].
    """
    batch_dims = means.shape[:-3]
    C, N = means.shape[-3:-1]
    assert means.shape == batch_dims + (C, N, 3), means.shape
    assert covars.shape == batch_dims + (C, N, 3, 3), covars.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

    x, y, z = torch.unbind(means, dim=-1)  # [..., C, N]

    fx = Ks[..., 0, 0, None]  # [..., C, 1]
    fy = Ks[..., 1, 1, None]  # [..., C, 1]
    cx = Ks[..., 0, 2, None]  # [..., C, 1]
    cy = Ks[..., 1, 2, None]  # [..., C, 1]

    eps = 0.0000001
    xy_len = (x**2 + y**2) ** 0.5 + eps
    theta = torch.atan2(xy_len, z + eps)
    means2d = torch.stack(
        [
            x * fx * theta / xy_len + cx,
            y * fy * theta / xy_len + cy,
        ],
        dim=-1,
    )  # [..., C, N, 2]

    x2 = x * x + eps
    y2 = y * y
    xy = x * y
    x2y2 = x2 + y2
    x2y2z2_inv = 1.0 / (x2y2 + z * z)
    b = torch.atan2(xy_len, z) / xy_len / x2y2
    a = z * x2y2z2_inv / (x2y2)
    J = torch.stack(
        [
            fx * (x2 * a + y2 * b),
            fx * xy * (a - b),
            -fx * x * x2y2z2_inv,
            fy * xy * (a - b),
            fy * (y2 * a + x2 * b),
            -fy * y * x2y2z2_inv,
        ],
        dim=-1,
    ).reshape(batch_dims + (C, N, 2, 3))

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    return means2d, cov2d  # [..., C, N, 2], [..., C, N, 2, 2]

# copy from meta_gauss_render/npu/tests/test_projection_3dgs_forward.py in https://codehub-y.huawei.com/LargeSpatialModel/MetaGaussianSplat/files/commit/492115ec6e1f68fe2468b10177fba98c046c9ade?ref=ascend_gauss_render_v0.3_optimize_1127
def _ortho_proj(
    means: Tensor,  # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of orthographic projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [..., C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [..., C, N, 3, 3].
        Ks: Camera intrinsics. [..., C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [..., C, N, 2].
        - **cov2d**: Projected covariances. [..., C, N, 2, 2].
    """
    batch_dims = means.shape[:-3]
    C, N = means.shape[-3:-1]
    assert means.shape == batch_dims + (C, N, 3), means.shape
    assert covars.shape == batch_dims + (C, N, 3, 3), covars.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

    fx = Ks[..., 0, 0, None]  # [..., C, 1]
    fy = Ks[..., 1, 1, None]  # [..., C, 1]

    O = torch.zeros(batch_dims + (C, 1), device=means.device, dtype=means.dtype)
    J = (
        torch.stack([fx, O, O, O, fy, O], dim=-1)
        .reshape(batch_dims + (C, 1, 2, 3))
        .repeat([1] * len(batch_dims) + [1, N, 1, 1])
    )

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = (
        means[..., :2] * Ks[..., None, [0, 1], [0, 1]] + Ks[..., None, [0, 1], [2, 2]]
    )  # [..., C, N, 2]
    return means2d, cov2d  # [..., C, N, 2], [..., C, N, 2, 2]

# copy from meta_gauss_render/npu/tests/test_projection_3dgs_forward.py in https://codehub-y.huawei.com/LargeSpatialModel/MetaGaussianSplat/files/commit/492115ec6e1f68fe2468b10177fba98c046c9ade?ref=ascend_gauss_render_v0.3_optimize_1127
def _world_to_cam(
    means: Tensor,  # [..., N, 3]
    covars: Tensor,  # [..., N, 3, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of world to camera transformation on Gaussians.

    Args:
        means: Gaussian means in world coordinate system. [..., N, 3].
        covars: Gaussian covariances in world coordinate system. [..., N, 3, 3].
        viewmats: world to camera transformation matrices. [..., C, 4, 4].

    Returns:
        A tuple:

        - **means_c**: Gaussian means in camera coordinate system. [..., C, N, 3].
        - **covars_c**: Gaussian covariances in camera coordinate system. [..., C, N, 3, 3].
    """
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert covars.shape == batch_dims + (N, 3, 3), covars.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape

    R = viewmats[..., :3, :3]  # [..., C, 3, 3]
    t = viewmats[..., :3, 3]  # [..., C, 3]

    means_c = (
        torch.einsum("...cij,...nj->...cni", R, means) + t[..., None, :]
    )  # [..., C, N, 3]
    covars_c = torch.einsum(
        "...cij,...njk,...clk->...cnil", R, covars, R
    )  # [..., C, N, 3, 3]
    return means_c, covars_c

# copy from meta_gauss_render/npu/tests/test_projection_3dgs_forward.py in https://codehub-y.huawei.com/LargeSpatialModel/MetaGaussianSplat/files/commit/492115ec6e1f68fe2468b10177fba98c046c9ade?ref=ascend_gauss_render_v0.3_optimize_1127
def _fully_fused_projection(
    means: Tensor,  # [..., N, 3]
    covars: Tensor,  # [..., N, 3, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert covars.shape == batch_dims + (N, 3, 3), covars.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

    assert (
        camera_model != "ftheta"
    ), "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"

    means_c, covars_c = _world_to_cam(means, covars, viewmats)

    if camera_model == "ortho":
        means2d, covars2d = _ortho_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "fisheye":
        means2d, covars2d = _fisheye_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "pinhole":
        means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)
    else:
        assert_never(camera_model)

    det_orig = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10) # torch.Size([1, 8, 8])

    if calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [..., C, N, 3]
    depths = means_c[..., 2]  # [..., C, N]

    b = (covars2d[..., 0, 0] + covars2d[..., 1, 1]) / 2  # (...,)
    tmp = torch.sqrt(torch.clamp(b**2 - det, min=0.01))
    v1 = b + tmp  # (...,)
    r1 = 3.33 * torch.sqrt(v1)
    radius_x = torch.ceil(torch.minimum(3.33 * torch.sqrt(covars2d[..., 0, 0]), r1))
    radius_y = torch.ceil(torch.minimum(3.33 * torch.sqrt(covars2d[..., 1, 1]), r1))

    radius = torch.stack([radius_x, radius_y], dim=-1)  # [..., C, N, 2]
    # valid, inside, filter are already included in ProjectionThreeDimsGaussianFused forward
    # valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    # inside = (
    #     (means2d[..., 0] + radius[..., 0] > 0)
    #     & (means2d[..., 0] - radius[..., 0] < width)
    #     & (means2d[..., 1] + radius[..., 1] > 0)
    #     & (means2d[..., 1] - radius[..., 1] < height)
    # )
    radii = radius.int()

    # filter = torch.logical_and(inside, valid)
    return radii, means2d, depths, conics, compensations, det, radius

class ProjectionThreeDimsGaussianFused(Function):
    @staticmethod
    def forward(ctx,
                means: torch.Tensor,
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
            quat = F.normalize(quat, p=2, dim=-1)
            covars = meta_gauss_render._C.quat_scales_to_covars(quat, scales)

        if os.getenv('RENDER_PROJ_FWD_USE_FUSED_KERNEL'):
            offset = create_offset_tensor(device=means.device)
            radii, means2d, depths, conics, compensations, det, radius = meta_gauss_render._C.projection_three_dims_gaussian_forward(
                    means,
                    covars,
                    opacities,
                    viewmats,
                    Ks,
                    offset,
                    width,
                    height,
                    eps,
                    calc_compensations,
                    camera_model
            )
            radii = radii.permute(0, 2, 3, 1).contiguous()
            means2d = means2d.permute(0, 2, 3, 1).contiguous()
            conics = conics.permute(0, 2, 3, 1).contiguous()
            radius = radius.permute(0, 2, 3, 1).contiguous()
        else:
            radii, means2d, depths, conics, compensations, det, radius = _fully_fused_projection(
                    means,
                    covars,
                    # opacities,
                    viewmats,
                    Ks,
                    # offset,
                    width,
                    height,
                    eps,
                    calc_compensations,
                    camera_model
            )

        valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
        # radius[~valid] = 0.0
        # radius *= valid.float().unsqueeze(-1)
        radius = torch.where(~(valid.unsqueeze(-1)), torch.tensor(0.0, device=radius.device), radius)
        inside = (
            (means2d[..., 0] + radius[..., 0] > 0)
            & (means2d[..., 0] - radius[..., 0] < width)
            & (means2d[..., 1] + radius[..., 1] > 0)
            & (means2d[..., 1] - radius[..., 1] < height)
        )
        # radius[~inside] = 0.0
        # radius *= inside.float().unsqueeze(-1)
        radius = torch.where(~(inside.unsqueeze(-1)), torch.tensor(0.0, device=radius.device), radius)
        radii = radius.int()
        filter = torch.logical_and(inside, valid)
        ctx.save_for_backward(means, conics, viewmats, quat, scales, Ks, filter)
        ctx.width = width
        ctx.height = height
        return radii, means2d, depths, conics, compensations, filter

    @staticmethod
    def backward(
        ctx, *v_args
    ):
        means, conics, viewmats, quats, scales, Ks, filter = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        v_radii, v_means2d, v_depths, v_conics, v_compensations, v_covars2d = v_args

        cam_num = v_means2d.shape[1]
        v_pW_list, v_quats_list, v_scales_list, v_R_list = [], [], [], []

        for c_idx in range(cam_num):
            v_pW, v_quats, v_scales, v_R = meta_gauss_render._C.fully_fused_projection_bwd(means, quats, scales, conics[:,c_idx,:][:,None], viewmats[:,c_idx,:][:,None],
                                                                                Ks[:,c_idx,:][:,None], v_means2d[:,c_idx,:][:,None], v_depths[:,c_idx,:][:,None], v_conics[:,c_idx,:][:,None], width, height)
            v_pW_list.append(v_pW[:,None])
            v_quats_list.append(v_quats[:,None])
            v_scales_list.append(v_scales[:,None])
            v_R_list.append(v_R[:,None])

        v_pW = torch.cat(v_pW_list, dim=1)
        v_quats = torch.cat(v_quats_list, dim=1)
        v_scales = torch.cat(v_scales_list, dim=1)
        v_R = torch.cat(v_R_list, dim=1)


        # To Do: Dynamic data processing, support multi-batch, multi-view
        # v_pW[~filter]=0
        # v_quats[~filter]=0
        # v_scales[~filter]=0
        # v_pW *= filter.float().unsqueeze(-1)
        # v_quats *= filter.float().unsqueeze(-1)
        # v_scales *= filter.float().unsqueeze(-1)
        v_pW = torch.where(~(filter.unsqueeze(-1)), torch.tensor(0.0, device=v_pW.device), v_pW)
        v_quats = torch.where(~(filter.unsqueeze(-1)), torch.tensor(0.0, device=v_quats.device), v_quats)
        v_scales = torch.where(~(filter.unsqueeze(-1)), torch.tensor(0.0, device=v_scales.device), v_scales)

        return v_pW, None, v_quats, v_scales, \
                None, None, None, None, None, None, None, None, None, None


    # Original
    # @staticmethod
    # def forward(ctx,
    #             means: torch.Tensor,
    #             covars: torch.Tensor = None,
    #             quat: torch.Tensor = None,
    #             scales: torch.Tensor = None,
    #             opacities: torch.Tensor = None,
    #             viewmats: torch.Tensor = None,
    #             Ks: torch.Tensor = None,
    #             width: int = 0,
    #             height: int = 0,
    #             eps: float = 0.3,
    #             near_plane: float = 0.01,
    #             far_plane: float = 1e10,
    #             calc_compensations: bool = False,
    #             camera_model: str = "pinhole"):

    #     if quat is not None:
    #         if scales is None:
    #             raise ValueError("'quat' and 'scales' are required together.")
    #         if covars is not None:
    #             raise ValueError("Invalid parameter combination: 'covars' and ('quat', 'scales')pair are mutually exclusive.")
    #         quat = F.normalize(quat, p=2, dim=-1)
    #         covars = meta_gauss_render._C.quat_scales_to_covars(quat, scales)

    #     offset = create_offset_tensor(device=means.device)
    #     radii, means2d, depths, conics, compensations, det, radius = meta_gauss_render._C.projection_three_dims_gaussian_forward(
    #             means,
    #             covars,
    #             opacities,
    #             viewmats,
    #             Ks,
    #             offset,
    #             width,
    #             height,
    #             eps,
    #             calc_compensations,
    #             camera_model
    #     )
    #     radii = radii.permute(0, 2, 3, 1).contiguous()
    #     means2d = means2d.permute(0, 2, 3, 1).contiguous()
    #     conics = conics.permute(0, 2, 3, 1).contiguous()
    #     radius = radius.permute(0, 2, 3, 1).contiguous()
    #     valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    #     radius[~valid] = 0.0
    #     inside = (
    #         (means2d[..., 0] + radius[..., 0] > 0)
    #         & (means2d[..., 0] - radius[..., 0] < width)
    #         & (means2d[..., 1] + radius[..., 1] > 0)
    #         & (means2d[..., 1] - radius[..., 1] < height)
    #     )
    #     radius[~inside] = 0.0
    #     radii = radius.int()
    #     filter = torch.logical_and(inside, valid)
    #     ctx.save_for_backward(means, conics, viewmats, quat, scales, Ks, filter)
    #     ctx.width = width
    #     ctx.height = height
    #     return radii, means2d, depths, conics, compensations, filter

    # @staticmethod
    # def backward(
    #     ctx, *v_args
    # ):
    #     means, conics, viewmats, quats, scales, Ks, filter = ctx.saved_tensors
    #     width = ctx.width
    #     height = ctx.height
    #     v_radii, v_means2d, v_depths, v_conics, v_compensations, v_covars2d = v_args
    #     v_pW, v_quats, v_scales, v_R = meta_gauss_render._C.fully_fused_projection_bwd(means, quats, scales, conics, viewmats,
    #                                                                         Ks, v_means2d, v_depths, v_conics, width, height)

    #     # To Do: Dynamic data processing, support multi-batch, multi-view
    #     v_pW[~filter[0]]=0
    #     v_quats[~filter[0]]=0
    #     v_scales[~filter[0]]=0
    #     return v_pW, None, v_quats, v_scales, \
    #             None, None, None, None, None, None, None, None, None, None

projection_three_dims_gaussian_fused = ProjectionThreeDimsGaussianFused.apply
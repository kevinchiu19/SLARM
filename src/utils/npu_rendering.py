import os
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from gsplat.cuda._torch_impl import (
    _world_to_cam, _ortho_proj, _fisheye_proj, _persp_proj,
    _quat_scale_to_covar_preci,
)

from meta_gauss_render import AscendGaussRender
from meta_gauss_render.npu import CalcRender, get_render_schedule, get_num_vector_core


# Consistent with _fully_fused_projection in gsplat source gsplat/cuda/_torch_impl.py, with additional covars2d output
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
    if not os.getenv('DEBUG_NAN'):
        means_c[means_c == 0] = 1e-6  # 0.0 in means_c would raise nan in covars2d and det

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
    det = det.clamp(min=1e-10)

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

    radius_x = torch.ceil(3.33 * torch.sqrt(covars2d[..., 0, 0]))
    radius_y = torch.ceil(3.33 * torch.sqrt(covars2d[..., 1, 1]))

    radius = torch.stack([radius_x, radius_y], dim=-1)  # [..., C, N, 2]

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )
    radius[~inside] = 0.0

    radii = radius.int()
    return radii, means2d, depths, conics, compensations, covars2d


def ascend_rasterization_single_view(
    ascend_render: AscendGaussRender,
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, D]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: int = 64,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    batch_per_iter: int = 100,
    packed: bool = False,
    radius_clip: float = 0.0,
) -> Tuple[Tensor, Tensor, Dict]:
    """A version of rasterization() that utilies on PyTorch's autograd.

    .. note::
        This function still relies on gsplat's CUDA backend for some computation, but the
        entire differentiable graph is on of PyTorch (and nerfacc) so could use Pytorch's
        autograd for backpropagation.

    .. note::
        This function relies on installing latest nerfacc, via:
        pip install git+https://github.com/nerfstudio-project/nerfacc

    .. note::
        Compared to rasterization(), this function does not support some arguments such as
        `packed`, `sparse_grad` and `absgrad`.
    """

    N = means.shape[0]
    C = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode
    assert packed == False
    assert radius_clip == 0.0

    assert sh_degree is None
    # treat colors as post-activation values, should be in shape [N, D] or [C, N, D]
    assert (colors.dim() == 2 and colors.shape[0] == N) or (
        colors.dim() == 3 and colors.shape[:2] == (C, N)
    ), colors.shape

    # Project Gaussians to 2D.
    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    covars, _ = _quat_scale_to_covar_preci(quats, scales, True, False, triu=False)
    radii, means2d, depths, conics, compensations, covars2d = _fully_fused_projection(
        means,
        covars,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model = camera_model
    )

    opacities = opacities.repeat(C, 1)  # [C, N]
    camera_ids, gaussian_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
    # Colors are post-activation values, with shape [N, D] or [C, N, D]
    if colors.dim() == 2:
        # Turn [N, D] into [C, N, D]
        colors = colors.expand(C, -1, -1)

    # replaced with ascend gauss render sorting
    with torch.no_grad():
        all_in_mask = [
                ascend_render.build_tile_gs_mask(
                    means2d[_cam_view, :, 0], means2d[_cam_view, :, 1],
                    radii[_cam_view].to(torch.int64).max(dim=1)[0], width, height,
                    cov00=covars2d[_cam_view, :, 0, 0], cov01=covars2d[_cam_view, :, 0, 1],
                    cov11=covars2d[_cam_view, :, 1, 1], opacity=opacities[_cam_view]
                ) for _cam_view in range(C)
            ]
        sorted_gs_ids = []
        tile_offsets = []
        for _cam_view in range(0, C):
            cf_sorted_gs_ids, cf_tile_offsets = ascend_render.sort_gs(
                    all_in_mask[_cam_view], depths[_cam_view]
                )
            sorted_gs_ids.append(cf_sorted_gs_ids)
            tile_offsets.append(cf_tile_offsets)

    render_colors = []
    render_alphas = []
    for _cam_view in range(0, C):
        cf_means2 = torch.index_select(means2d[_cam_view], 0, sorted_gs_ids[_cam_view])[:, :2].transpose(0, 1).contiguous()
        cf_conics0 = torch.index_select(conics[_cam_view, :, 0], 0, sorted_gs_ids[_cam_view]).contiguous()
        cf_conics1 = torch.index_select(conics[_cam_view, :, 1], 0, sorted_gs_ids[_cam_view]).contiguous()
        cf_conics2 = torch.index_select(conics[_cam_view, :, 2], 0, sorted_gs_ids[_cam_view]).contiguous()
        cf_opacity = torch.index_select(opacities[_cam_view], 0, sorted_gs_ids[_cam_view]).contiguous()
        cf_colors3 = torch.index_select(colors[_cam_view][:, 0:3], 0, sorted_gs_ids[_cam_view]).transpose(0, 1).contiguous()
        if colors[_cam_view].shape[1] == 6:
            cf_flows3 = torch.index_select(colors[_cam_view][:, 3:6], 0, sorted_gs_ids[_cam_view]).transpose(0, 1).contiguous()
        else:
            assert colors[_cam_view].shape[1] == 3
            cf_flows3 = None
        cf_depths = torch.index_select(depths[_cam_view], 0, sorted_gs_ids[_cam_view])[None, :].contiguous()
        pix_coords = ascend_render.pix_coord.reshape(
                        ascend_render.padded_height // tile_size, tile_size,
                        ascend_render.padded_width // tile_size, tile_size, 2
                    ).permute(0, 2, 1, 3, 4).reshape(
                        ascend_render.padded_height // tile_size * ascend_render.padded_width // tile_size, tile_size * tile_size, 2
                    ).permute(0, 2, 1).to(torch.float32).contiguous()
        nums = torch.cat([tile_offsets[_cam_view][:1], tile_offsets[_cam_view][1:] - tile_offsets[_cam_view][:-1]])
        # lb_sched composition: [cumsum of tile counts per vector core, corresponding tile ids, corresponding tile offsets]
        lb_sched = torch.tensor(get_render_schedule(nums.cpu(), get_num_vector_core()), dtype=torch.int64, device=means2d.device)

        # render color
        cf_render_colors, cf_render_depths, _ = CalcRender.apply(cf_means2, cf_conics0, cf_conics1, cf_conics2,
                                                                 cf_opacity, cf_colors3, cf_depths, pix_coords,
                                                                 lb_sched)
        # render alpha
        cf_colors3_for_alphas = torch.ones_like(cf_colors3)
        cf_render_alphas, cf_render_depths, _ = CalcRender.apply(cf_means2, cf_conics0, cf_conics1, cf_conics2,
                                                                 cf_opacity, cf_colors3_for_alphas, cf_depths, pix_coords,
                                                                 lb_sched)
        cf_render_alphas = cf_render_alphas[0:1]  # any single channel is sufficient

        cf_render_colors = ascend_render.tile2image(cf_render_colors.permute(1, 2, 0), tile_size=tile_size)
        cf_render_depths = ascend_render.tile2image(cf_render_depths.permute(1, 2, 0), tile_size=tile_size)
        cf_render_alphas = ascend_render.tile2image(cf_render_alphas.permute(1, 2, 0), tile_size=tile_size)

        if cf_flows3 is not None:
            # render flow
            cf_render_flows, _, _ = CalcRender.apply(cf_means2, cf_conics0, cf_conics1, cf_conics2,
                                                     cf_opacity, cf_flows3, cf_depths, pix_coords,
                                                     lb_sched)
            cf_render_flows = ascend_render.tile2image(cf_render_flows.permute(1, 2, 0), tile_size=tile_size)
            cf_render_colors = torch.cat([cf_render_colors, cf_render_flows, cf_render_depths], dim=0)
        else:
            # format alignment using dummy_flows
            dummy_flows = cf_render_colors.detach()
            cf_render_colors = torch.cat([cf_render_colors, dummy_flows, cf_render_depths], dim=0)

        render_colors.append(cf_render_colors.permute(1, 2, 0))
        render_alphas.append(cf_render_alphas.permute(1, 2, 0))
    render_colors = torch.stack(render_colors)
    render_alphas = torch.stack(render_alphas)
    # depth uses alpha normalization, accuracy needs further verification
    # if render_mode in ["ED", "RGB+ED"]:
    #     # normalize the accumulated depth to get the expected depth
    #     render_colors = torch.cat(
    #         [
    #             render_colors[..., :-1],
    #             # render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
    #             render_colors[..., -1:] / render_alphas.clamp(min=1e-6),
    #         ],
    #         dim=-1,
    #     )

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opacities,
        "tile_width": ascend_render.padded_width // tile_size,
        "tile_height": ascend_render.padded_height // tile_size,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
    }

    # keep output format consistent with gsplat.rendering.rasterization
    return render_colors, render_alphas, meta


def ascend_rasterization(
    ascend_render: AscendGaussRender,
    means: Tensor,  # [N, 3] or [T, N, 3]
    quats: Tensor,  # [N, 4] or [T, N, 4]
    scales: Tensor,  # [N, 3] or [T, N, 3]
    opacities: Tensor,  # [N] or [T, N]
    colors: Tensor,  # [N, D] or [T, N, D]
    viewmats: Tensor,  # [C, 4, 4] or [C, B, 4, 4]
    Ks: Tensor,  # [C, 3, 3] or [C, B, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: int = 64,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    batch_per_iter: int = 100,
    packed: bool = False,
    radius_clip: float = 0.0,
) -> Tuple[Tensor, Tensor, Dict]:
    """A version of rasterization() that utilies on PyTorch's autograd.

    .. note::
        This function still relies on gsplat's CUDA backend for some computation, but the
        entire differentiable graph is on of PyTorch (and nerfacc) so could use Pytorch's
        autograd for backpropagation.

    .. note::
        This function relies on installing latest nerfacc, via:
        pip install git+https://github.com/nerfstudio-project/nerfacc

    .. note::
        Compared to rasterization(), this function does not support some arguments such as
        `packed`, `sparse_grad` and `absgrad`.
    """

    is_batched = len(means.shape) == 3
    if is_batched:
        assert len(quats.shape) == len(scales.shape) == len(colors.shape) == 3
        assert len(opacities.shape) == 2
        assert len(viewmats.shape) == len(Ks.shape) == 4
        assert means.shape[0] == quats.shape[0] == scales.shape[0] == colors.shape[0] == opacities.shape[0] \
            == viewmats.shape[0] == Ks.shape[0]
    else:
        assert len(means.shape) == len(quats.shape) == len(scales.shape) == len(colors.shape) == 2
        assert len(opacities.shape) == 1
        assert len(viewmats.shape) == len(Ks.shape) == 3
        assert means.shape[0] == quats.shape[0] == scales.shape[0] == colors.shape[0] == opacities.shape[0]
        assert viewmats.shape[0] == Ks.shape[0]

    if is_batched:
        V = means.shape[0]
        render_colors = []
        render_alphas = []
        for v in range(V):
            means_single_view = means[v]
            quats_single_view = quats[v]
            scales_single_view = scales[v]
            colors_single_view = colors[v]
            opacities_single_view = opacities[v]
            viewmats_single_view = viewmats[v]
            Ks_single_view = Ks[v]

            render_colors_single_view, render_alphas_single_view, _ = ascend_rasterization_single_view(
                ascend_render,
                means_single_view,
                quats_single_view,
                scales_single_view,
                opacities_single_view,
                colors_single_view,
                viewmats_single_view,
                Ks_single_view,
                width,
                height,
                near_plane,
                far_plane,
                eps2d,
                sh_degree,
                tile_size,
                backgrounds,
                render_mode,
                rasterize_mode,
                channel_chunk,
                camera_model,
                batch_per_iter,
                packed,
                radius_clip
            )
            render_colors.append(render_colors_single_view)
            render_alphas.append(render_alphas_single_view)
        render_colors = torch.stack(render_colors)
        render_alphas = torch.stack(render_alphas)
        # keep output format consistent with gsplat.rendering.rasterization
        return render_colors, render_alphas, None
        # return render_colors, None, None
    else:
        return ascend_rasterization_single_view(
            ascend_render,
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
            near_plane,
            far_plane,
            eps2d,
            sh_degree,
            tile_size,
            backgrounds,
            render_mode,
            rasterize_mode,
            channel_chunk,
            camera_model,
            batch_per_iter,
            packed,
            radius_clip
        )

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

import acl
from meta_gauss_render import (
    projection_three_dims_gaussian_fused,
    spherical_harmonics,
    flash_gaussian_build_mask,
    gaussian_sort, calc_render,
    get_render_schedule_cpp
)


def validate_inputs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    render_mode: str,
    sh_degree: Optional[int],
    N: int,
    C: int
) -> None:
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [N, D] or [C, N, D]
        assert (colors.dim() == 2 and colors.shape[0] == N) or (
            colors.dim() == 3 and colors.shape[:2] == (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3
        ) or (
            colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape


class Rasterizer:
    def __init__(self, tile_size=32, camera_model="pinhole") -> None:
        self.tile_size = tile_size
        self.camera_model=camera_model

    def tile2image(self, rendered_image, height, width, channel_dim=3):
        rendered_image = rendered_image.reshape(math.ceil(self.padded_height/self.tile_size),
                                                math.ceil(self.padded_width/self.tile_size), self.tile_size, self.tile_size, -1)
        rendered_image = rendered_image.permute(0,2,1,3,4)
        rendered_image = rendered_image.reshape(math.ceil(self.padded_height/self.tile_size)*self.tile_size,
                                                math.ceil(self.padded_width/self.tile_size)*self.tile_size, -1)
        return rendered_image.permute(2, 0, 1)[:, :height, :width]

    def ascend_rasterize_splats(
        self,
        viewmats: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        splats: dict,
        colors,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        tile_size = self.tile_size
        if not hasattr(self, "tile_grid"):
            tile_size = self.tile_size
            self.padded_width = math.ceil(width/tile_size)*tile_size
            self.padded_height = math.ceil(height/tile_size)*tile_size
            self.tile_grid = torch.stack(torch.meshgrid(torch.arange(0, self.padded_height, tile_size), \
                                                        torch.arange(0, self.padded_width, tile_size), indexing='ij'),dim=-1).view(-1,2).to(splats["means"].device)
            self.pix_coord = torch.stack(torch.meshgrid(torch.arange(self.padded_width), torch.arange(self.padded_height), indexing='xy'), dim=-1).to(splats["means"].device)

        means = splats["means"]  # [N, 3]
        quats = splats["quats"]  # [N, 4]
        scales = splats["scales"]  # [N, 3]
        opacities = splats["opacities"]  # [N,]

        flow = None
        if colors.shape[-1] == 6:  # split rgb and flow
            flow = colors[:, 3:]
            colors = colors[:, :3]

        rasterize_mode = "classic"
        render_colors, render_depths, render_alphas, info = self._ascend_rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            tile_size=tile_size,
            rasterize_mode=rasterize_mode,
            camera_model="pinhole",
            **kwargs,
        )

        if flow is not None:
            render_flows, _, _, _ = self._ascend_rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=flow,
                viewmats=viewmats,  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                tile_size=tile_size,
                rasterize_mode=rasterize_mode,
                camera_model="pinhole",
                **kwargs,
            )
            render_colors = torch.cat([render_colors, render_flows, render_depths], dim=-1)
        else:
            render_colors = torch.cat([render_colors, render_depths], dim=-1)

        return render_colors, render_alphas, info

    def inverse_cov2d_v2(self, cov2_00, cov2_01, cov2_11, scale=1):
        det = cov2_00 * cov2_11 - cov2_01 * cov2_01
        inv_x_0 = cov2_11 / det * scale
        inv_x_1 = -cov2_01 / det * scale
        inv_x_2 = cov2_00 / det * scale
        return inv_x_0, inv_x_1, inv_x_2

    def _ascend_rasterization(
        self,
        means: Tensor,  # [..., N, 3]
        quats: Tensor,  # [..., N, 4]
        scales: Tensor,  # [..., N, 3]
        opacities: Tensor,  # [..., N]
        colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        eps2d: float = 0.3,
        sh_degree: Optional[int] = None,
        tile_size: int = 32,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    ) -> Tuple[Tensor, Tensor, Dict]:
        """A version of rasterization() that utilizes on PyTorch's autograd.
        Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

        .. note::
            This function relies on gsplat's CUDA backend for some computation, but the
            entire differentiable graph is built with PyTorch (and nerfacc), so
            back-propagation is handled by PyTorch's autograd.

        ..note::
            Compared to rasterization(), this function does not support some arguments such as
            `packed`, `sparse_grad` and `absgrad`.
        """

        N = means.shape[0]
        C = viewmats.shape[0]
        B = 1
        validate_inputs(means, quats, scales, opacities, colors, viewmats, Ks, render_mode, sh_degree, N, C)

        colors = colors[:, None]
        if sh_degree is None:
            # Colors are post-activation values, with shape [N, D] or [C, N, D]
            if colors.dim() == 2:
                # Turn [N, D] into [C, N, D]
                colors = colors.expand(C, -1, -1)
            else:
                # colors is already [C, N, D]
                pass
        else:
            # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3]
            camtoworlds = torch.inverse(viewmats) # [C, 4, 4]
            if colors.dim() == 3:
                # Turn [N, K, 3] into [C, N, K, 3]
                shs = colors.expand(C, -1, -1, -1)
            else:
                # colors is already [C, N, K, 3]
                shs = colors

            def build_color(means3D, shs, sh_degree, camera_center, B):
                rays_o = camera_center
                rays_d = means3D - rays_o
                rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
                # color = eval_sh(sh_degree, shs.permute(1, 2, 0), rays_d.transpose(0, 1))
                k = (sh_degree + 1) ** 2
                color = spherical_harmonics(sh_degree, rays_d.reshape(B, N, 3), shs[:, :k, :].reshape(B, N, k, 3))

                color = (color + 0.5).clip(min=0.0)
                return color
            colors = build_color(means3D=means,
                                shs=shs[0],
                                sh_degree=sh_degree,
                                camera_center=camtoworlds[0, :3, 3],
                                B = B)

        # NOTE: when sh_degree is not None, shape of colors is (1, 3, N), when sh_degree is None, shape of colors is (N, 1, 3)
        if sh_degree is None:
            '''
            colors shape is inconsistent when sh_degree is None vs non-None
            colors shape in projection_three_dims_gaussian_fused needs to match the shape when sh_degree is non-None
            otherwise it will cause rendering accuracy issues
            '''
            colors = colors.permute(1, 2, 0).contiguous()

        means2d, depths, conics, opacities, radius, covars2d, colors, cnt = projection_three_dims_gaussian_fused(
            means.reshape(B,N,3),
            colors.contiguous(),
            None,
            quats.reshape(B,N,4),
            scales.reshape(B,N,3),
            opacities.reshape(B,N),
            viewmats.reshape(B,C,4,4).contiguous(),
            Ks.reshape(B,C,3,3),
            width,
            height,
            0.3,
            # 0.2
            near_plane
        )

        camera_ids, gaussian_ids = None, None

        # ascend gauss render sorting
        with torch.no_grad():
            mask = flash_gaussian_build_mask(means2d, opacities[None,:], conics, covars2d,
                                            cnt[None,:], self.tile_grid.float(),
                                            width, height, tile_size)
            sorted_gs_ids = []
            tile_offsets = []
            for _cam_view in range(0, C):
                cf_sorted_gs_ids, cf_tile_offsets = gaussian_sort(mask[0,_cam_view],depths[0,_cam_view])
                sorted_gs_ids.append(cf_sorted_gs_ids)
                tile_offsets.append(cf_tile_offsets)
        render_colors = []
        render_depths = []
        render_alphas = []
        for _cam_view in range(0, C):
            cf_means2 = means2d[0, _cam_view]
            cf_colors3 = colors[0,_cam_view]
            cf_opacity = opacities[0, _cam_view]

            inv_x_0 = conics[0,_cam_view,0,:]
            inv_x_1 = conics[0,_cam_view,1,:]
            inv_x_2 = conics[0,_cam_view,2,:]

            cf_depths = depths[_cam_view]

            padded_height = self.padded_height
            padded_width = self.padded_width
            pix_coords = self.pix_coord.reshape(padded_height//tile_size, tile_size, padded_width//tile_size, tile_size, 2) \
                .permute(0, 2, 1, 3, 4).reshape(padded_height//tile_size*padded_width//tile_size, tile_size*tile_size, 2) \
                .permute(0, 2, 1).to(torch.float32).contiguous()
            # nums: Number of gaussians per tile
            nums = torch.cat([tile_offsets[_cam_view][:1], tile_offsets[_cam_view][1:] - tile_offsets[_cam_view][:-1]])
            # lb_sched: cat[cumsum of tile counts per vector core, corresponding tile ids, corresponding tile offsets]
            lb_sched = get_render_schedule_cpp(nums.cpu().to(torch.int64), acl.get_device_capability(0,1)[0]).clone().detach().to(torch.int64).npu()

            # render color
            cf_render_colors, cf_render_depths = calc_render(cf_means2,
                                                             inv_x_0, inv_x_1, inv_x_2,
                                                             cf_opacity,
                                                             cf_colors3,
                                                             cf_depths,
                                                             pix_coords,
                                                             lb_sched,
                                                             sorted_gs_ids[_cam_view]
                                                             )
            # render alpha
            cf_colors3_for_alphas = torch.ones_like(cf_colors3)
            cf_render_alphas, _ = calc_render(cf_means2,
                                              inv_x_0, inv_x_1, inv_x_2,
                                              cf_opacity,
                                              cf_colors3_for_alphas,
                                              cf_depths,
                                              pix_coords,
                                              lb_sched,
                                              sorted_gs_ids[_cam_view],
                                              )
            cf_render_alphas = cf_render_alphas[0:1]  # Take any one channel

            cf_render_colors = self.tile2image(cf_render_colors.permute(1, 2, 0), height, width)
            cf_render_depths = self.tile2image(cf_render_depths.permute(1, 2, 0), height, width)
            cf_render_alphas = self.tile2image(cf_render_alphas.permute(1, 2, 0), height, width)

            render_colors.append(cf_render_colors.permute(1, 2, 0))
            render_depths.append(cf_render_depths.permute(1, 2, 0))
            render_alphas.append(cf_render_alphas.permute(1, 2, 0))
        render_colors = torch.stack(render_colors)
        render_depths = torch.stack(render_depths)
        render_alphas = torch.stack(render_alphas)

        meta = {
            "camera_ids": camera_ids,
            "gaussian_ids": gaussian_ids,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
            "tile_width": padded_width // tile_size,
            "tile_height": padded_height // tile_size,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
        }
        return render_colors, render_depths, render_alphas, meta


def new_ascend_rasterization(
    ascend_render: Rasterizer,
    means: Tensor,  # [N, 3] or [T, N, 3]
    quats: Tensor,  # [N, 4] or [T, N, 4]
    scales: Tensor,  # [N, 3] or [T, N, 3]
    opacities: Tensor,  # [N] or [T, N]
    colors: Tensor,  # [N, D] or [T, N, D]
    viewmats: Tensor,  # [C, 4, 4] or [T, C, 4, 4]
    Ks: Tensor,  # [C, 3, 3] or [T, C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: int = 32,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    batch_per_iter: int = 100,
    packed: bool = False,
    radius_clip: float = 0.0,
) -> Tuple[Tensor, Tensor, Dict]:

    assert len(means.shape) == 3
    assert len(quats.shape) == len(scales.shape) == len(colors.shape) == 3
    assert len(opacities.shape) == 2
    assert len(viewmats.shape) == len(Ks.shape) == 4
    assert means.shape[0] == quats.shape[0] == scales.shape[0] == colors.shape[0] == opacities.shape[0] \
        == viewmats.shape[0] == Ks.shape[0]

    B = means.shape[0]
    render_colors = []
    render_alphas = []
    for b in range(B):
        means_single_sample = means[b]
        quats_single_sample = quats[b]
        scales_single_sample = scales[b]
        opacities_single_sample = opacities[b]
        colors_single_sample = colors[b]
        viewmats_single_sample = viewmats[b]
        Ks_single_sample = Ks[b]
        splats = {
            "means": means_single_sample,
            "quats": quats_single_sample,
            "scales": scales_single_sample,
            "opacities": opacities_single_sample
        }

        kwargs = {
            "near_plane": near_plane,
            "far_plane": far_plane,
            "eps2d": eps2d,
            "sh_degree": None,
            "render_mode": render_mode,
        }

        V = viewmats_single_sample.shape[0]
        assert Ks_single_sample.shape[0] == V

        cols_single_sample, alphas_single_sample = [], []

        for v in range(V):
            cols_single_view, alphas_single_view, _ = ascend_render.ascend_rasterize_splats(
                viewmats=viewmats_single_sample[v:v + 1].contiguous(),
                Ks=Ks_single_sample[v:v + 1],
                width=width,
                height=height,
                splats=splats,
                colors=colors_single_sample,** kwargs
            )
            cols_single_sample.append(cols_single_view)
            alphas_single_sample.append(alphas_single_view)

        cols_single_sample = torch.cat(cols_single_sample)
        alphas_single_sample = torch.cat(alphas_single_sample)

        render_colors.append(cols_single_sample)
        render_alphas.append(alphas_single_sample)

    render_colors = torch.stack(render_colors)
    render_alphas = torch.stack(render_alphas)

    return render_colors, render_alphas, None

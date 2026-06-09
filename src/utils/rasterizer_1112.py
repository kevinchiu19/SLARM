import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from meta_gauss_render import (
    calc_render,
    gs_sort,
    build_tile_gs_mask,
    projection_three_dims_gaussian_fused,
    get_render_schedule_cpp
)

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit direction
    using hardcoded SH polynomials.
    Work with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1)**2]
        dirs: jnp.ndarrayunit directions [..., 3]
    Returns:
        [...,C]
    """
    assert deg <= 3 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[0] >= coeff

    result = C0 * sh[0, ...]
    if deg > 0:
        x, y, z = dirs[0:1, ...], dirs[1:2, ...], dirs[2:3, ...]
        result = (result -
                C1 * y * sh[1, ...] +
                C1 * z * sh[2, ...] -
                C1 * x * sh[3, ...])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result
                + C2[0] * xy * sh[4, ...]
                + C2[1] * yz * sh[5, ...]
                + C2[2] * (2.0 * zz - xx - yy) * sh[6, ...]
                + C2[3] * xz * sh[7, ...]
                + C2[4] * (xx - yy) * sh[8, ...])

            if deg > 2:
                result = (result
                    + C3[0] * y * (3.0 * xx - yy) * sh[9, ...]
                    + C3[1] * xy * z * sh[10, ...]
                    + C3[2] * y * (4.0 * zz - xx - yy) * sh[11, ...]
                    + C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12, ...]
                    + C3[4] * x * (4.0 * zz - xx - yy) * sh[13, ...]
                    + C3[5] * z * (xx - yy) * sh[14, ...]
                    + C3[6] * x * (xx - 3.0 * yy) * sh[15, ...])

    return result

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
            colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] in (3, 6)
        ) or (
            colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] in (3, 6)
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape


class Rasterizer:
    def __init__(self, tile_size=32, camera_model="pinhole") -> None:
        self.tile_size = tile_size
        self.camera_model=camera_model

    @torch.no_grad()
    def sort_gs(self, all_in_mask, depths):
        num_tile = self.tile_grid.shape[1]
        tile_offset = torch.sum(all_in_mask, dim=0).cumsum(dim=0)                         # [144] mask[N,144] first compute overlapping gaussians per tile, gaussians in each tile are contiguous, so memory position per tile is cumulative count of gaussians, aligned? 411 us
        sorted_gs_ids = torch.zeros(tile_offset[-1], dtype=torch.int32, device=all_in_mask.device)     # Total number of gaussians across all tiles 615us
        for tile_id in range(num_tile):
                prev_offset = tile_offset[tile_id-1] if tile_id > 0 else 0                   # Current tile's tile_offset, i.e., how many gaussians before current tile 56us
                tile_in_mask = all_in_mask[:, tile_id]
                tile_depths = depths[tile_in_mask]                                          # Get depths of all gaussians affecting current tile 589us purely indexing,
                tile_gs_ids =tile_in_mask.nonzero()[:,0]                                         # All gaussians for current tile, assign id based on position among all gaussians 712us nonzero400 to150
                _, local_sort_index = torch.sort(tile_depths, stable=True)
                sorted_gs_ids[prev_offset:tile_offset[tile_id]]  = tile_gs_ids[local_sort_index] # Get these gaussians' positions among all gaussians 129us index

        return sorted_gs_ids, tile_offset

    def tile2image(self, rendered_image, height, width, channel_dim=3):
        rendered_image = rendered_image.reshape(math.ceil(self.padded_height/self.tile_size), math.ceil(self.padded_width/self.tile_size), self.tile_size, self.tile_size,-1)
        rendered_image = rendered_image.permute(0,2,1,3,4)
        rendered_image = rendered_image.reshape(math.ceil(self.padded_height/self.tile_size)*self.tile_size, math.ceil(self.padded_width/self.tile_size)*self.tile_size,-1)
        return rendered_image.permute(2,0,1)[:,:height,:width]

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
        self.tile_size = self.tile_size
        self.padded_width = math.ceil(width/self.tile_size)*self.tile_size
        self.padded_height = math.ceil(height/self.tile_size)*self.tile_size
        self.tile_grid = torch.stack(torch.meshgrid(torch.arange(0, self.padded_height, self.tile_size), \
                                                    torch.arange(0, self.padded_width, self.tile_size), indexing='ij'),dim=-1).view(-1,2).to(splats["means"].device).T.contiguous()
        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(self.padded_width), torch.arange(self.padded_height), indexing='xy'), dim=-1).to(splats["means"].device)

        means = splats["means"]  # [N, 3]
        quats = splats["quats"]  # [N, 4]
        scales = splats["scales"]  # [N, 3]
        opacities = splats["opacities"]  # [N,]

        rasterize_mode = "classic"

        render_colors, render_alphas, info = self._ascend_rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            rasterize_mode=rasterize_mode,
            camera_model=self.camera_model,
            tile_size=self.tile_size,
            **kwargs,
        )
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

        radii_new, means2d_new, depths_new, conics, compensations, filter = projection_three_dims_gaussian_fused(
            means.reshape(B,N,3),
            None,
            quats.reshape(B,N,4),
            scales.reshape(B,N,3),
            opacities.reshape(B,N),
            viewmats.reshape(B,C,4,4),
            Ks.reshape(B,C,3,3),
            width,
            height,
            eps2d,
            near_plane
        )

        batch_idx = 0  # NOTE: batch size = 1, batch_idx = 0
        radii_list = []
        means2d_list = []
        depths_list = []
        opacities_list = []
        conics_list = []
        colors_list = []
        for _cam_view in range(C):
            means2d_list.append(means2d_new[batch_idx][_cam_view][filter[batch_idx][_cam_view]])
            radii_list.append(radii_new[batch_idx][_cam_view][filter[batch_idx][_cam_view]])
            depths_list.append(depths_new[batch_idx][_cam_view][filter[batch_idx][_cam_view]])
            opacities_list.append(opacities[filter[batch_idx][_cam_view]])
            conics_list.append(conics[batch_idx][_cam_view][filter[batch_idx][_cam_view]])
            colors_list.append(colors[filter[batch_idx][_cam_view]])

        camera_ids, gaussian_ids = None, None

        # if compensations is not None:
        #     opacities = opacities * compensations

        assert sh_degree is None  # NOTE: sh_degree is not supported in multi cam/view currently

        # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
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

            # standard PyTorch implementation of spherical_harmonics
            def build_color(means3D, shs, sh_degree, camera_center):
                rays_o = camera_center
                rays_d = means3D - rays_o
                rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
                color = eval_sh(sh_degree, shs.permute(1, 2, 0), rays_d.transpose(0, 1))
                color = (color+0.5).clip(min=0.0).transpose(0, 1)
                return color
            colors = build_color(means3D=means,
                                shs=shs[0],
                                sh_degree=sh_degree,
                                camera_center=camtoworlds[0, :3, 3])


        # ascend gauss render sorting
        with torch.no_grad():
            all_in_mask = [build_tile_gs_mask(
                means2d_list[_cam_view][:, :2].T.contiguous(),
                radii_list[_cam_view].to(torch.float32).max(dim=1)[0],
                self.tile_grid,
                width, height, self.tile_size)  for _cam_view in range(C)]

            sorted_gs_ids = []
            tile_offsets = []
            for _cam_view in range(0, C):
                # cf_sorted_gs_ids_new, cf_tile_offsets_new, ret_code = gs_sort(all_in_mask[_cam_view].contiguous(), depths[_cam_view].contiguous())
                # print("-0- cf_sorted_gs_ids_new", ret_code, cf_sorted_gs_ids_new)
                ret_code =-1
                if ret_code != 0:
                    cf_sorted_gs_ids_new, cf_tile_offsets_new = self.sort_gs(all_in_mask[_cam_view].T.to(torch.bool), depths_list[_cam_view])
                    # print("-1- cf_sorted_gs_ids_new", ret_code, cf_sorted_gs_ids_new)
                # else:
                #   cf_tile_offsets_new = cf_tile_offsets_new.to(torch.long)
                # print("-2- cf_sorted_gs_ids_new", ret_code, cf_sorted_gs_ids_new, "cf_sorted_gs_ids_new.shape", cf_sorted_gs_ids_new.shape)
                sorted_gs_ids.append(cf_sorted_gs_ids_new)
                tile_offsets.append(cf_tile_offsets_new)

        render_colors = []
        render_alphas = []  # add alpha channel storage
        render_depths = []
        for _cam_view in range(0, C):
            cf_means2 = means2d_list[_cam_view][:, :2].transpose(0, 1).contiguous()
            cf_opacity = opacities_list[_cam_view].contiguous()
            inv_x_0 = conics_list[_cam_view][:, 0]
            inv_x_1 = conics_list[_cam_view][:, 1]
            inv_x_2 = conics_list[_cam_view][:, 2]
            cf_depths = depths_list[_cam_view]

            padded_height = self.padded_height
            padded_width = self.padded_width
            pix_coords = self.pix_coord.reshape(padded_height//tile_size, tile_size, padded_width//tile_size, tile_size, 2) \
                .permute(0, 2, 1, 3, 4).reshape(padded_height//tile_size*padded_width//tile_size, tile_size*tile_size, 2) \
                .permute(0, 2, 1).to(torch.float32).contiguous()
            # nums: gs count per tile
            nums = torch.cat([tile_offsets[_cam_view][:1], tile_offsets[_cam_view][1:] - tile_offsets[_cam_view][:-1]])
            # lb_sched: cumsum of tile counts per vector core, corresponding tile ids, corresponding tile offsets
            lb_sched = torch.tensor(get_render_schedule_cpp(nums.cpu().to(torch.int64), 40), dtype=torch.int64, device=means2d_list[_cam_view].device)

            # separate RGB and Flow channels
            cf_colors3 = colors_list[_cam_view][..., :3].transpose(0, 1).contiguous()   #(D,N)
            if colors_list[_cam_view].shape[-1] == 6:
                cf_flows3 = colors_list[_cam_view][..., 3:6].transpose(0, 1).contiguous()   #(D,N)
            else:
                assert colors_list[_cam_view].shape[-1] == 3
                cf_flows3 = None

            # render color
            cf_render_colors, cf_render_depths = calc_render(cf_means2,
                                                            inv_x_0, inv_x_1, inv_x_2,
                                                            cf_opacity.squeeze(dim=-1),
                                                            cf_colors3,
                                                            cf_depths[None, :],
                                                            pix_coords,
                                                            lb_sched,
                                                            sorted_gs_ids[_cam_view],
                                                            )

            # render alpha
            cf_colors3_for_alphas = torch.ones_like(cf_colors3)
            cf_render_alphas, cf_render_depths = calc_render(cf_means2,
                                                            inv_x_0, inv_x_1, inv_x_2,
                                                            cf_opacity.squeeze(dim=-1),
                                                            cf_colors3_for_alphas,
                                                            cf_depths[None, :],
                                                            pix_coords,
                                                            lb_sched,
                                                            sorted_gs_ids[_cam_view],
                                                            )
            cf_render_alphas = cf_render_alphas[0:1]  # any single channel is sufficient

            cf_render_colors = self.tile2image(cf_render_colors.permute(1, 2, 0), height, width)
            cf_render_depths = self.tile2image(cf_render_depths.permute(1, 2, 0), height, width)
            cf_render_alphas = self.tile2image(cf_render_alphas.permute(1, 2, 0), height, width)

            if cf_flows3 is not None:
                # render flow
                cf_render_flows, _ = calc_render(cf_means2,
                                                inv_x_0, inv_x_1, inv_x_2,
                                                cf_opacity.squeeze(dim=-1),
                                                cf_flows3,
                                                cf_depths[None, :],
                                                pix_coords,
                                                lb_sched,
                                                sorted_gs_ids[_cam_view],
                                                )
                cf_render_flows = self.tile2image(cf_render_flows.permute(1, 2, 0), height, width)
                cf_render_colors = torch.cat([cf_render_colors, cf_render_flows, cf_render_depths], dim=0)
            else:
                cf_render_colors = torch.cat([cf_render_colors, cf_render_depths], dim=0)

            # TODO: check depth norm
            # depth accumulation has no physical meaning, needs alpha normalization for perceptual expected depth
            # cf_last_cumsum = self.ascend_render.tile2image(cf_last_cumsum.unsqueeze(0).permute(1,2,0), tile_size=tile_size)
            # cf_render_alphas = 1 - torch.exp(cf_last_cumsum)
            # cf_render_depths = cf_render_depths / cf_render_alphas.clamp(min=1e-10)

            render_colors.append(cf_render_colors.permute(1, 2, 0))
            render_depths.append(cf_render_depths.permute(1, 2, 0))
            render_alphas.append(cf_render_alphas.permute(1, 2, 0))
        render_colors = torch.stack(render_colors)
        render_depths = torch.stack(render_depths)
        render_alphas = torch.stack(render_alphas)

        meta = {
            "camera_ids": camera_ids,
            "gaussian_ids": gaussian_ids,
            "radii": radii_list,
            "means2d": means2d_list,
            "depths": depths_list,
            "conics": conics_list,
            "opacities": opacities_list,
            "tile_width": padded_width // tile_size,
            "tile_height": padded_height // tile_size,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
        }
        return render_colors, render_alphas, meta

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
            opacities_single_view = opacities[v]
            colors_single_view = colors[v]
            viewmats_single_view = viewmats[v]
            Ks_single_view = Ks[v]
            splats = {
                "means": means_single_view,
                "quats": quats_single_view,
                "scales": scales_single_view,
                "opacities": opacities_single_view
            }

            kwargs = {
                "near_plane": near_plane,
                "far_plane": far_plane,
                "eps2d": eps2d,
                "sh_degree": None,
                "render_mode": render_mode,
            }

            cols_single_view, alphas_single_view, info_single_view = ascend_render.ascend_rasterize_splats(
                viewmats=viewmats_single_view.contiguous(),
                Ks=Ks_single_view,
                width=width,
                height=height,
                splats=splats,
                colors=colors_single_view,** kwargs
            )

            render_colors.append(cols_single_view)
            render_alphas.append(alphas_single_view)

        render_colors = torch.stack(render_colors)
        render_alphas = torch.stack(render_alphas)

        return render_colors, render_alphas, None

    else:
        splats = {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities
        }

        kwargs = {
            "near_plane": near_plane,
            "far_plane": far_plane,
            "eps2d": eps2d,
            "sh_degree": None,
            "render_mode": render_mode,
        }
        cols, alphas, info = ascend_render.ascend_rasterize_splats(
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            splats=splats,
            colors=colors,
            **kwargs
        )
        return cols, alphas, info


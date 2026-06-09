import functools
import math
import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from torch_scatter import scatter_max, scatter_add

from tools import is_ascend_npu
if is_ascend_npu():
    render_op_version = os.getenv("RENDER_OP_VERSION", "1212")  # 0730, 1112, 1212
    if render_op_version == '1112':
        from src.utils.rasterizer_1112 import Rasterizer, new_ascend_rasterization
    elif render_op_version == '1212':
        from src.utils.rasterizer_1212 import Rasterizer, new_ascend_rasterization
    elif render_op_version == '0730':
        from meta_gauss_render import AscendGaussRender
        from src.utils.npu_rendering import ascend_rasterization
    else:
        raise ValueError(f"Unsupported RENDER_OP_VERSION: {render_op_version}")
else:
    from gsplat.rendering import rasterization, rasterization_2dgs
from .decoder import ConvDecoder, DummyDecoder, ModulatedLinearLayer
from .embedders import PluckerEmbedder, TimestepEmbedder
from .layers import LayerNorm2d, Mlp
from .vit import VisionTransformer as ViT
from .components.utils.geometry import unproject_depth_map_to_point_map, angular_velocity_to_quaternion, quaternion_multiply, angle_axis_to_quaternion


class STORM(ViT):
    def __init__(
        self,
        img_size=224,
        in_chans=9,
        gs_dim=3,
        decoder_type="dummy",
        near=0.2,
        far=400,
        opacity_offset=-2.0,
        num_cams=3,  # to ablate
        max_scale=0.5,
        use_ms3_motion=False,
        gs_marbles=False,
        add_angular_velocity=False,
        render_context_view=False,
        render_context_frame_contribution=False,
        use_render_novel_view=False,
        pred_gs_conf=False,
        voxelize=False,
        voxel_size=0.2,
        disable_pos_embed=False,
        use_sky_token=True,
        use_affine_token=True,
        num_motion_tokens=32,
        tau=0.5,
        projected_motion_dim=32,
        # ViT parameters
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        grad_checkpointing=True,
        sigmoid_rgb=True,  # False, # a legacy oversight: the sigmoid was accidentally omitted in the earlier implementation
        **kwargs,
    ):
        super(STORM, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            grad_checkpointing=grad_checkpointing,
        )
        # basic attributes
        self.disable_pos_embed = disable_pos_embed
        self.gs_dim = gs_dim

        self.render_context_view = render_context_view
        self.render_context_frame_contribution = render_context_frame_contribution

        self.use_render_novel_view = use_render_novel_view

        self.pred_gs_conf = pred_gs_conf
        self.voxelize = voxelize
        self.voxel_size = voxel_size
        if self.voxelize:
            assert self.pred_gs_conf, 'Voxelization requires gs confidence calculation weights.'

        # base gaussian parameters
        self.gs_params_name = ["depth", "scales", "quats", "opacitys", "colors"]
        self.gs_params_size = [1, 3, 4, 1, self.gs_dim]
        self.out_channels = sum(self.gs_params_size)

        if self.pred_gs_conf:
            self.out_channels += 1
            self.gs_params_name.append("confs")
            self.gs_params_size.append(1)

        self.num_cams = num_cams
        self.grad_checkpointing = grad_checkpointing

        # ------- STORM v.s. Latent-STORM -------
        self.decoder_type = decoder_type
        self.decoder_upsample_ratio = decoder_upsample_ratio = self.patch_size

        # ------- motion predictor -------
        self.num_motion_tokens = num_motion_tokens
        self.tau = tau
        self.add_angular_velocity = add_angular_velocity

        # ------- embedders -------
        self.plucker_embedder = PluckerEmbedder(img_size=img_size)
        self.time_embedder = TimestepEmbedder(embed_dim)

        # ------- auxiliary tokens -------
        self.use_sky_token = use_sky_token
        self.use_affine_token = use_affine_token

        if self.use_sky_token:
            self.sky_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.sky_head = ModulatedLinearLayer(
                3,
                hidden_channels=512,
                condition_channels=embed_dim,
                out_channels=self.gs_dim,
            )

        if self.use_affine_token:
            self.affine_token = nn.Parameter(torch.randn(1, self.num_cams, embed_dim) * 0.02)
            self.affine_linear = nn.Linear(embed_dim, self.gs_dim * (self.gs_dim + 1))

        # ------- gs predictor and mask decoder -------
        if decoder_type == "dummy":
            self.gs_pred = nn.Linear(embed_dim, decoder_upsample_ratio**2 * self.out_channels)
            self.decoder = DummyDecoder()
            self.unpatch_size = decoder_upsample_ratio

            if self.decoder_upsample_ratio == 8:
                # used for upscaling the low-resolution image features to the pixel-resolution
                # very handcrafted and never tuned
                self.output_upscaling = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
                    LayerNorm2d(512),
                    nn.GELU(),
                    nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                    LayerNorm2d(256),
                    nn.GELU(),
                    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                    LayerNorm2d(128),
                    nn.GELU(),
                )
            elif self.decoder_upsample_ratio == 16:
                # used for upscaling the low-resolution image features to the pixel-resolution
                # very handcrafted and never tuned
                self.output_upscaling = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
                    LayerNorm2d(512),
                    nn.GELU(),
                    nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                    LayerNorm2d(256),
                    nn.GELU(),
                    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                    LayerNorm2d(128),
                    nn.GELU(),
                    nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                    LayerNorm2d(128),
                    nn.GELU(),
                )

        elif decoder_type == "conv":
            self.gs_pred = nn.Linear(embed_dim, self.out_channels)
            # latent-STORM decoder
            self.decoder = ConvDecoder(
                latent_dim=self.gs_dim,
                out_channels=4,  # 3 for RGB, 1 for depth
                num_res_blocks=3,
                channels=[512, 256, 256, 128],  # 8 times upsample
                grad_checkpointing=grad_checkpointing,
            )
            self.unpatch_size = 1
            # upscaling the low-resolution image features to the pixel-resolution
            # the "pixel" resolution here is essentially the feature map resolution
            # which is 1/patch_size of the image resolution
            self.output_upscaling = nn.Sequential(
                nn.Conv2d(embed_dim, 512, kernel_size=1),
                LayerNorm2d(512),
                nn.GELU(),
                nn.Conv2d(512, 256, kernel_size=1),
                LayerNorm2d(256),
                nn.GELU(),
                nn.Conv2d(256, 128, kernel_size=1),
                LayerNorm2d(128),
                nn.GELU(),
            )
        # ------- activation functions for gs parameters -------
        self.gs_marbles = gs_marbles
        self.max_scale = nn.Parameter(torch.tensor([float(max_scale)]), requires_grad=False)
        self.scale_offset = float(torch.log(torch.tensor([self.max_scale])))  # NOTE: learn from large to small

        if self.gs_marbles:
            # gs marbles
            self.scale_act_fn = lambda x: torch.minimum(torch.exp(x.mean(-1, True).expand_as(x) + self.scale_offset), self.max_scale)
            self.quat_act_fn = lambda x: x.new_tensor((1, 0, 0, 0)).expand(*x.shape[:-1], 4)
        else:
            # gs anisotropic
            self.scale_act_fn = lambda x: torch.minimum(torch.exp(x + self.scale_offset), self.max_scale)
            # self.scale_act_fn = lambda x: torch.minimum(torch.exp(x -2.3), self.max_scale)  # old storm
            self.quat_act_fn = lambda x: x  # NOTE: gsplat normalizes internally, so F.normalize(x, dim=-1) is not needed

        self.opacity_act_fn = lambda x: torch.sigmoid(x + opacity_offset)
        self.depth_act_fn = lambda x: near + torch.sigmoid(x) * (far - near)
        # self.rgb_act_fn = lambda x: torch.sigmoid(x) * 2 - 1 if sigmoid_rgb else x  # NOTE: if normalize rgb need * 2 - 1
        self.rgb_act_fn = lambda x: torch.sigmoid(x) if sigmoid_rgb else x

        if self.pred_gs_conf:
            self.gs_conf_act_fn = lambda x: torch.sigmoid(x)

        self.near, self.far = near, far

        self.use_ms3_motion = use_ms3_motion

        # ------- motion predictor -------
        self.motion_key_head = Mlp(128, 256, projected_motion_dim)
        if self.use_ms3_motion:
            self.ms3_deg = 3
            self.omega_deg = 3
            self.ms3_factorials = torch.tensor([math.factorial(i+1) for i in range(self.ms3_deg)])
            self.omega_factorials = torch.tensor([math.factorial(i+1) for i in range(self.omega_deg)])
            self.ms3_deg_downmax_mult = 8.0
            self.sigmoid_ms3_bias = -6.9068
            self.sigmoid_ms3_min = 0.0
            self.sigmoid_ms3_max = 100  # 2.0
            self.ms3_clamp = 0.0001
            num_velocity_channels = 4 * self.ms3_deg
            if self.add_angular_velocity:
                num_velocity_channels += 4 * self.omega_deg
        else:
            num_velocity_channels = 3
            if self.add_angular_velocity:
                num_velocity_channels += 3
        if self.num_motion_tokens > 0:
            self.motion_tokens = nn.Parameter(torch.randn(1, num_motion_tokens, embed_dim) * 0.02)
            self.motion_query_heads = nn.ModuleList(
                [
                    Mlp(embed_dim, embed_dim, projected_motion_dim)
                    for _ in range(self.num_motion_tokens)
                ]
            )
            self.motion_basis_decoder = Mlp(embed_dim, 256, num_velocity_channels)
        else:
            self.motion_basis_decoder = Mlp(projected_motion_dim, 256, num_velocity_channels)

        if is_ascend_npu():
            # Ascend NPU rendering does not support latest gsplat and 2DGS
            render_op_version = os.getenv("RENDER_OP_VERSION", "1112")  # 0730, 1112, 1212
            if render_op_version in ['1112', '1212']:
                self.gs_renderer_npu = Rasterizer(tile_size=32, camera_model='pinhole')
                self.rasterization_func = functools.partial(
                    new_ascend_rasterization,
                    ascend_render=self.gs_renderer_npu
                )
            elif render_op_version == '0730':
                self.gs_renderer_npu = AscendGaussRender(width=self.img_size[1], height=self.img_size[0],
                                                        active_sh_degree=0, isect_mode='flashgs', cpu_radix_sort=True)
                                                        #  active_sh_degree=0, isect_mode='original', cpu_radix_sort=True)
                self.gs_renderer_npu.tile_grid = self.gs_renderer_npu.tile_grid.to('npu')
                self.gs_renderer_npu.pix_coord = self.gs_renderer_npu.pix_coord.to('npu')
                self.rasterization_func = functools.partial(ascend_rasterization, ascend_render=self.gs_renderer_npu)
            else:
                raise ValueError(f"Unsupported RENDER_OP_VERSION: {render_op_version}!")
        else:
            self.rasterization_func = rasterization

        self.init_weights()
        if disable_pos_embed:  # remove the default pos_embed in vit
            del self.pos_embed
            self.pos_embed = None

    def _pos_embed(self, x: Tensor) -> Tensor:
        if not self.disable_pos_embed:
            return super()._pos_embed(x)
        return rearrange(x, "b h w c -> b (h w) c")

    def _time_embed(self, x: Tensor, time: Tensor, num_views=1) -> Tensor:
        if time.ndim == 3:
            b, t, v = time.shape
            time_embedding = (
                self.time_embedder(time.flatten())  # (bt, c)
                .view(b, t, v, -1)  # (b, t, v, c)
                .view(-1, 1, self.embed_dim)  # (btv, 1, c)
                .repeat(1, x.shape[1], 1)  # (btv, n, c)
            )
        else:
            time_embedding = (
                self.time_embedder(time.flatten())  # (bt, c)
                .view(time.shape[0], time.shape[1], 1, -1)  # (b, t, 1, c)
                .repeat(1, 1, num_views, 1)  # (b, t, v, c)
                .view(-1, 1, self.embed_dim)  # (btv, 1, c)
                .repeat(1, x.shape[1], 1)  # (btv, n, c)
            )
        return x + time_embedding

    def forward_decoder(self, render_results):
        render_results["rgb_key"] = "rendered_image"
        render_results["depth_key"] = "rendered_depth"
        render_results["alpha_key"] = "rendered_alpha"
        render_results["flow_key"] = "rendered_flow"
        render_results["decoder_depth_key"] = None
        render_results["decoder_alpha_key"] = None
        render_results["decoder_flow_key"] = None
        render_results = self.decoder(render_results)
        decoded_depth_key = render_results["decoder_depth_key"]
        if decoded_depth_key is not None:
            decoded_depth = self.depth_act_fn(render_results[decoded_depth_key])
            render_results[decoded_depth_key] = decoded_depth
        return render_results

    def forward_features(self, x, plucker_embeds, time):
        b, t, v, c, h, w = x.size()
        x = rearrange(x, "b t v c h w -> (b t v) c h w")
        plucker_embeds = rearrange(plucker_embeds, "b t v h w c-> (b t v) c h w")
        x = torch.cat([x, plucker_embeds], dim=1)
        x = self.patch_embed(x)  # (b t v) h w c2  [48, 9, 160, 240] -> [48, 20, 30, 768]
        x = self._pos_embed(x)  # (b t v) (h w) c2  resample_abs_pos_embed add [48, 600, 768]
        x = self._time_embed(x, time, num_views=v)  # add [48, 600, 768]
        x = rearrange(x, "(b t v) hw c -> b (t v hw) c", t=t, v=v)  # [4, 7200, 768]
        if self.num_motion_tokens > 0:
            motion_tokens = repeat(self.motion_tokens, "1 k d -> b k d", b=x.shape[0])  # [4, 16, 768]
            x = torch.cat([motion_tokens, x], dim=-2)
        if self.use_affine_token:
            affine_token = repeat(self.affine_token, "1 k d -> b k d", b=b)  # [4, 3, 768]
            x = torch.cat([affine_token, x], dim=-2)
        if self.use_sky_token:
            sky_token = repeat(self.sky_token, "1 1 d -> b 1 d", b=x.shape[0])  # [4, 1, 768]
            x = torch.cat([sky_token, x], dim=-2)
        x = self.transformer(x)
        x = self.norm(x)
        return x

    def forward_motion_predictor(self, x, motion_tokens=None, gs_params=None):
        b, t, v, h, w, _ = gs_params["means"].shape
        img_embeds = self.unpatchify(
            rearrange(x, "b (t v hw) c -> (b t v) hw c", t=t, v=v),
            hw=(h // self.unpatch_size, w // self.unpatch_size),
            patch_size=1,
        )  # [4, 7200, 768] -> [48, 768, 20, 30]
        # NOTE: using checkpoint here causes errors
        # if self.grad_checkpointing:
        #     img_embeds = checkpoint(self.output_upscaling, img_embeds)  # [48, 768, 20, 30] -> [48, 128, 160, 240]
        # else:
        img_embeds = self.output_upscaling(img_embeds)
        img_embeds = rearrange(img_embeds, "(b t v) c h w -> b t v h w c", t=t, v=v)  # [4, 4, 3, 160, 240, 128]
        img_keys = self.motion_key_head(img_embeds)  # [4, 4, 3, 160, 240, 128] -> [4, 4, 3, 160, 240, 32]

        if self.num_motion_tokens > 0:
            hyper_in_list = []
            for i in range(self.num_motion_tokens):
                hyper_in = self.motion_query_heads[i](motion_tokens[:, i])  # [4, 768] -> [4, 32]
                hyper_in_list.append(hyper_in)
            motion_token_queries = torch.stack(hyper_in_list, dim=1)  # [4, 16, 32]
            dot_product_similarity = torch.einsum(
                "b k c, b t v h w c -> b t v h w k",
                motion_token_queries,  # [1, 16, 32]
                img_keys,  # [1, 4, 3, 160, 240, 32]
            )  # [1, 4, 3, 160, 240, 16]
            motion_weights = torch.softmax(dot_product_similarity / self.tau, dim=-1)
            motion_bases = self.motion_basis_decoder(motion_tokens)  # [4, 16, 768] -> [4, 16, 3]
            motion_final = torch.einsum(
                "b t v h w k, b k c -> b t v h w c", motion_weights, motion_bases
            )
            gs_params["motion_weights"] = motion_weights
            gs_params["motion_bases"] = motion_bases
        else:
            # if there's no motion token, directly predict the velocity from the upsampled image features
            motion_final = self.motion_basis_decoder(img_keys)

        if self.use_ms3_motion:
            if self.add_angular_velocity:
                ms3, omega = motion_final.split([4 * self.ms3_deg, 4 * self.omega_deg], dim=-1)
                forward_ms3 = torch.concat([self.decode_flow(ms3), self.decode_flow(omega)], dim=-1)
            else:
                ms3 = motion_final
                forward_ms3 = self.decode_flow(ms3)
            gs_params["forward_ms3"] = forward_ms3
        else:
            gs_params["forward_flow"] = motion_final
        return {k: v for k, v in gs_params.items() if v is not None}

    def decode_flow(self, ms3):
        # Extract degree of marginal scale (number of scale components)
        ms3_deg = ms3.shape[-1] // 4
        # Extract speed components (every 4th element starting from index 3)
        speed = ms3[..., 3::4, None]  # [B, T, H, W, ms3_deg, 1]
        # Reshape spatial components (first 3 of every 4 elements)
        ms3 = torch.cat(
            [ms3[..., None, i * 4 : i * 4 + 3] for i in range(ms3_deg)], dim=-2
        )  # [B, T, H, W, ms3_deg, 3]

        # Rescale speed with sigmoid and apply clamping threshold
        speed = (speed + self.sigmoid_ms3_bias).sigmoid() * (
            self.sigmoid_ms3_max - self.sigmoid_ms3_min
        ) + self.sigmoid_ms3_min
        speed = (speed - self.ms3_clamp).clamp(0)  # Zero out speeds below threshold

        # Apply decay factor to speed based on scale level
        # Higher scale levels get progressively smaller speeds
        speed = torch.cat(
            [speed[..., i : i + 1, :] / self.ms3_deg_downmax_mult**i
                for i in range(ms3_deg)], dim=-2
        )  # [B, T, H, W, ms3_deg, 1]

        # Apply speed-modulated normalized marginal scales
        ms3 = speed * F.normalize(ms3[..., :3], dim=-1)  # Normalize and modulate by speed
        ms3 = ms3.reshape(ms3.shape[:-2] + (-1,))  # Flatten to [B, T, H, W, ms3_deg*3]
        return ms3

    def voxelizaton_using_confidence(self, gs_xyz, gs_conf, voxel_size):
        voxel_indices = (gs_xyz / voxel_size).round().int()  # [N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(voxel_indices, dim=0, return_inverse=True, return_counts=True)

        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(gs_conf, inverse_indices, dim=0)
        conf_exp = torch.exp(gs_conf - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0)  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1)  # [N, 1]

        return weights, inverse_indices

    def pad_tensor_list(self, tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def forward_gs_predictor(self, x, origins, directions, gt_depth=None):
        b, t, v, h, w, _ = origins.shape  # [4, 4, 3, 160, 240, 3]
        x = rearrange(x, "b (t v hw) c -> (b t v) hw c", t=t, v=v)
        gs_params = self.gs_pred(x)  # [48, 600, 768] -> [48, 600, 768(12*8*8)]
        gs_params = self.unpatchify(gs_params, hw=(h, w), patch_size=self.unpatch_size)  # [48, 12, 160, 240]
        gs_params = rearrange(gs_params, "(b t v) c h w -> b t v h w c", t=t, v=v)
        gs_params_dict = dict(zip(self.gs_params_name, gs_params.split(self.gs_params_size, dim=-1)))
        if gt_depth is not None:
            depths = gt_depth[..., None]
        else:
            depths = self.depth_act_fn(gs_params_dict["depth"])
        scales = self.scale_act_fn(gs_params_dict["scales"])
        quats = self.quat_act_fn(gs_params_dict["quats"])
        opacitys = self.opacity_act_fn(gs_params_dict["opacitys"])
        colors = self.rgb_act_fn(gs_params_dict["colors"])
        means = origins + directions * depths
        output =  {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacitys.squeeze(-1),
            "colors": colors,
            "depths": depths.squeeze(-1),
        }
        if self.pred_gs_conf:
            confs = self.gs_conf_act_fn(gs_params_dict["confs"])
            output["confs"] = confs
        return output

    def forward_renderer_context_view(self, gs_params, data_dict, radius_clip=0.0):
        b, t, v, h, w, _ = gs_params["means"].shape
        tgt_h, tgt_w = data_dict["height"], data_dict["width"]
        tgt_t, tgt_v = data_dict["context_camtoworlds"].shape[1:3]
        means = rearrange(gs_params["means"], "b t v h w c -> (b t v) (h w) c")
        scales = rearrange(gs_params["scales"], "b t v h w c -> (b t v) (h w) c")
        quats = rearrange(gs_params["quats"], "b t v h w c -> (b t v) (h w) c")
        opacities = rearrange(gs_params["opacities"], "b t v h w -> (b t v) (h w)")
        colors = rearrange(gs_params["colors"], "b t v h w c -> (b t v) (h w) c")

        camtoworlds_batched = data_dict["context_camtoworlds"].view(b * tgt_t * v, -1, 4, 4)
        viewmats_batched = torch.linalg.inv(camtoworlds_batched.float())
        Ks_batched = data_dict["context_intrinsics"].view(b * tgt_t * v, -1, 3, 3)

        with torch.autocast("cuda", enabled=False):
            rendered_color, rendered_alpha, _ = self.rasterization_func(
                means=means.float(),
                quats=quats.float(),
                scales=scales.float(),
                opacities=opacities.float(),
                colors=colors.float(),
                viewmats=viewmats_batched,
                Ks=Ks_batched,
                width=tgt_w,
                height=tgt_h,
                render_mode="RGB+ED",
                near_plane=self.near,
                far_plane=self.far,
                packed=False,
                radius_clip=radius_clip,
            )
        color, depth = rendered_color.split([self.gs_dim, 1], dim=-1)

        output_dict = {
            "rendered_image": color.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "rendered_depth": depth.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
            "rendered_alpha": rendered_alpha.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
        }
        return output_dict

    def forward_renderer(self, gs_params, data_dict, render_motion_seg=not is_ascend_npu(), radius_clip=0.0, idx=None):
        b, t, v, h, w, _ = gs_params["means"].shape
        tgt_h, tgt_w = data_dict["height"], data_dict["width"]
        tgt_t, tgt_v = data_dict["target_camtoworlds"].shape[1:3]
        means = rearrange(gs_params["means"], "b t v h w c -> b (t v h w) c")
        scales = rearrange(gs_params["scales"], "b t v h w c -> b (t v h w) c")
        quats = rearrange(gs_params["quats"], "b t v h w c -> b (t v h w) c")
        opacities = rearrange(gs_params["opacities"], "b t v h w -> b (t v h w)")
        colors = rearrange(gs_params["colors"], "b t v h w c -> b (t v h w) c")

        means_batched = means.repeat_interleave(tgt_t, dim=0)
        scales_batched = scales.repeat_interleave(tgt_t, dim=0)
        quats_batched = quats.repeat_interleave(tgt_t, dim=0)
        opacities_batched = opacities.repeat_interleave(tgt_t, dim=0)
        color_batched = colors.repeat_interleave(tgt_t, dim=0)

        ctx_time = data_dict["context_time"] * data_dict["timespan"]  # [1, 4, 3]
        tgt_time = data_dict["target_time"] * data_dict["timespan"]   # [1, 20, 3]
        if tgt_time.ndim == 3:
            tdiff_forward = tgt_time.unsqueeze(2) - ctx_time.unsqueeze(1)  # [1, 20, 4, 3]
            tdiff_forward = tdiff_forward.view(b * tgt_t, t * v, 1)
            tdiff_forward_batched = tdiff_forward.repeat_interleave(h * w, dim=1)  # [20, 460800, 1]
        else:
            tdiff_forward = tgt_time.unsqueeze(-1) - ctx_time.unsqueeze(-2)
            tdiff_forward = tdiff_forward.view(b * tgt_t, t, 1)
            tdiff_forward_batched = tdiff_forward.repeat_interleave(v * h * w, dim=1)

        if not self.use_ms3_motion:
            forward_v = rearrange(gs_params["forward_flow"], "b t v h w c -> b (t v h w) c")
            if self.add_angular_velocity:
                forward_v, forward_angular_v = forward_v.split([3, 3], dim=-1)
            forward_v_batched = forward_v.repeat_interleave(tgt_t, dim=0)
            if self.add_angular_velocity:
                forward_angular_v_batched = forward_angular_v.repeat_interleave(tgt_t, dim=0)

            forward_translation = forward_v_batched * tdiff_forward_batched

            # means = context frame + offset cur
            means_batched = means_batched + forward_translation

            if self.add_angular_velocity:
                # rotation_offset = (wx, wy, wz) * dt
                quats_offset_batched = angular_velocity_to_quaternion(forward_angular_v_batched, tdiff_forward_batched)
                # new rotation = rotation + rotation_offset
                quats_batched = quaternion_multiply(quats_batched, quats_offset_batched)
        else:
            forward_ms3 = gs_params["forward_ms3"][..., :self.ms3_deg*3]
            forward_ms3 = rearrange(forward_ms3, "b t v h w c -> b (t v h w) c")
            forward_ms3_batched = forward_ms3.repeat_interleave(tgt_t, dim=0)
            if self.add_angular_velocity:
                forward_omega = gs_params["forward_ms3"][..., -self.omega_deg*3:]
                forward_omega = rearrange(forward_omega, "b t v h w c -> b (t v h w) c")
                forward_omega_batched = forward_omega.repeat_interleave(tgt_t, dim=0)
                # angular velocity
                angle_axis_offset_batched = torch.stack(
                    [forward_omega_batched[..., i * 3 : (i + 1) * 3] * tdiff_forward_batched ** (i + 1) / self.omega_factorials[i] \
                    for i in range(self.omega_deg)]
                ).sum(0)
                quats_offset_batched = angle_axis_to_quaternion(angle_axis_offset_batched)
                quats_batched = quaternion_multiply(quats_batched, quats_offset_batched)

            # offset cur
            forward_translation_cur = torch.stack(
                [forward_ms3_batched[..., i * 3 : (i + 1) * 3] * tdiff_forward_batched ** (i + 1) / self.ms3_factorials[i] \
                for i in range(self.ms3_deg)]
            ).sum(0)

            delta_time = float(1 / data_dict['fps'])

            # offset next
            forward_translation_next = torch.stack(
                [forward_ms3_batched[..., i * 3 : (i + 1) * 3] * (tdiff_forward_batched + delta_time) ** (i + 1) / self.ms3_factorials[i] \
                for i in range(self.ms3_deg)]
            ).sum(0)

            # means = context frame + offset cur
            means_batched = means_batched + forward_translation_cur

            # velocity: cur (t) -> next (t+1)
            forward_v_batched = (forward_translation_next - forward_translation_cur) / delta_time

            gs_params["forward_flow"] = gs_params["forward_ms3"]

        if not self.training:  # mask out some noisy flow
            forward_v_batched[forward_v_batched.norm(dim=-1) < 1.0] = 0.0

        # Visualize the effect of each context frame
        if idx is not None:
            means_batched = means_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            scales_batched = scales_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            quats_batched = quats_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            opacities_batched = opacities_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            color_batched = color_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            forward_v_batched = forward_v_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]

        if self.voxelize and self.pred_gs_conf:
            gs_confs = rearrange(gs_params["confs"], "b t v h w c -> b (t v h w) c")
            gs_confs_batched = gs_confs.repeat_interleave(tgt_t, dim=0)
            if idx is not None:
                gs_confs_batched = gs_confs_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]

            gs_attrs = {
                'means': means_batched,
                'scales': scales_batched,
                'quats': quats_batched,
                'opacities': opacities_batched.unsqueeze(-1),
                'color': color_batched,
                'forward_v': forward_v_batched,
            }
            # if feats is not None:
            #     gs_attrs['feats'] = feats_batched

            gs_attrs_b_t_lists = {attr: [] for attr in gs_attrs.keys()}
            for b_idx in range(b):
                for t_idx in range(tgt_t):
                    b_t_idx = b_idx * tgt_t + t_idx

                    # Voxelize using a specific voxelsize, and calculate the weight by confidence
                    weights, inverse_indices = self.voxelizaton_using_confidence(gs_attrs['means'][b_t_idx],
                                                                                 gs_confs_batched[b_t_idx].squeeze(1),
                                                                                #  life_span_coef[b_t_idx],  # TODO: check whether to use gs conf or lifespan (opacity)
                                                                                 self.voxel_size)

                    # Loop through each gaussian attribute
                    for name, attr in gs_attrs.items():
                        # Compute weighted average of gaussian attribute
                        weighted_attrs_b_t = attr[b_t_idx] * weights  # TODO: aggregate on feature dimension or after activation
                        # Aggregate per voxel
                        voxel_attrs_b_t = scatter_add(weighted_attrs_b_t, inverse_indices, dim=0)

                        gs_attrs_b_t_lists[name].append(voxel_attrs_b_t)

            # NOTE: dynamic shape
            max_voxels = max(f.shape[0] for attr_b_t in gs_attrs_b_t_lists.values() for f in attr_b_t)
            min_voxels = min(f.shape[0] for attr_b_t in gs_attrs_b_t_lists.values() for f in attr_b_t)

            # Padding
            gs_attrs_voxel_padded = {attr: None for attr in gs_attrs.keys()}
            # Loop through each gaussian attribute
            for name, gs_attrs_b_t_list in gs_attrs_b_t_lists.items():
                gs_attrs_voxel_padded[name] = self.pad_tensor_list(
                    gs_attrs_b_t_list, (max_voxels,), value=0.0
                )

            '''
            print('Original gaussian count      ', means_batched.shape[1])
            print('Max gaussian count after pruning', max_voxels)
            print('Min gaussian count after pruning', min_voxels)
            '''

            # Apply gs pruning results
            means_batched = gs_attrs_voxel_padded['means']
            scales_batched = gs_attrs_voxel_padded['scales']
            quats_batched = gs_attrs_voxel_padded['quats']
            opacities_batched = gs_attrs_voxel_padded['opacities'].squeeze(-1)
            color_batched = gs_attrs_voxel_padded['color']
            forward_v_batched = gs_attrs_voxel_padded['forward_v']
            # if feats is not None:
            #     feats_batched = gs_attrs_voxel_padded['feats']

        colors_batched = torch.cat([color_batched, forward_v_batched], dim=-1)

        if not self.training and self.num_motion_tokens > 0 and render_motion_seg:
            # render the motion segmentation map
            motion_weights = rearrange(gs_params["motion_weights"], "b t v h w k -> b (t v h w) k")
            weights_batched = motion_weights.repeat_interleave(tgt_t, dim=0)
            if idx is not None:
                weights_batched = weights_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]

            colors_batched = torch.cat([colors_batched, weights_batched], dim=-1)

        camtoworlds_batched = data_dict["target_camtoworlds"].view(b * tgt_t, -1, 4, 4)
        viewmats_batched = torch.linalg.inv(camtoworlds_batched.float())
        Ks_batched = data_dict["target_intrinsics"].view(b * tgt_t, -1, 3, 3)

        motion_seg = None
        if True:
            if not self.training:
                with torch.autocast("cuda", enabled=False):
                    rendered_color, rendered_alpha, _ = self.rasterization_func(
                        means=means_batched.float(),
                        quats=quats_batched.float(),
                        scales=scales_batched.float(),
                        opacities=opacities_batched.float(),
                        colors=(
                            colors_batched[..., : -self.num_motion_tokens].float()
                            if self.num_motion_tokens > 0 and render_motion_seg
                            else colors_batched.float()
                        ),
                        viewmats=viewmats_batched,
                        Ks=Ks_batched,
                        width=tgt_w,
                        height=tgt_h,
                        render_mode="RGB+ED",  # render color with expected depth
                        near_plane=self.near,
                        far_plane=self.far,
                        packed=False,
                        radius_clip=radius_clip,
                    )
                    color, forward_flow, depth = rendered_color.split([self.gs_dim, 3, 1], dim=-1)
                    if self.num_motion_tokens > 0 and render_motion_seg:
                        chunksize = 32
                        assignment_map = []
                        rendered_colors = colors_batched[..., -self.num_motion_tokens :]
                        for i in range(0, self.num_motion_tokens, chunksize):
                            weights, _, _ = self.rasterization_func(
                                means=means_batched.float(),
                                quats=quats_batched.float(),
                                scales=scales_batched.float(),
                                opacities=opacities_batched.float(),
                                colors=rendered_colors[..., i : i + chunksize],
                                viewmats=viewmats_batched,
                                Ks=Ks_batched,
                                width=tgt_w,
                                height=tgt_h,
                                render_mode="RGB+ED",
                                near_plane=self.near,
                                far_plane=self.far,
                                packed=False,
                                radius_clip=radius_clip,
                            )
                            weights = weights.split([weights.size(-1) - 1, 1], dim=-1)[0]
                            assignment_map.append(weights)
                        motion_seg = torch.cat(assignment_map, dim=-1)
                        motion_seg = motion_seg.reshape(b, tgt_t, tgt_v, tgt_h, tgt_w, -1).argmax(
                            dim=-1
                        )
            else:
                with torch.autocast("cuda", enabled=False):
                    rendered_color, rendered_alpha, _ = self.rasterization_func(
                        means=means_batched.float(),
                        quats=quats_batched.float(),
                        scales=scales_batched.float(),
                        opacities=opacities_batched.float(),
                        colors=colors_batched.float(),
                        viewmats=viewmats_batched,
                        Ks=Ks_batched,
                        width=tgt_w,
                        height=tgt_h,
                        render_mode="RGB+ED",
                        near_plane=self.near,
                        far_plane=self.far,
                        packed=False,
                        radius_clip=radius_clip,
                    )
                color, forward_flow, depth = rendered_color.split([self.gs_dim, 3, 1], dim=-1)
        output_dict = {
            "rendered_image": color.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "rendered_depth": depth.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
            "rendered_alpha": rendered_alpha.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
            "rendered_flow": forward_flow.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "means_batched": means_batched,
        }
        if motion_seg is not None:
            output_dict["rendered_motion_seg"] = motion_seg.squeeze(-1)
        return output_dict

    def get_ray_dict(self, data_dict):
        ray_dict = self.plucker_embedder(
            data_dict["context_intrinsics"],
            data_dict["context_camtoworlds"],
            image_size=data_dict["context_image"].shape[-2:],
        )
        if self.decoder_type != "dummy":
            feat_ray_dict = self.plucker_embedder(
                data_dict["context_intrinsics"],
                data_dict["context_camtoworlds"],
                image_size=data_dict["context_image"].shape[-2:],
                patch_size=self.patch_size,
            )
            ray_dict["origins"] = feat_ray_dict["origins"]
            ray_dict["dirs"] = feat_ray_dict["dirs"]

            tgt_intrinsics = data_dict["target_intrinsics"]
            tgt_intrinsics[..., 0, 0] = tgt_intrinsics[..., 0, 0] / self.patch_size
            tgt_intrinsics[..., 1, 1] = tgt_intrinsics[..., 1, 1] / self.patch_size
            tgt_intrinsics[..., 0, 2] = tgt_intrinsics[..., 0, 2] / self.patch_size
            tgt_intrinsics[..., 1, 2] = tgt_intrinsics[..., 1, 2] / self.patch_size
            data_dict["target_intrinsics"] = tgt_intrinsics
            data_dict["width"] //= self.patch_size
            data_dict["height"] //= self.patch_size
        return data_dict, ray_dict

    def apply_novelview_rt(self, data_dict,
                                 degree_x=0,
                                 degree_y=-10,
                                 degree_z=0,
                                 trans_x=-7,
                                 trans_y=-5,
                                 trans_z=2):
        def rotation_matrix_y(theta_degrees):
            theta = np.radians(theta_degrees)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])

        def rotation_matrix_z(theta_degrees):
            theta = np.radians(theta_degrees)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])

        def rotation_matrix_x(theta_degrees):
            theta = np.radians(theta_degrees)  # convert degrees to radians
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return np.array([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ])

        R_z = torch.from_numpy(rotation_matrix_z(degree_z))[None, None, None, ...].to(torch.float32).cuda()
        R_y = torch.from_numpy(rotation_matrix_y(degree_y))[None, None, None, ...].to(torch.float32).cuda()
        R_x = torch.from_numpy(rotation_matrix_x(degree_x))[None, None, None, ...].to(torch.float32).cuda()

        # translation
        data_dict["target_camtoworlds"][..., :3, 3] += torch.tensor([[[[trans_x, trans_y, trans_z]]]]).cuda()
        # rotation
        data_dict["target_camtoworlds"][..., :3, :3] @= (R_z @ R_y @ R_x)

    def forward(self, data_dict):
        x = data_dict["context_image"]
        b, t, v, c, h, w = x.size()
        data_dict, ray_dict = self.get_ray_dict(data_dict)
        x = self.forward_features(x, ray_dict["plucker"], data_dict["context_time"])

        sky_token, affine_tokens, motion_tokens = None, None, None
        if self.use_sky_token:
            sky_token = x[:, :1]
            x = x[:, 1:]

        if self.use_affine_token:
            affine_tokens = x[:, : self.num_cams]
            x = x[:, self.num_cams :]

        if self.num_motion_tokens > 0:
            motion_tokens = x[:, : self.num_motion_tokens]
            x = x[:, self.num_motion_tokens :]

        gs_params = self.forward_gs_predictor(x, ray_dict["origins"], ray_dict["dirs"])
        # gs_params = self.forward_gs_predictor(x, ray_dict["origins"], ray_dict["dirs"], gt_depth=data_dict['context_depth'])
        gs_params = self.forward_motion_predictor(x, motion_tokens, gs_params)
        # sometimes the number of views is too large, so we split the rendering into chunks
        step = 20

        if not self.training and self.use_render_novel_view:
            self.apply_novelview_rt(data_dict, degree_x=1.5, degree_y=0, degree_z=0, trans_x=-5, trans_y=0, trans_z=1)

        if data_dict["target_camtoworlds"].shape[1] <= step:
            # Rendering the results of context frame aggregation.
            render_results = self.forward_renderer(gs_params, data_dict)

            # Rendering every context frame (only itself, no aggregation) in target view.
            if not self.training and self.render_context_frame_contribution:
                for idx in range(t):
                    context_render_results = self.forward_renderer(gs_params, data_dict, idx=idx)
                    render_results[f'context_{idx}_rendered_image'] = context_render_results['rendered_image']
                    render_results[f'context_{idx}_rendered_depth'] = context_render_results['rendered_depth']
                    render_results[f'context_{idx}_rendered_alpha'] = context_render_results['rendered_alpha']
                    render_results[f'context_{idx}_rendered_flow'] = context_render_results['rendered_flow']
                del context_render_results
        else:
            chunk_data_dict = data_dict.copy()
            for chunk_start in range(0, data_dict["target_camtoworlds"].shape[1], step):
                chunk_end = min(chunk_start + step, data_dict["target_camtoworlds"].shape[1])
                chunk_data_dict["target_camtoworlds"] = data_dict["target_camtoworlds"][
                    :, chunk_start:chunk_end
                ]
                chunk_data_dict["target_intrinsics"] = data_dict["target_intrinsics"][
                    :, chunk_start:chunk_end
                ]
                chunk_data_dict["target_time"] = data_dict["target_time"][:, chunk_start:chunk_end]
                chunk_render_results = self.forward_renderer(gs_params, chunk_data_dict)
                if chunk_start == 0:
                    render_results = chunk_render_results
                else:
                    for k, v in chunk_render_results.items():
                        render_results[k] = torch.cat([render_results[k], v], dim=1)

                # Rendering every context frame (only itself, no aggregation) in target view.
                if not self.training and self.render_context_frame_contribution:
                    for idx in range(t):
                        chunk_context_render_results = self.forward_renderer(gs_params, chunk_data_dict, idx=idx)
                        if chunk_start == 0:
                            render_results[f'context_{idx}_rendered_image'] = chunk_context_render_results['rendered_image']
                            render_results[f'context_{idx}_rendered_depth'] = chunk_context_render_results['rendered_depth']
                            render_results[f'context_{idx}_rendered_alpha'] = chunk_context_render_results['rendered_alpha']
                            render_results[f'context_{idx}_rendered_flow'] = chunk_context_render_results['rendered_flow']
                        else:
                            render_results[f'context_{idx}_rendered_image'] = torch.cat([render_results[f'context_{idx}_rendered_image'], chunk_context_render_results['rendered_image']], dim=1)
                            render_results[f'context_{idx}_rendered_depth'] = torch.cat([render_results[f'context_{idx}_rendered_depth'], chunk_context_render_results['rendered_depth']], dim=1)
                            render_results[f'context_{idx}_rendered_alpha'] = torch.cat([render_results[f'context_{idx}_rendered_alpha'], chunk_context_render_results['rendered_alpha']], dim=1)
                            render_results[f'context_{idx}_rendered_flow'] = torch.cat([render_results[f'context_{idx}_rendered_flow'], chunk_context_render_results['rendered_flow']], dim=1)
                        del chunk_context_render_results

        images, opacities = render_results["rendered_image"], render_results["rendered_alpha"]

        # Rendering every context frame in its own view.
        if self.render_context_view:
            context_render_results = self.forward_renderer_context_view(gs_params, data_dict)
            context_images, context_depths, context_opacities = context_render_results['rendered_image'].clone(), context_render_results['rendered_depth'].clone(), context_render_results['rendered_alpha'].clone()

        if self.use_sky_token:
            target_ray_dict = self.plucker_embedder(
                data_dict["target_intrinsics"],
                data_dict["target_camtoworlds"],
                image_size=(data_dict["height"], data_dict["width"]),
            )
            if data_dict["target_camtoworlds"].shape[1] <= step:
                # target
                target_sky = self.sky_head(target_ray_dict["dirs"], sky_token)
                images = images + (1 - opacities[..., None]) * target_sky
                # per-context in target view
                if not self.training and self.render_context_frame_contribution:
                    for idx in range(t):
                        render_results[f'context_{idx}_rendered_image'] = render_results[f'context_{idx}_rendered_image'] + (1 - render_results[f'context_{idx}_rendered_alpha'][..., None]) * target_sky
            else:
                for chunk_start in range(0, data_dict["target_camtoworlds"].shape[1], step):
                    target_dirs = target_ray_dict["dirs"][:, chunk_start : chunk_start + step]
                    chunk_target_sky = self.sky_head(target_dirs, sky_token)
                    images[:, chunk_start : chunk_start + step] += (
                        1 - opacities[:, chunk_start : chunk_start + step][..., None]
                    ) * chunk_target_sky
                    # per-context in target view
                    if not self.training and self.render_context_frame_contribution:
                        for idx in range(t):
                            render_results[f'context_{idx}_rendered_image'][:, chunk_start : chunk_start + step] += (1 - render_results[f'context_{idx}_rendered_alpha'][:, chunk_start : chunk_start + step][..., None]) * chunk_target_sky
            if self.render_context_view:
                context_ray_dict = self.plucker_embedder(
                    data_dict["context_intrinsics"],
                    data_dict["context_camtoworlds"],
                    image_size=(data_dict["height"], data_dict["width"]),
                )
                if data_dict["context_camtoworlds"].shape[1] <= step:
                    context_sky = self.sky_head(context_ray_dict["dirs"], sky_token)
                    context_images = context_images + (1 - context_opacities[..., None]) * context_sky
                else:
                    for chunk_start in range(0, data_dict["context_camtoworlds"].shape[1], step):
                        context_dirs = context_ray_dict["dirs"][:, chunk_start : chunk_start + step]
                        chunk_context_sky = self.sky_head(context_dirs, sky_token)
                        context_images[:, chunk_start : chunk_start + step] += (
                            1 - context_opacities[:, chunk_start : chunk_start + step][..., None]
                        ) * chunk_context_sky
            gs_params["sky_token"] = sky_token

        if self.use_affine_token:
            ''' old storm
            affine = self.affine_linear(affine_tokens)  # b v (gs_dim * (gs_dim + 1))
            affine = rearrange(affine, "b v (p q) -> b v p q", p=self.gs_dim)
            images = torch.einsum("b t v h w p, b v p q -> b t v h w p", images, affine)
            context_images = torch.einsum("b t v h w p, b v p q -> b t v h w p", context_images, affine)

            if not self.training:
                for idx in range(t):
                    render_results[f'context_{idx}_rendered_image'] = torch.einsum("b t v h w p, b v p q -> b t v h w p", render_results[f'context_{idx}_rendered_image'], affine)
            '''

            affine = self.affine_linear(affine_tokens)  # b v (gs_dim * (gs_dim + 1))
            affine_matrix = rearrange(affine, "b v (p q) -> b v p q", p=self.gs_dim)
            linear_part = affine_matrix[..., :3]  # b, v, 3, 3
            translation_part = affine_matrix[..., 3]  # b, v, 3
            translation_part = translation_part.view(b, 1, v, 1, 1, 3)
            gs_params["images_without_affine"] = images.clone()  # TODO: whether to add regularization to keep it consistent with the original image
            # apply linear and translation
            images = torch.einsum('btvhwi,bvij->btvhwj', images, linear_part) + translation_part
            if self.render_context_view:
                context_images = torch.einsum('btvhwi,bvij->btvhwj', context_images, linear_part) + translation_part
            if not self.training and self.render_context_frame_contribution:
                for idx in range(t):
                    render_results[f'context_{idx}_rendered_image'] = torch.einsum('btvhwi,bvij->btvhwj', \
                        render_results[f'context_{idx}_rendered_image'], linear_part) + translation_part
            gs_params["affine"] = affine
        render_results["rendered_image"] = images
        render_results = self.forward_decoder(render_results)
        output = {
            "ray_dict": ray_dict,
            "gs_params": gs_params,
            "render_results": render_results,
            "pred_context_depth": gs_params['depths'],
        }
        # context
        if self.render_context_view:
            output["rendered_context_image"] = context_images
            output["rendered_context_depth"] = context_depths
            output["rendered_context_alpha"] = context_opacities
        return output

    def from_gs_params_to_output(self, gs_params, target_dict, num_cams=1):
        # self.render_novel_view(gs_params, target_dict)
        render_results = self.forward_renderer(
            gs_params, target_dict, render_motion_seg=not is_ascend_npu(), radius_clip=0.0
        )
        rendered_images = render_results["rendered_image"]
        if self.use_sky_token:
            sky_token = gs_params["sky_token"]
            target_ray_dict = self.plucker_embedder(
                target_dict["target_intrinsics"],
                target_dict["target_camtoworlds"],
                image_size=(target_dict["height"], target_dict["width"]),
            )
            sky = self.sky_head(target_ray_dict["dirs"], sky_token)  # [1, 20, 3, 160, 240, 3]
            rendered_opacities = render_results["rendered_alpha"]
            rendered_images = rendered_images + (1 - rendered_opacities[..., None]) * sky

        if self.use_affine_token:
            if num_cams == 1:
                affine = gs_params["affine"].mean(dim=1)
                rendered_images = torch.einsum(
                    "b t v h w p, b p q -> b t v h w p", rendered_images, affine
                )
            else:
                affine = gs_params["affine"]
                rendered_images = torch.einsum(
                    "b t v h w p, b v p q -> b t v h w p", rendered_images, affine
                )
        render_results["rendered_image"] = rendered_images
        render_results = self.forward_decoder(render_results)
        return {"render_results": render_results}

    def render_novel_view(self, gs_params, target_dict, num_cams=1):
        # test
        import copy
        target_dict_ = copy.deepcopy(target_dict)

        import numpy as np
        def rotation_matrix_y(theta_degrees):
            theta = np.radians(theta_degrees)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])

        def rotation_matrix_z(theta_degrees):
            theta = np.radians(theta_degrees)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])

        def rotation_matrix_x(theta_degrees):
            theta = np.radians(theta_degrees)  # convert degrees to radians
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return np.array([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ])

        degree = -10
        R_z = torch.from_numpy(rotation_matrix_z(degree))[None, None, None, ...].to(torch.float32).cuda()
        R_y = torch.from_numpy(rotation_matrix_y(degree))[None, None, None, ...].to(torch.float32).cuda()
        R_x = torch.from_numpy(rotation_matrix_x(degree))[None, None, None, ...].to(torch.float32).cuda()

        target_dict = copy.deepcopy(target_dict_)
        # translation
        target_dict["target_camtoworlds"][..., :3, 3] += torch.tensor([[[[-25, 5, 10]]]]).cuda()
        # rotation
        target_dict["target_camtoworlds"][..., :3, :3] @= R_x

        render_results = self.forward_renderer(
            gs_params, target_dict, render_motion_seg=not is_ascend_npu(), radius_clip=4.0
        )
        rendered_images = render_results["rendered_image"]
        if self.use_sky_token:
            sky_token = gs_params["sky_token"]
            target_ray_dict = self.plucker_embedder(
                target_dict["target_intrinsics"],
                target_dict["target_camtoworlds"],
                image_size=(target_dict["height"], target_dict["width"]),
            )
            sky = self.sky_head(target_ray_dict["dirs"], sky_token)  # [1, 20, 3, 160, 240, 3]
            rendered_opacities = render_results["rendered_alpha"]
            rendered_images = rendered_images + (1 - rendered_opacities[..., None]) * sky

        if self.use_affine_token:
            if num_cams == 1:
                affine = gs_params["affine"].mean(dim=1)
                rendered_images = torch.einsum(
                    "b t v h w p, b p q -> b t v h w p", rendered_images, affine
                )
            else:
                affine = gs_params["affine"]
                rendered_images = torch.einsum(
                    "b t v h w p, b v p q -> b t v h w p", rendered_images, affine
                )
        MEAN = [0.5, 0.5, 0.5]
        STD = [0.5, 0.5, 0.5]
        mean = torch.tensor([[MEAN]]).to(rendered_images.device)
        std = torch.tensor([[STD]]).to(rendered_images.device)

        from matplotlib import pyplot as plt
        plt.imshow((rendered_images[0, 0, 1].to(torch.float32) * std + mean).detach().cpu().numpy())
        plt.savefig('test.png')

        import imageio
        images = (rendered_images[0, :, 1].to(torch.float32) * std + mean).detach().cpu().numpy()

        # reorder dimensions to [30, h, w, 3]
        # images = images.transpose(0, 2, 3, 1)

        # save as MP4
        output_file = 'output.mp4'
        fps = 6
        with imageio.get_writer(output_file, fps=fps) as writer:
            for frame in images:
                writer.append_data(frame)

        print(f"Video saved as {output_file}")

    def get_gs_params(self, data_dict):
        x = data_dict["context_image"]
        data_dict, ray_dict = self.get_ray_dict(data_dict)
        x = self.forward_features(x, ray_dict["plucker"], data_dict["context_time"])

        sky_token, affine_tokens, motion_tokens = None, None, None
        if self.use_sky_token:
            sky_token = x[:, :1]
            x = x[:, 1:]

        if self.use_affine_token:
            affine_tokens = x[:, : self.num_cams]
            x = x[:, self.num_cams :]

        if self.num_motion_tokens > 0:
            motion_tokens = x[:, : self.num_motion_tokens]
            x = x[:, self.num_motion_tokens :]

        gs_params = self.forward_gs_predictor(x, ray_dict["origins"], ray_dict["dirs"])
        # gs_params = self.forward_gs_predictor(x, ray_dict["origins"], ray_dict["dirs"], gt_depth=data_dict['context_depth'])
        gs_params = self.forward_motion_predictor(x, motion_tokens, gs_params)
        if self.use_sky_token:
            gs_params["sky_token"] = sky_token

        if self.use_affine_token:
            affine = self.affine_linear(affine_tokens)  # b v (gs_dim * (gs_dim + 1))
            affine = rearrange(affine, "b v (p q) -> b v p q", p=self.gs_dim)
            gs_params["affine"] = affine
        return gs_params


def STORM_B_8(**kwargs):
    return STORM(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)


def STORM_L_8(**kwargs):
    return STORM(patch_size=8, embed_dim=1024, depth=24, num_heads=16, **kwargs)


def STORM_B_16(**kwargs):
    return STORM(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)


def STORM_L_16(**kwargs):
    return STORM(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)


def STORM_XL_8(**kwargs):
    return STORM(patch_size=8, embed_dim=1152, depth=28, num_heads=16, **kwargs)


def STORM_H_8(**kwargs):
    return STORM(patch_size=8, embed_dim=1280, depth=32, num_heads=16, **kwargs)


def STORM_H_16(**kwargs):
    return STORM(patch_size=16, embed_dim=1280, depth=32, num_heads=16, **kwargs)


STORM_models = {
    "STORM-B/8": STORM_B_8,
    "STORM-L/8": STORM_L_8,
    "STORM-XL/8": STORM_XL_8,
    "STORM-H/8": STORM_H_8,
    "STORM-B/16": STORM_B_16,
    "STORM-L/16": STORM_L_16,
    "STORM-H/16": STORM_H_16,
}

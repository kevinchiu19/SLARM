import os
import copy
import re
import time
import math
import functools
from typing import List

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_max, scatter_add
from einops import rearrange, repeat
from huggingface_hub import PyTorchModelHubMixin

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
from .layers import LayerNorm2d, Mlp
from .components.aggregator.aggregator import Aggregator
from .components.heads.camera_head import CameraHead
from .components.heads.dpt_head import DPTHead
from .components.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from .components.utils.geometry import unproject_depth_map_to_point_map, angular_velocity_to_quaternion, quaternion_multiply, angle_axis_to_quaternion, \
                                        compute_normals_scales_torch, rot_from_normals_torch, scale_from_dxdy_torch
from tools.export_ply import save_ply
from src.dataset.constants import SEMANTIC_LABEL_LIST, SEMANTIC_ID_TO_COLOR
if os.getenv("FEAT_DIST"):
    from tools.feats_tools import get_text_label_feats, feat2class


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class SLARM(nn.Module, PyTorchModelHubMixin):
    '''Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes'''

    output_format='bcthw'
    def __init__(
        self,
        img_size=[168, 252],
        num_cams=3,  # to ablate
        gs_dim=3,
        in_chans=3,
        embed_dim=1024,
        patch_size=14,
        decoder_type="dummy",
        # depth
        near=0.2,
        far=400,
        # model config
        disable_pos_embed=False,
        use_sky_token=True,
        use_affine_token=False,  # TODO: only needed for multiview?
        use_pred_camera_pose=False,
        use_pred_depth=False,
        use_time_token=True,
        add_patch_plucker_embed=True,
        add_camera_embed=True,
        concat_plucker_embed=True,
        shortcut_rgb=True,
        pred_gs_conf=False,
        enable_lifespan=False,
        use_last_token=False,
        enable_depth_head=False,
        enable_camera_head=False,
        enable_point_head=False,
        use_ms3_motion=False,
        add_angular_velocity=False,
        render_context_view=False,
        render_context_frame_contribution=False,
        voxelize=False,
        voxel_size=0.2,
        similarity_probs_threshold=0.2,
        # patch emb
        patch_embed="dinov2_vitl14_reg",
        num_register_tokens=4,
        # attention block
        depth=24,
        # head
        gs_dense_reg_head_type='mlp',   # 'conv'
        feat_dense_reg_head_type='mlp',   # 'conv'
        motion_dense_key_head_type='mlp',   # 'conv'
        num_motion_tokens=0,
        tau=0.5,
        projected_motion_dim=32,
        pred_feat_dim=64,
        with_feat=True,
        # target_feat_types=[],  # pe3r, lseg
        # gs activation
        gs_marbles=False,
        max_scale=0.5,
        opacity_offset=-2.0,
        sigmoid_rgb=True,  # False, # a legacy oversight: the sigmoid was accidentally omitted in the earlier implementation
        # other
        grad_checkpointing=True,
        vggt_pretrained_weight_filepath='',
        use_2dgs=False,
        pesudo_3dgs=False,
        save_gaussian=False,
        gaussian_save_path='output_gs',
        save_rendered_pc=False,
        rendered_pc_save_path = 'output_rendered_pc',
        use_render_novel_view=False,
        # ms3 motion
        ms3_deg=3,
        omega_deg=3,
        ms3_deg_downmax_mult=1.0,
        sigmoid_ms3_bias=-6.9068,
        sigmoid_ms3_min=0.0,
        sigmoid_ms3_max=100,  # 2.0
        ms3_clamp=0.0001,
        # stream
        mode="full", #  default use full attention
        **kwargs,
    ):
        super().__init__()

        # basic attributes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.num_cams = num_cams
        self.gs_dim = gs_dim
        self.depth = depth
        self.num_register_tokens = num_register_tokens
        self.patch_embed = patch_embed

        self.near = near
        self.far = far

        self.use_last_token = use_last_token

        self.enable_depth_head = enable_depth_head
        self.enable_camera_head = enable_camera_head
        self.enable_point_head = enable_point_head
        self.use_pred_camera_pose = use_pred_camera_pose
        if self.use_pred_camera_pose:
            assert self.enable_camera_head
        self.use_pred_depth = use_pred_depth
        if self.use_pred_depth:
            assert self.enable_depth_head

        self.render_context_view = render_context_view
        self.render_context_frame_contribution = render_context_frame_contribution

        self.concat_plucker_embed = concat_plucker_embed
        self.add_patch_plucker_embed = add_patch_plucker_embed
        self.add_camera_embed = add_camera_embed

        self.in_chans = in_chans
        self.shortcut_rgb = shortcut_rgb

        self.gs_dense_reg_head_type = gs_dense_reg_head_type
        self.feat_dense_reg_head_type = feat_dense_reg_head_type
        self.motion_dense_key_head_type = motion_dense_key_head_type

        self.pred_gs_conf = pred_gs_conf
        self.voxelize = voxelize
        self.voxel_size = voxel_size
        if self.voxelize:
            assert self.pred_gs_conf, 'Voxelization requires gs confidence calculation weights.'

        self.similarity_probs_threshold = similarity_probs_threshold

        # base gaussian parameters
        self.gs_params_name = ["depth", "scales", "quats", "opacitys", "colors"]
        self.gs_params_size = [1, 3, 4, 1, self.gs_dim]
        self.out_channels = sum(self.gs_params_size)

        if self.pred_gs_conf:
            self.out_channels += 1
            self.gs_params_name.append("confs")
            self.gs_params_size.append(1)

        self.enable_lifespan = enable_lifespan
        if self.enable_lifespan:
            self.out_channels += 1
            self.gs_params_name.append("lifespans")
            self.gs_params_size.append(1)

        # ------- motion predictor -------
        self.num_motion_tokens = num_motion_tokens
        self.tau = tau
        self.use_ms3_motion = use_ms3_motion
        self.add_angular_velocity = add_angular_velocity
        if self.use_ms3_motion:
            self.ms3_deg = ms3_deg
            self.omega_deg = omega_deg
            self.ms3_factorials = torch.tensor([math.factorial(i+1) for i in range(self.ms3_deg)])
            self.omega_factorials = torch.tensor([math.factorial(i+1) for i in range(self.omega_deg)])
            self.ms3_deg_downmax_mult = ms3_deg_downmax_mult
            self.sigmoid_ms3_bias = sigmoid_ms3_bias
            self.sigmoid_ms3_min = sigmoid_ms3_min
            self.sigmoid_ms3_max = sigmoid_ms3_max
            self.ms3_clamp = ms3_clamp
            num_velocity_channels = 4 * self.ms3_deg
            if self.add_angular_velocity:
                num_velocity_channels += 4 * self.omega_deg
        else:
            num_velocity_channels = 3
            if self.add_angular_velocity:
                num_velocity_channels += 3

        # ------- mode -------
        assert mode == "full" or mode == "causal" or bool(re.match(r'^window_(\d+)$', mode))
        self.mode = mode

        # ------- auxiliary tokens -------
        self.use_time_token = use_time_token
        self.use_sky_token = use_sky_token
        self.use_affine_token = use_affine_token

        self.disable_pos_embed = disable_pos_embed

        self.projected_motion_dim = projected_motion_dim
        self.pred_feat_dim = pred_feat_dim
        # self.target_feat_types = target_feat_types
        # self.with_feat = len(target_feat_types) > 0
        self.with_feat = with_feat

        self.decoder_type = decoder_type
        if self.decoder_type == "dummy":
            self.decoder = DummyDecoder()
        self.decoder_upsample_ratio = decoder_upsample_ratio = self.patch_size

        self.grad_checkpointing = grad_checkpointing
        self.vggt_pretrained_weight_filepath = vggt_pretrained_weight_filepath

        self.use_2dgs = use_2dgs
        self.pesudo_3dgs = pesudo_3dgs

        self.save_gaussian = save_gaussian
        self.gaussian_save_path = gaussian_save_path
        self.save_rendered_pc = save_rendered_pc
        self.rendered_pc_save_path = rendered_pc_save_path

        if is_ascend_npu():
            # Ascend NPU rendering does not support latest gsplat and 2DGS for now
            self.use_2dgs = False
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
            if not self.use_2dgs:
                self.rasterization_func = rasterization
            else:
                self.rasterization_func = rasterization_2dgs

        self.use_render_novel_view = use_render_novel_view

        self.use_reentrant = False # hardcoded to False

        if depth == 12:
            self.intermediate_layer_idx = [2, 5, 8, 11]
        elif depth == 24:
            self.intermediate_layer_idx = [4, 11, 17, 23]
        else:
            raise ValueError('only support depth layer 12 or 24!')

        # Aggregator
        self.aggregator = Aggregator(
                                    in_chans=self.in_chans,
                                    num_cams=self.num_cams,
                                    img_size=self.img_size,
                                    patch_size=self.patch_size,
                                    decoder_type=self.decoder_type,
                                    embed_dim=self.embed_dim,
                                    depth=self.depth,
                                    num_register_tokens=self.num_register_tokens,
                                    patch_embed=self.patch_embed,
                                    num_motion_tokens=self.num_motion_tokens,
                                    use_time_token=self.use_time_token,
                                    use_sky_token=self.use_sky_token,
                                    use_affine_token=self.use_affine_token,
                                    concat_plucker_embed=self.concat_plucker_embed,
                                    add_patch_plucker_embed=self.add_patch_plucker_embed,
                                    add_camera_embed=self.add_camera_embed,
                                    grad_checkpointing=self.grad_checkpointing
                                )

        # 3D annotation head
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if self.use_pred_camera_pose else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, intermediate_layer_idx=self.intermediate_layer_idx, patch_size=self.patch_size,
                                  activation="exp", conf_activation="expp1") if self.use_pred_depth else None # TODO: activation
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, intermediate_layer_idx=self.intermediate_layer_idx, patch_size=self.patch_size,
                                  activation="inv_log", conf_activation="expp1") if self.enable_point_head else None

        # ------- embedders -------
        self.plucker_embedder = self.aggregator.plucker_embedder

        if self.num_motion_tokens > 0:
            self.motion_token_norm = nn.LayerNorm(2 * embed_dim)
            self.motion_query_heads = nn.ModuleList(
                [
                    Mlp(2 * embed_dim, 2 * embed_dim, projected_motion_dim)
                    for _ in range(self.num_motion_tokens)
                ]
            )
            self.motion_basis_decoder = Mlp(2 * embed_dim, 256, num_velocity_channels)
        else:
            self.motion_basis_decoder = Mlp(projected_motion_dim, 256, num_velocity_channels)

        if self.use_affine_token:
            self.affine_token_norm = nn.LayerNorm(2 * embed_dim)
            self.affine_linear = nn.Linear(2 * embed_dim, self.gs_dim * (self.gs_dim + 1))

        if self.use_sky_token:
            self.sky_token_norm = nn.LayerNorm(2 * embed_dim)
            self.sky_head = ModulatedLinearLayer(
                3,
                hidden_channels=512,
                condition_channels=2 * embed_dim,
                out_channels=self.gs_dim,
            )

        if self.use_last_token:
            self.aggregated_last_tokens_norm = nn.LayerNorm(2 * embed_dim)
            # gs head and motion head
            self.__build_gshead_and_motionhead_without_dpthead__(embed_dim=embed_dim, decoder_upsample_ratio=self.decoder_upsample_ratio,
                                                                 projected_motion_dim=self.projected_motion_dim,
                                                                 decoder_type=self.decoder_type, grad_checkpointing=self.grad_checkpointing)
            if self.with_feat:
                self.__build_feat_head_without_dpthead__(embed_dim=embed_dim, decoder_upsample_ratio=self.decoder_upsample_ratio,
                                                                    decoder_type=self.decoder_type)
        else:
            # gs head
            self.dense_feats_dim = 256  # 256 is dpt default feature dim
            self.gs_feature_head = DPTHead(dim_in=2 * embed_dim, feature_only=True, intermediate_layer_idx=self.intermediate_layer_idx, patch_size=self.patch_size)

            if self.gs_dense_reg_head_type == 'mlp':
                if self.shortcut_rgb:
                    self.gs_dense_reg_head = Mlp(self.dense_feats_dim + 3, 2 * self.dense_feats_dim, self.out_channels)
                else:
                    self.gs_dense_reg_head = Mlp(self.dense_feats_dim, 2 * self.dense_feats_dim, self.out_channels)
            elif self.gs_dense_reg_head_type == 'conv':
                if self.shortcut_rgb:
                    self.gs_dense_reg_head = nn.Sequential(
                            nn.Conv2d(self.dense_feats_dim + 3, (2 * embed_dim) // 16, kernel_size=3, stride=1, padding=1),
                            nn.GELU(),
                            nn.Conv2d((2 * embed_dim) // 16, (2 * embed_dim) // 32, kernel_size=3, stride=1, padding=1),
                            nn.GELU(),
                            nn.Conv2d((2 * embed_dim) // 32, self.out_channels , kernel_size=1, stride=1, padding=0)
                        )
                else:
                    self.gs_dense_reg_head = nn.Sequential(
                            nn.Conv2d(self.dense_feats_dim, (2 * embed_dim) // 16, kernel_size=3, stride=1, padding=1),
                            nn.GELU(),
                            nn.Conv2d((2 * embed_dim) // 16, (2 * embed_dim) // 32, kernel_size=3, stride=1, padding=1),
                            nn.GELU(),
                            nn.Conv2d((2 * embed_dim) // 32, self.out_channels , kernel_size=1, stride=1, padding=0)
                        )
            else:
                raise ValueError(f"Unsupported gs_dense_reg_head_type: {self.gs_dense_reg_head_type}")

            # motion head
            self.motion_feature_head = DPTHead(dim_in=2 * embed_dim, feature_only=True, intermediate_layer_idx=self.intermediate_layer_idx, patch_size=self.patch_size)
            if self.motion_dense_key_head_type == 'mlp':
                self.motion_dense_key_head = Mlp(self.dense_feats_dim, 2 * self.dense_feats_dim, projected_motion_dim)
            elif self.motion_dense_key_head_type == 'conv':
                self.motion_dense_key_head = nn.Sequential(
                        nn.Conv2d(self.dense_feats_dim, (2 * embed_dim) // 16, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d((2 * embed_dim) // 16, (2 * embed_dim) // 32, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d((2 * embed_dim) // 32, projected_motion_dim , kernel_size=1, stride=1, padding=0)
                    )
            else:
                raise ValueError(f"Unsupported motion_dense_key_head_type: {self.motion_dense_key_head_type}")

            # feat head
            if self.with_feat:
                self.feat_feature_head = DPTHead(dim_in=2 * embed_dim, feature_only=True, intermediate_layer_idx=self.intermediate_layer_idx, patch_size=self.patch_size)

                if self.feat_dense_reg_head_type == 'mlp':
                    self.feat_dense_reg_head = Mlp(self.dense_feats_dim, 2 * self.dense_feats_dim, self.pred_feat_dim)
                elif self.feat_dense_reg_head_type == 'conv':
                    self.feat_dense_reg_head = nn.Sequential(
                            nn.Conv2d(self.dense_feats_dim, (2 * embed_dim) // 16, kernel_size=3, stride=1, padding=1),
                            nn.GELU(),
                            nn.Conv2d((2 * embed_dim) // 16, (2 * embed_dim) // 32, kernel_size=3, stride=1, padding=1),
                            nn.GELU(),
                            nn.Conv2d((2 * embed_dim) // 32, self.pred_feat_dim , kernel_size=1, stride=1, padding=0)
                        )
                else:
                    raise ValueError(f"Unsupported feat_dense_reg_head_type: {self.feat_dense_reg_head_type}")

        # feature decoder head
        if self.with_feat:
            # feat_decoders = []
            # for feat_type in self.target_feat_types:
            #     if feat_type == 'pe3r':
            #         feat_decoder = nn.Sequential(  # default 64 -> 1024
            #                             nn.Linear(self.pred_feat_dim, 128),
            #                             nn.ReLU(),
            #                             nn.Linear(128, 256),
            #                             nn.ReLU(),
            #                             nn.Linear(256, 512),
            #                             nn.ReLU(),
            #                             nn.Linear(512, 1024)
            #                         )
            #     elif feat_type == 'lseg':
            #         feat_decoder = nn.Sequential(  # default 64 -> 512
            #                             nn.Linear(self.pred_feat_dim, 128),
            #                             nn.ReLU(),
            #                             nn.Linear(128, 256),
            #                             nn.ReLU(),
            #                             nn.Linear(256, 512)
            #                         )
            #     else:
            #         raise NotImplementedError
            #     feat_decoders.append(feat_decoder)
            # self.feat_decoders = nn.ModuleList(feat_decoders)
            self.feat_decoder = nn.Sequential(  # default 64 -> 512
                nn.Linear(self.pred_feat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512)
            )
            self.feat_decoders = nn.ModuleList([self.feat_decoder])  # Adapt to existing model checkpoints: feat_decoders.0.xx
            '''
            Will still report: Missing key(s) in state_dict: "feat_decoder.xxx"
            But won't report: Unexpected key(s) in state_dict: "feat_decoders.0.xxx"
            '''

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
            self.quat_act_fn = lambda x: x  # NOTE: gsplat normalizes internally, so F.normalize(x, dim=-1) is not needed

        self.opacity_act_fn = lambda x: torch.sigmoid(x + opacity_offset)
        self.depth_act_fn = lambda x: near + torch.sigmoid(x) * (far - near)
        # self.rgb_act_fn = lambda x: torch.sigmoid(x) * 2 - 1 if sigmoid_rgb else x  # NOTE: if normalize rgb need * 2 - 1
        self.rgb_act_fn = lambda x: torch.sigmoid(x) if sigmoid_rgb else x

        if self.pred_gs_conf:
            self.gs_conf_act_fn = lambda x: torch.sigmoid(x)

        if self.enable_lifespan:
            # self.lifespan_act_fn = lambda x: F.sigmoid(x - 2.0) * 10000  # much more relaxed for static scenes
            # self.lifespan_act_fn = lambda x: F.sigmoid(x - 2.0) * (100 - 0.1) + 0.1  # default to 50s lifespan
            self.lifespan_act_fn = lambda x: F.sigmoid(x - 4.0) * (100 - 0.1) + 0.1  # default to 1.2s lifespan
            # self.lifespan_act_fn = lambda x: F.sigmoid(x - 4.5) * (100 - 0.1) + 0.1  # default to 0.5s lifespan
            # self.lifespan_act_fn = lambda x: F.sigmoid(x - 5.0) * (100 - 0.1) + 0.1  # default to 0.2s lifespan

        self.init_weights()

        if os.path.exists(self.vggt_pretrained_weight_filepath):
            assert self.aggregator.depth == 24 and self.aggregator.embed_dim == 1024 and self.patch_size == 14
            self.load_pretrained_vggt(self.vggt_pretrained_weight_filepath)  # TODO: additional learning rate

            def zero_module(module):
                """
                Zero out the parameters of a module and return it.
                """
                for p in module.parameters():
                    p.detach().zero_()
                return module

            # zero initialization
            if self.add_patch_plucker_embed:
                self.aggregator.patch_plucker_embed_mlp = zero_module(self.aggregator.patch_plucker_embed_mlp)
            if self.add_camera_embed:
                self.aggregator.pose_encoding_mlp = zero_module(self.aggregator.pose_encoding_mlp)

    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def load_pretrained_vggt(self, vggt_ckpts_filepath=''):
        vggt_pretrained_weight = torch.load(vggt_ckpts_filepath)

        patch_embed_state_dict = {}
        for old_key, value in vggt_pretrained_weight.items():
            if 'patch_embed' in old_key:
                patch_embed_state_dict[old_key.removeprefix('aggregator.patch_embed.')] = value
        self.aggregator.patch_embed.load_state_dict(patch_embed_state_dict, strict=True)

        global_blocks_state_dict = {}
        for old_key, value in vggt_pretrained_weight.items():
            if 'global_blocks' in old_key:
                global_blocks_state_dict[old_key.removeprefix('aggregator.global_blocks.')] = value
        self.aggregator.global_blocks.load_state_dict(global_blocks_state_dict, strict=True)

        frame_blocks_state_dict = {}
        for old_key, value in vggt_pretrained_weight.items():
            if 'frame_blocks' in old_key:
                frame_blocks_state_dict[old_key.removeprefix('aggregator.frame_blocks.')] = value
        self.aggregator.frame_blocks.load_state_dict(frame_blocks_state_dict, strict=True)

        if self.depth_head is not None:
            depth_head_state_dict = {}
            for old_key, value in vggt_pretrained_weight.items():
                if 'depth_head' in old_key:
                    depth_head_state_dict[old_key.removeprefix('depth_head.')] = value
            self.depth_head.load_state_dict(depth_head_state_dict, strict=True)

        if self.camera_head is not None:
            camera_head_state_dict = {}
            for old_key, value in vggt_pretrained_weight.items():
                if 'camera_head' in old_key:
                    camera_head_state_dict[old_key.removeprefix('camera_head.')] = value
            self.camera_head.load_state_dict(camera_head_state_dict, strict=True)

        if self.point_head is not None:
            point_head_state_dict = {}
            for old_key, value in vggt_pretrained_weight.items():
                if 'point_head' in old_key:
                    point_head_state_dict[old_key.removeprefix('point_head.')] = value
            self.point_head.load_state_dict(point_head_state_dict, strict=True)

        if self.aggregator.camera_token is not None:
            camera_token_state_dict = {}
            for old_key, value in vggt_pretrained_weight.items():
                if 'camera_token' in old_key:
                    camera_token_state_dict[old_key.removeprefix('aggregator.')] = value
            self.aggregator.camera_token.data = camera_token_state_dict['camera_token']

        if self.aggregator.register_token is not None:
            register_token_state_dict = {}
            for old_key, value in vggt_pretrained_weight.items():
                if 'register_token' in old_key:
                    register_token_state_dict[old_key.removeprefix('aggregator.')] = value
            self.aggregator.register_token.data = register_token_state_dict['register_token']

    def __build_gshead_and_motionhead_without_dpthead__(
        self,
        embed_dim,
        decoder_upsample_ratio,
        projected_motion_dim,
        decoder_type,
        grad_checkpointing,
        ):
        # ------- gs predictor and mask decoder -------
        gs_pred_out_chans = 32 if self.shortcut_rgb else self.out_channels
        if decoder_type == "dummy":
            self.gs_pred = nn.Linear(2 * embed_dim, decoder_upsample_ratio**2 * gs_pred_out_chans)
            self.unpatch_size = decoder_upsample_ratio

            if self.decoder_upsample_ratio == 8:
                # used for upscaling the low-resolution image features to the pixel-resolution
                # very handcrafted and never tuned
                self.output_upscaling = nn.Sequential(
                    nn.ConvTranspose2d(2 * embed_dim, 512, kernel_size=2, stride=2),
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
                    nn.ConvTranspose2d(2 * embed_dim, 512, kernel_size=2, stride=2),
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
            elif self.decoder_upsample_ratio == 14:
                # used for upscaling the low-resolution image features to the pixel-resolution
                # very handcrafted and never tuned
                self.output_upscaling = nn.Sequential(
                    nn.ConvTranspose2d(2 * embed_dim, 512, kernel_size=2, stride=2),
                    LayerNorm2d(512),
                    nn.GELU(),
                    nn.ConvTranspose2d(512, 256, kernel_size=1, stride=1),
                    LayerNorm2d(256),
                    nn.GELU(),
                    nn.ConvTranspose2d(256, 128, kernel_size=7, stride=7),
                    LayerNorm2d(128),
                    nn.GELU(),
                    nn.ConvTranspose2d(128, 128, kernel_size=1, stride=1),
                    LayerNorm2d(128),
                    nn.GELU(),
                )

        elif decoder_type == "conv":
            self.gs_pred = nn.Linear(2 * embed_dim, self.out_channels)
            # latent-XXX decoder
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
                nn.Conv2d(2 * embed_dim, 512, kernel_size=1),
                LayerNorm2d(512),
                nn.GELU(),
                nn.Conv2d(512, 256, kernel_size=1),
                LayerNorm2d(256),
                nn.GELU(),
                nn.Conv2d(256, 128, kernel_size=1),
                LayerNorm2d(128),
                nn.GELU(),
            )

        if self.shortcut_rgb:
            self.gs_pred_with_rgb = nn.Linear(gs_pred_out_chans + 3, self.out_channels)

        self.motion_key_head = Mlp(128, 256, projected_motion_dim)

    def __build_feat_head_without_dpthead__(
        self,
        embed_dim,
        decoder_upsample_ratio,
        decoder_type,
        ):
        # ------- feat predictor -------
        if decoder_type == "dummy":
            self.feat_pred = nn.Linear(2 * embed_dim, decoder_upsample_ratio**2 * self.pred_feat_dim)
            self.unpatch_size = decoder_upsample_ratio
        else:
            raise NotImplementedError

    def unpatchify(self, x, hw=None, channel_first=True, patch_size=None) -> torch.Tensor:
        hw = hw or self.img_size
        imgs = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=self.patch_size if patch_size is None else patch_size,
            p2=self.patch_size if patch_size is None else patch_size,
            h=hw[0] // (self.patch_size if patch_size is None else patch_size),
            w=hw[1] // (self.patch_size if patch_size is None else patch_size),
        )
        if not channel_first:
            imgs = rearrange(imgs, "b c h w -> b h w c")
        return imgs

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

    def decode_flow(self, ms3):
        # Extract degree of marginal scale (number of scale components)
        ms3_deg = ms3.shape[-1] // 4
        # Extract speed components (every 4th element starting from index 3)
        speed = ms3[..., 3::4, None]  # [B, T, H, W, ms3_deg, 1]
        # Reshape spatial components (first 3 of every 4 elements)
        ms3 = torch.cat(
            [ms3[..., None, i * 4:i * 4 + 3] for i in range(ms3_deg)], dim=-2
        )  # [B, T, H, W, ms3_deg, 3]

        # Rescale speed with sigmoid and apply clamping threshold
        speed = (speed + self.sigmoid_ms3_bias).sigmoid() * (
            self.sigmoid_ms3_max - self.sigmoid_ms3_min
        ) + self.sigmoid_ms3_min
        speed = (speed - self.ms3_clamp).clamp(0)  # Zero out speeds below threshold

        # Apply decay factor to speed based on scale level
        # Higher scale levels get progressively smaller speeds
        speed = torch.cat(
            [speed[..., i:i + 1, :] / self.ms3_deg_downmax_mult**i
                for i in range(ms3_deg)], dim=-2
        )  # [B, T, H, W, ms3_deg, 1]

        # Apply speed-modulated normalized marginal scales
        ms3 = speed * F.normalize(ms3[..., :3], dim=-1)  # Normalize and modulate by speed
        ms3 = ms3.reshape(ms3.shape[:-2] + (-1,))  # Flatten to [B, T, H, W, ms3_deg*3]
        return ms3

    def forward_motion_predictor(self, x, motion_tokens=None, gs_params=None, dense_feat=False):
        b, t, v, h, w, _ = gs_params["means"].shape
        if dense_feat:
            if self.motion_dense_key_head_type == 'mlp':
                img_keys = self.motion_dense_key_head(rearrange(x, 'b (t v) c h w -> b t v h w c', t=t, v=v))
            elif self.motion_dense_key_head_type == 'conv':
                img_keys = self.motion_dense_key_head(rearrange(x, "b (t v) c h w -> (b t v) c h w", t=t, v=v))
                img_keys = rearrange(img_keys, "(b t v) c h w -> b t v h w c", t=t, v=v)
        else:
            img_embeds = self.unpatchify(
                rearrange(x, "b (t v) hw c -> (b t v) hw c", t=t, v=v),
                hw=(h // self.unpatch_size, w // self.unpatch_size),
                patch_size=1,
            )
            if self.grad_checkpointing:
                img_embeds = checkpoint(self.output_upscaling, img_embeds, use_reentrant=self.use_reentrant)
            else:
                img_embeds = self.output_upscaling(img_embeds)
            img_embeds = rearrange(img_embeds, "(b t v) c h w -> b t v h w c", t=t, v=v)
            img_keys = self.motion_key_head(img_embeds)

        if self.num_motion_tokens > 0:
            hyper_in_list = []
            for i in range(self.num_motion_tokens):
                hyper_in = self.motion_query_heads[i](motion_tokens[:, i])
                hyper_in_list.append(hyper_in)
            motion_token_queries = torch.stack(hyper_in_list, dim=1)
            dot_product_similarity = torch.einsum(
                "b k c, b t v h w c -> b t v h w k",
                motion_token_queries,
                img_keys,
            )
            motion_weights = torch.softmax(dot_product_similarity / self.tau, dim=-1)
            motion_bases = self.motion_basis_decoder(motion_tokens)
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

    def forward_gs_predictor(self, x, origins, directions, activated_depth=None, rgb=None, dense_feat=False):
        b, t, v, h, w, _ = origins.shape
        if dense_feat:
            # shortcut rgb
            if self.shortcut_rgb and rgb is not None:
                x = torch.concat([rearrange(rgb, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v), x], dim=2)
            if self.gs_dense_reg_head_type == 'mlp':
                gs_params = self.gs_dense_reg_head(rearrange(x, "b (t v) c h w -> b t v h w c", t=t, v=v))
            elif self.gs_dense_reg_head_type == 'conv':
                gs_params = self.gs_dense_reg_head(rearrange(x, "b (t v) c h w -> (b t v) c h w", t=t, v=v))
                gs_params = rearrange(gs_params, "(b t v) c h w -> b t v h w c", t=t, v=v)
        else:
            x = rearrange(x, "b (t v) hw c -> (b t v) hw c", t=t, v=v)
            gs_params = self.gs_pred(x)
            gs_params = self.unpatchify(gs_params, hw=(h, w), patch_size=self.unpatch_size)
            # shortcut rgb
            if self.shortcut_rgb and rgb is not None:
                gs_params = torch.concat([gs_params, rgb], dim=1)
                gs_params = self.gs_pred_with_rgb(rearrange(gs_params, '(b t v) c h w -> (b t v) h w c', t=t, v=v))
                gs_params = rearrange(gs_params, "(b t v) h w c -> b t v h w c", t=t, v=v)
            else:
                gs_params = rearrange(gs_params, "(b t v) c h w -> b t v h w c", t=t, v=v)
        gs_params_dict = dict(zip(self.gs_params_name, gs_params.split(self.gs_params_size, dim=-1)))
        if activated_depth is not None:
            depths = activated_depth
        else:
            depths = self.depth_act_fn(gs_params_dict["depth"])
        means = origins + directions * depths

        # pesudo_3dgs
        if self.pesudo_3dgs:
            scale_limit = 4
            normals, delta_x, delta_y, dx, dy = compute_normals_scales_torch(rearrange(means, 'b t v h w c -> (b t v) h w c'))  # B, H, W, C
            quats = rot_from_normals_torch(normals.reshape(-1, 3), up=dy)
            quats = rearrange(quats, '(b t v h w) c -> b t v h w c', b=b, t=t, v=v, h=h, w=w)
            scale_limit = (scale_limit * depths * repeat(self.azimuth_tan, '... -> ... 1 1 3'))  # azimuth angle limit
            scales = scale_from_dxdy_torch(delta_x, delta_y)
            scales = rearrange(scales, '(b t v) h w c -> b t v h w c', b=b, t=t, v=v)
            scales = torch.where(scales > scale_limit, scale_limit, scales)
        else:
            scales = self.scale_act_fn(gs_params_dict["scales"])
            quats = self.quat_act_fn(gs_params_dict["quats"])
        colors = self.rgb_act_fn(gs_params_dict["colors"])
        opacitys = self.opacity_act_fn(gs_params_dict["opacitys"])
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
        if self.enable_lifespan:
            lifespans = self.lifespan_act_fn(gs_params_dict["lifespans"])
            output["lifespans"] = lifespans.squeeze(-1)
        return output

    def forward_feat_predictor(self, x, shape, dense_feat=False):
        h, w, t, v = shape
        if dense_feat:
            if self.feat_dense_reg_head_type == 'mlp':
                pred_feat = self.feat_dense_reg_head(rearrange(x, "b (t v) c h w -> b t v h w c", t=t))
            elif self.feat_dense_reg_head_type == 'conv':
                pred_feat = self.feat_dense_reg_head(rearrange(x, "b tv c h w -> (b tv) c h w"))
                pred_feat = rearrange(pred_feat, "(b t v) c h w -> b t v h w c", t=t, v=v)
        else:
            x = rearrange(x, "b tv hw c -> (b tv) hw c")
            pred_feat = self.feat_pred(x)  # [48, 600, 768] -> [48, 600, 768(12*8*8)]
            pred_feat = self.unpatchify(pred_feat, hw=(h, w), patch_size=self.unpatch_size)
            pred_feat = rearrange(pred_feat, "(b t v) c h w -> b t v h w c", t=t, v=v)
        pred_feat = torch.sigmoid(pred_feat)
        return pred_feat

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

    def forward_renderer_target_view_feat(self, render_results, data_dict, pred_feats, radius_clip=0.0, chunk_size=3):
        ''' render semantic probability '''
        b, t, v, _, h, w = data_dict['context_image'].shape
        tgt_h, tgt_w = data_dict["height"], data_dict["width"]
        tgt_t, tgt_v = data_dict["target_camtoworlds"].shape[1:3]

        means = rearrange(render_results["gs_means"], "b tgt_t (tvhw) c -> (b tgt_t) (tvhw) c")
        scales = rearrange(render_results["gs_scales"], "b tgt_t (tvhw) c -> (b tgt_t) (tvhw) c")
        quats = rearrange(render_results["gs_quats"], "b tgt_t (tvhw) c -> (b tgt_t) (tvhw) c")
        opacities = rearrange(render_results["gs_opacities"], "b tgt_t (tvhw) c -> (b tgt_t) (tvhw c)")
        # colors = rearrange(render_results["gs_color"], "b tgt_t (tvhw) c -> (b tgt_t) (tvhw) c")

        pred_feats = rearrange(pred_feats, "... c -> (...) c")
        probs = feat2class(pred_feats, get_text_label_feats(SEMANTIC_LABEL_LIST), similarity_probs_threshold=self.similarity_probs_threshold,
                              return_probs=True)
        probs = rearrange(probs, '(b t v h w) c -> b (t v h w) c', b=b, t=t, v=v, h=h, w=w)
        probs_batched = repeat(probs, "b ... -> (b t) ...", t=tgt_t)

        camtoworlds_batched = data_dict["target_camtoworlds"].view(b * tgt_t, -1, 4, 4)
        viewmats_batched = torch.linalg.inv(camtoworlds_batched.float())
        Ks_batched = data_dict["target_intrinsics"].view(b * tgt_t, -1, 3, 3)

        # NOTE: current npu rendering only supports 3D color, pad with zeros if not divisible by 3
        class_num = probs.shape[-1]
        pad_len = (chunk_size - class_num % chunk_size) % chunk_size

        if pad_len > 0:
            probs_batched = F.pad(probs_batched, (0, pad_len), mode='constant', value=0)

        probs_list = []
        for slice_i in range(0, probs.shape[-1], chunk_size):
            with torch.autocast("cuda", enabled=False):
                rendered_res, _, _ = self.rasterization_func(
                    means=means.float(),
                    quats=quats.float(),
                    scales=scales.float(),
                    opacities=opacities.float(),
                    colors=probs_batched[..., slice_i:slice_i+chunk_size].float(),
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
            rendered_probs, _ = rendered_res.split([self.gs_dim, 1], dim=-1)
            probs_list.append(rendered_probs)
        rendered_probs = torch.concat(probs_list, dim=-1)
        rendered_probs = rendered_probs[..., :class_num]

        # argmax
        rendered_semantic = torch.argmax(rendered_probs, dim=-1)
        rendered_semantic = rendered_semantic.long()
        return rendered_semantic.view(b, tgt_t, tgt_v, tgt_h, tgt_w)

    def forward_renderer(self, gs_params, data_dict, feats=None, render_motion_seg=not is_ascend_npu(),
                         radius_clip=0.0, time_step=5, concat_feat_render=True, idx=None, static_render=False):
        if os.getenv("CONTEXT_FEAT"):
            feats = None
        b, t, v, h, w, _ = gs_params["means"].shape
        tgt_h, tgt_w = data_dict["height"], data_dict["width"]
        tgt_t, tgt_v = data_dict["target_camtoworlds"].shape[1:3]
        means = rearrange(gs_params["means"], "b t v h w c -> b (t v h w) c")
        scales = rearrange(gs_params["scales"], "b t v h w c -> b (t v h w) c")
        quats = rearrange(gs_params["quats"], "b t v h w c -> b (t v h w) c")
        opacities = rearrange(gs_params["opacities"], "b t v h w -> b (t v h w)")
        colors = rearrange(gs_params["colors"], "b t v h w c -> b (t v h w) c")
        feats = rearrange(feats, "b t v h w c -> b (t v h w) c") if feats is not None else None

        means_batched = repeat(means, "b ... -> (b t) ...", t=tgt_t)
        scales_batched = repeat(scales, "b ... -> (b t) ...", t=tgt_t)
        quats_batched = repeat(quats, "b ... -> (b t) ...", t=tgt_t)
        opacities_batched = repeat(opacities, "b ... -> (b t) ...", t=tgt_t)
        color_batched = repeat(colors, "b ... -> (b t) ...", t=tgt_t)
        feats_batched = repeat(feats, "b ... -> (b t) ...", t=tgt_t) if feats is not None else None

        ctx_time = data_dict["context_time"] * data_dict["timespan"]  # [1, 4, 3]
        tgt_time = data_dict["target_time"] * data_dict["timespan"]   # [1, 20, 3]
        if tgt_time.ndim == 3:
            tdiff_forward = tgt_time.unsqueeze(2) - ctx_time.unsqueeze(1)  # [1, 20, 4, 3]
            tdiff_forward = tdiff_forward.view(b * tgt_t, t * v, 1)
            tdiff_forward_batched = repeat(tdiff_forward, "bt tv 1 -> bt (tv hw) 1", hw=h * w)  # [20, 460800, 1]
        else:
            tdiff_forward = tgt_time.unsqueeze(-1) - ctx_time.unsqueeze(-2)
            tdiff_forward = tdiff_forward.view(b * tgt_t, t, 1)
            tdiff_forward_batched = repeat(tdiff_forward, "bt t 1 -> bt (t vhw) 1", hw=v * h * w)

        if not self.use_ms3_motion:
            forward_v = rearrange(gs_params["forward_flow"], "b t v h w c -> b (t v h w) c")
            if self.add_angular_velocity:
                forward_v, forward_angular_v = forward_v.split([3, 3], dim=-1)
            forward_v_batched = repeat(forward_v, "b ... -> (b t) ...", t=tgt_t)
            if self.add_angular_velocity:
                forward_angular_v_batched = repeat(forward_angular_v, "b ... -> (b t) ...", t=tgt_t)

            forward_translation = forward_v_batched * tdiff_forward_batched

            # means = context frame + offset cur
            means_batched = means_batched + forward_translation

            if self.add_angular_velocity:
                # rotation_offset = (wx, wy, wz) * dt
                quats_offset_batched = angular_velocity_to_quaternion(forward_angular_v_batched, tdiff_forward_batched)
                # new rotation = rotation + rotation_offset
                quats_batched = quaternion_multiply(quats_batched, quats_offset_batched)
        else:
            forward_ms3 = gs_params["forward_ms3"][..., :self.ms3_deg * 3]
            forward_ms3 = rearrange(forward_ms3, "b t v h w c -> b (t v h w) c")
            forward_ms3_batched = repeat(forward_ms3, "b ... -> (b t) ...", t=tgt_t)
            if self.add_angular_velocity:
                forward_omega = gs_params["forward_ms3"][..., -self.omega_deg * 3:]
                forward_omega = rearrange(forward_omega, "b t v h w c -> b (t v h w) c")
                forward_omega_batched = repeat(forward_omega, "b ... -> (b t) ...", t=tgt_t)
                # angular velocity
                angle_axis_offset_batched = torch.stack(
                    [forward_omega_batched[..., i * 3:(i + 1) * 3] * tdiff_forward_batched ** (i + 1) / self.omega_factorials[i] \
                    for i in range(self.omega_deg)]
                ).sum(0)
                quats_offset_batched = angle_axis_to_quaternion(angle_axis_offset_batched)
                quats_batched = quaternion_multiply(quats_batched, quats_offset_batched)

            # offset cur
            forward_translation_cur = torch.stack(
                [forward_ms3_batched[..., i * 3:(i + 1) * 3] * tdiff_forward_batched ** (i + 1) / self.ms3_factorials[i] \
                for i in range(self.ms3_deg)]
            ).sum(0)

            delta_time = float(1 / data_dict['fps'])

            # offset next
            # forward_translation_next = torch.stack(
            #     [forward_ms3_batched[..., i * 3:(i + 1) * 3] * (tdiff_forward_batched + delta_time) ** (i + 1) / self.ms3_factorials[i] \
            #     for i in range(self.ms3_deg)]
            # ).sum(0)

            # velocity: cur (t) -> next (t+1)
            # forward_v_batched = (forward_translation_next - forward_translation_cur) / delta_time

            # offset prev
            forward_translation_prev = torch.stack(
                [forward_ms3_batched[..., i * 3:(i + 1) * 3] * (tdiff_forward_batched - delta_time) ** (i + 1) / self.ms3_factorials[i] \
                for i in range(self.ms3_deg)]
            ).sum(0)

            # velocity:  prev (t-1) -> cur (t)  ref: https://arxiv.org/pdf/2103.01306v5
            forward_v_batched = (forward_translation_cur - forward_translation_prev) / delta_time

            if len(ctx_time[0]) > 1:
                delta_ctx_time = ctx_time[0][1][0] - ctx_time[0][0][0]
            else:
                delta_ctx_time = time_step * delta_time # for stream inference with kv cache

            if "window" in self.mode or "causal" in self.mode:
                # e.g for frame 0, only used in frame 0 ~ frame 5; for frame 5, only used in frame 0 ~ frame 10;
                time_mask_backward = (tdiff_forward_batched < 0.5*delta_time) & (tdiff_forward_batched > -1.0*delta_ctx_time + 0.5*delta_time)
                # time_mask_backward = (tdiff_forward_batched < 0.5*delta_time) & (tdiff_forward_batched > -1.0*delta_ctx_time - 0.5*delta_time)
                time_mask_forward = (tdiff_forward_batched < delta_ctx_time + 0.5*delta_time) & (tdiff_forward_batched > 0.5*delta_time)
                static_mask = forward_v_batched.norm(dim=-1) < 1.0  # TODO
                final_mask = (static_mask.unsqueeze(-1) & time_mask_forward) | time_mask_backward

            # means = context frame + offset cur
            if not static_render:
                # stream mode only move backward 5 frames
                if "window" in self.mode or "causal" in self.mode:
                    forward_translation_cur = forward_translation_cur * time_mask_backward
                # means = context frame + offset cur
                means_batched = means_batched + forward_translation_cur

            if "window" in self.mode or "causal" in self.mode:
                # render flow only from future frame_(i+5)
                forward_v_batched = forward_v_batched * time_mask_backward
                opacities_batched = opacities_batched * final_mask.squeeze(-1)

            gs_params["forward_flow"] = gs_params["forward_ms3"]

        if not self.training:  # mask out some noisy flow
            forward_v_batched[forward_v_batched.norm(dim=-1) < 1.0] = 0.0

        if self.enable_lifespan:
            # opacity with lifespan
            lifespans = rearrange(gs_params["lifespans"], "b t v h w -> b (t v h w)")
            lifespans_batched = repeat(lifespans, "b ... -> (b t) ...", t=tgt_t)

            reduction_factor = 0.05
            sigma = (lifespans_batched ** 2) / (torch.log(torch.tensor(reduction_factor)) / -0.5)
            life_span_coef = torch.exp(-0.5 * (tdiff_forward_batched.squeeze(-1)) ** 2 / sigma)

            opacities_batched = opacities_batched * life_span_coef

            '''
            import numpy as np
            import matplotlib.pyplot as plt
            x = np.linspace(-1.5, 1.5, 100)

            def f(x):
                reduction_factor = 0.05
                pred_lifespan = F.sigmoid(torch.tensor( - 4.0)) * (100 - 0.1) + 0.1
                sigma = (pred_lifespan ** 2) / (torch.log(torch.tensor(reduction_factor)) / -0.5)
                return (torch.exp(-0.5 * (torch.tensor(x) ** 2) / (sigma ** 2)))

            y = f(x)
            # create figure
            plt.figure(figsize=(8, 6))
            plt.plot(x, y)

            plt.grid(True)
            plt.legend()
            plt.savefig('test.png')
            '''

        # Visualize the effect of each context frame
        if idx is not None:
            means_batched = means_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            scales_batched = scales_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            quats_batched = quats_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            opacities_batched = opacities_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            color_batched = color_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            forward_v_batched = forward_v_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            if self.enable_lifespan:
                life_span_coef = life_span_coef[:, (idx)*v*h*w: (idx+1)*v*h*w]

        gs_attrs = {
            'means': means_batched,
            'scales': scales_batched,
            'quats': quats_batched,
            'opacities': opacities_batched.unsqueeze(-1),
            'color': color_batched,
            'forward_v': forward_v_batched,
        }
        if feats is not None:
            gs_attrs['feats'] = feats_batched

        if self.voxelize and self.pred_gs_conf:
            gs_confs = rearrange(gs_params["confs"], "b t v h w c -> b (t v h w) c")
            gs_confs_batched = repeat(gs_confs, "b ... -> (b t) ...", t=tgt_t)
            if idx is not None:
                gs_confs_batched = gs_confs_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]

            gs_attrs_b_t_lists = {attr: [] for attr in gs_attrs.keys()}
            for b_idx in range(b):
                for t_idx in range(tgt_t):
                    b_t_idx = b_idx * tgt_t + t_idx

                    # Voxelize using a specific voxelsize, and calculate the weight by confidence
                    weights, inverse_indices = self.voxelizaton_using_confidence(gs_attrs['means'][b_t_idx],
                                                                                 gs_confs_batched[b_t_idx].squeeze(1),
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
            print('Original gaussian count       ', means_batched.shape[1])
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
            gs_attrs = {
                'means': means_batched,
                'scales': scales_batched,
                'quats': quats_batched,
                'opacities': opacities_batched.unsqueeze(-1),
                'color': color_batched,
                'forward_v': forward_v_batched,
            }
            if feats is not None:
                feats_batched = gs_attrs_voxel_padded['feats']
                gs_attrs['feats'] = feats_batched

        if self.training:
            colors_batched = color_batched  # do not render flow during training
            forward_flow = None
        else:
            colors_batched = torch.cat([color_batched, forward_v_batched], dim=-1)  # render flow during non-training
        if feats_batched is not None and concat_feat_render:
            if idx is not None:
                feats_batched = feats_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]
            colors_batched = torch.cat([colors_batched, feats_batched], dim=-1)

        if not self.training and self.num_motion_tokens > 0 and render_motion_seg:
            # render the motion segmentation map
            motion_weights = rearrange(gs_params["motion_weights"], "b t v h w k -> b (t v h w) k")
            weights_batched = repeat(motion_weights, "b ... -> (b t) ...", t=tgt_t)
            if idx is not None:
                weights_batched = weights_batched[:, (idx)*v*h*w: (idx+1)*v*h*w]

            colors_batched = torch.cat([colors_batched, weights_batched], dim=-1)

        camtoworlds_batched = data_dict["target_camtoworlds"].view(b * tgt_t, -1, 4, 4)
        viewmats_batched = torch.linalg.inv(camtoworlds_batched.float())
        Ks_batched = data_dict["target_intrinsics"].view(b * tgt_t, -1, 3, 3)

        # NOTE: not sure if this is a gsplat bug
        if self.use_2dgs:
            colors_batched = colors_batched[:, None]

        if is_ascend_npu():
            # NPU does not support feature rendering for now
            assert feats is None
            assert render_motion_seg == False

            if self.training:
                # current strategy: only render rgb during training
                assert colors_batched.shape[-1] == 3

        if os.environ.get('TIME_COUNT_TYPE2'):
            start = time.time()

        motion_seg = None
        feat = None
        if not self.training and self.num_motion_tokens > 0 and render_motion_seg:
            colors_to_render = colors_batched[..., :-self.num_motion_tokens].float()
        else:
            colors_to_render = colors_batched.float()

        with torch.autocast("cuda", enabled=False):
            rendered_color, rendered_alpha, *_ = self.rasterization_func(
                means=means_batched.float(),
                quats=quats_batched.float(),
                scales=scales_batched.float(),
                opacities=opacities_batched.float(),
                colors=colors_to_render,
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
        if feats is not None and concat_feat_render:
            if not self.training:
                color, forward_flow, feat, depth = rendered_color.split(
                    [self.gs_dim, 3, self.pred_feat_dim, 1], dim=-1
                )
            else:
                color, feat, depth = rendered_color.split(
                    [self.gs_dim, self.pred_feat_dim, 1], dim=-1
                )
        else:
            if not self.training:
                color, forward_flow, depth = rendered_color.split([self.gs_dim, 3, 1], dim=-1)
            else:
                color, depth = rendered_color.split([self.gs_dim, 1], dim=-1)

        if not self.training and self.num_motion_tokens > 0 and render_motion_seg:
            with torch.autocast("cuda", enabled=False):
                chunksize = 32
                assignment_map = []
                rendered_colors = colors_batched[..., -self.num_motion_tokens :]
                for i in range(0, self.num_motion_tokens, chunksize):
                    weights, *_ = self.rasterization_func(
                        means=means_batched.float(),
                        quats=quats_batched.float(),
                        scales=scales_batched.float(),
                        opacities=opacities_batched.float(),
                        colors=rendered_colors[..., i:i + chunksize],
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

        if os.environ.get('TIME_COUNT_TYPE2'):
            torch.cuda.synchronize()
            print('Rendering time - forward: ', time.time() - start)

        output_dict = {
            "rendered_image": color.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "rendered_depth": depth.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
        }
        if forward_flow is not None:
            # forward_flow = forward_flow / (rendered_alpha + 1e-8)  # ED mode: It will be affected by the opacity of the sky.
            output_dict["rendered_flow"] = forward_flow.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1)
        if rendered_alpha is not None:
            output_dict["rendered_alpha"] = rendered_alpha.view(b, tgt_t, tgt_v, tgt_h, tgt_w)
        else:
            output_dict["rendered_alpha"] = None

        if feats is not None and not concat_feat_render:
            # render features separately
            with torch.autocast("cuda", enabled=False):
                rendered_feat, _, _ = self.rasterization_func(
                    means=means_batched.detach().float(),
                    quats=quats_batched.detach().float(),
                    scales=scales_batched.detach().float(),
                    opacities=opacities_batched.detach().float(),
                    colors=feats_batched.float(),
                    viewmats=viewmats_batched.detach(),
                    Ks=Ks_batched.detach(),
                    width=tgt_w,
                    height=tgt_h,
                    render_mode="RGB+ED",  # render color with expected depth
                    near_plane=self.near,
                    far_plane=self.far,
                    packed=False,
                    radius_clip=radius_clip,
                )
                feat, _ = rendered_feat.split([self.pred_feat_dim, 1], dim=-1)

        if feat is not None:
            output_dict["rendered_feat"] = feat.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1)
        if motion_seg is not None:
            output_dict["rendered_motion_seg"] = motion_seg.squeeze(-1)
        if self.save_gaussian or (self.with_feat and os.getenv("CONTEXT_FEAT")):
            for k, v in gs_attrs.items():
                output_dict[f"gs_{k}"] = v.unsqueeze(0)
        return output_dict

    def voxelizaton_using_confidence(self, gs_xyz, gs_conf, voxel_size):
        voxel_indices = (gs_xyz / voxel_size).round().int()  # [N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(voxel_indices, dim=0, return_inverse=True, return_counts=True)

        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(gs_conf, inverse_indices, dim=0)
        conf_exp = torch.exp(gs_conf - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0)  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1)  # [N, 1]

        return weights, inverse_indices

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

    def forward(self,
                data_dict,
                stream_save=True,
                aggregator_kv_cache_list: List[List[torch.Tensor]] = None,
                camera_head_kv_cache_list: List[List[List[torch.Tensor]]] = None):
        # if not self.training:
        #     print(f"Begin Inference And Use {self.mode} Mode")

        if os.environ.get('TIME_COUNT_TYPE2'):
            start = time.time()
        images = data_dict["context_image"]
        b, t, v, c, h, w = images.size()

        # GT camera pose
        _, ray_dict = self.get_ray_dict(data_dict)

        # Normalize images and reshape for patch embed
        # NOTE: apply this normalization when rgb is in 0~1 range
        images = (images - self.aggregator._resnet_mean) / self.aggregator._resnet_std # (b, t, v, c, h, w)

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(b * t * v, c, h, w)

        if aggregator_kv_cache_list is not None:
            output_list, self.patch_start_idx, aggregator_kv_cache_list = self.aggregator(data_dict, mode=self.mode, kv_cache_list=aggregator_kv_cache_list)
        else:
            output_list, self.patch_start_idx = self.aggregator(data_dict, mode=self.mode)

        with torch.cuda.amp.autocast(enabled=False):
            last_tokens = output_list[-1]
            others_last_tokens = last_tokens[:, :, :self.patch_start_idx]  # Exclude patch token

            sky_token, affine_tokens, motion_tokens, time_tokens = None, None, None, None
            if self.use_sky_token:
                sky_token = others_last_tokens[:, :, -1:]
                sky_token = self.sky_token_norm(sky_token)  # NOTE: token need LayerNorm
                sky_token = sky_token.mean(1)   # remove extra copied parts above
                others_last_tokens = others_last_tokens[:, :, :-1]

            if self.use_affine_token:
                affine_tokens = others_last_tokens[:, :, -self.num_cams:]
                affine_tokens = self.affine_token_norm(affine_tokens)  # NOTE: token need LayerNorm
                affine_tokens = affine_tokens.mean(1)  # remove extra copied parts above
                others_last_tokens = others_last_tokens[:, :, :-self.num_cams]

            if self.num_motion_tokens > 0:
                motion_tokens = others_last_tokens[:, :,  -self.num_motion_tokens:]
                motion_tokens = self.motion_token_norm(motion_tokens)  # NOTE: token need LayerNorm
                motion_tokens = motion_tokens.mean(1)  # remove extra copied parts above
                others_last_tokens = others_last_tokens[:, :, :-self.num_motion_tokens]

            if self.use_time_token:
                time_tokens = others_last_tokens[:, :, -1:]
                others_last_tokens = others_last_tokens[:, :, :-1]

            pose_enc_list = None
            if self.camera_head is not None:
                if camera_head_kv_cache_list is not None:
                    pose_enc_list, camera_head_kv_cache_list = self.camera_head(output_list, t, v, mode=self.mode, kv_cache_list=camera_head_kv_cache_list)
                else:
                    pose_enc_list = self.camera_head(output_list, t, v, mode=self.mode)
                pose_enc = pose_enc_list[-1]
                pred_extrinsic, pred_intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                # TODO: what to do if pred_intrinsic has nan?
                # world to camera -> camera to world
                pred_camtoworlds = torch.concat([pred_extrinsic, repeat(torch.tensor([[0, 0, 0, 1]]).to(images.device), '... -> b tv ...', b=b, tv=t*v)], dim=-2).inverse()

                pred_ray_dict = self.plucker_embedder(
                    rearrange(pred_intrinsic, 'b (t v) ... -> b t v ...', t=t, v=v),
                    rearrange(pred_camtoworlds, 'b (t v) ... -> b t v ...', t=t, v=v),
                    image_size=images.shape[-2:],
                )

            pred_context_depth, pred_context_depth_conf = None, None
            if self.depth_head is not None and self.use_pred_depth:
                pred_context_depth, pred_context_depth_conf = self.depth_head(
                    output_list, images=rearrange(images, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v), patch_start_idx=self.patch_start_idx
                )
                # unproject_depth_map_to_point_map(depth[0], extrinsic[0], intrinsic[0])
                # TODO: apply sigmoid activation
                # pred_context_depth = self.near + pred_context_depth * (self.far - self.near)
                pred_context_depth = torch.clamp(pred_context_depth, min=self.near, max=(self.far - self.near))
                pred_context_depth = rearrange(pred_context_depth, 'b (t v) ... -> b t v ...', b=b, t=t, v=v)
                pred_context_depth_conf = rearrange(pred_context_depth_conf, 'b (t v) ... -> b t v ...', b=b, t=t, v=v)

            pred_context_pts3d, pred_context_pts3d_conf = None, None
            if self.point_head is not None:
                pred_context_pts3d, pred_context_pts3d_conf = self.point_head(
                    output_list, images=rearrange(images, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v), patch_start_idx=self.patch_start_idx
                )
                pred_context_pts3d = rearrange(pred_context_pts3d, 'b (t v) ... -> b t v ...', b=b, t=t, v=v)
                pred_context_pts3d_conf = rearrange(pred_context_pts3d_conf, 'b (t v) ... -> b t v ...', b=b, t=t, v=v)

            # TODO: switch between camera pose + depth and pointmap

            # pass results from camera head and depth head
            if self.use_pred_camera_pose:
                ray_origins = pred_ray_dict["origins"]
                ray_dirs = pred_ray_dict["dirs"]
            else:
                ray_origins = ray_dict["origins"]
                ray_dirs = ray_dict["dirs"]
            if self.use_pred_depth:
                activated_depth = pred_context_depth
            else:
                activated_depth = None

            pred_feat = None
            if self.use_last_token:
                # last layer's token
                aggregated_last_tokens = last_tokens[:, :, self.patch_start_idx:]  # aggregated patch token
                aggregated_last_tokens = self.aggregated_last_tokens_norm(aggregated_last_tokens)  # NOTE: token need LayerNorm

                if self.pesudo_3dgs:
                    self.azimuth_tan = 1 / data_dict['context_intrinsics'][:, :, :, 0, 0]  # compute_azimuth_tan

                # Gaussian head
                gs_params = self.forward_gs_predictor(aggregated_last_tokens, ray_origins, ray_dirs, activated_depth=activated_depth, rgb=images)

                # Motion head
                gs_params = self.forward_motion_predictor(aggregated_last_tokens, motion_tokens, gs_params)

                # Feature head
                if self.with_feat:
                    pred_feat = self.forward_feat_predictor(aggregated_last_tokens, shape=(h, w, t, v))
            else:
                # Gaussian head (Dpt head)
                gs_dense_feats = self.gs_feature_head(output_list, images=rearrange(images, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v), patch_start_idx=self.patch_start_idx)
                gs_params = self.forward_gs_predictor(gs_dense_feats, ray_origins, ray_dirs, activated_depth=activated_depth, rgb=images, dense_feat=True)

                # Motion head (Dpt head)
                motion_dense_feats = self.motion_feature_head(output_list, images=rearrange(images, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v), patch_start_idx=self.patch_start_idx)
                gs_params = self.forward_motion_predictor(motion_dense_feats, motion_tokens, gs_params, dense_feat=True)

                # Feature head (Dpt head)
                if self.with_feat:
                    feat_dense_feats = self.feat_feature_head(output_list, images=rearrange(images, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v), patch_start_idx=self.patch_start_idx)
                    pred_feat = self.forward_feat_predictor(feat_dense_feats, shape=(h, w, t, v), dense_feat=True)

        if os.environ.get('TIME_COUNT_TYPE2'):
            torch.cuda.synchronize()
            print('Computation time - forward: ', time.time() - start)

        if not self.training and self.use_render_novel_view:
            self.apply_novelview_rt(data_dict, degree_x=0, degree_y=-10, degree_z=0, trans_x=-7, trans_y=-5, trans_z=2)

        output = self.post_processing(data_dict, gs_params, ray_dict=ray_dict,
                                      pred_feat=pred_feat,
                                      sky_token=sky_token,
                                      affine_tokens=affine_tokens,
                                      pose_enc_list=pose_enc_list,
                                      pred_context_depth=pred_context_depth,
                                      pred_context_depth_conf=pred_context_depth_conf,
                                      pred_context_pts3d=pred_context_pts3d,
                                      pred_context_pts3d_conf=pred_context_pts3d_conf,
                                      )

        # save gaussian
        if not self.training and self.save_gaussian and stream_save:
            self.save_gs_params_to_ply(data_dict, output["render_results"],
                                       target_sky=gs_params["target_sky"] if self.use_sky_token else None,
                                       affine=gs_params["affine"] if self.use_affine_token else None,
                                       save_path=self.gaussian_save_path)

        # save rendered pointcloud
        if not self.training and self.save_rendered_pc and stream_save:
            self.save_rendered_pointcloud(data_dict, output, save_path=self.rendered_pc_save_path)

        output["aggregator_kv_cache_list"] = None
        output["camera_head_kv_cache_list"] = None

        if aggregator_kv_cache_list is not None:
            output["aggregator_kv_cache_list"] = aggregator_kv_cache_list

        if camera_head_kv_cache_list is not None:
            output["camera_head_kv_cache_list"] = camera_head_kv_cache_list

        return output

    def post_processing(self, data_dict, gs_params, time_step=5, ray_dict=None,
                        pred_feat=None, sky_token=None, affine_tokens=None,
                        pose_enc_list=None, pred_context_depth=None, pred_context_depth_conf=None,
                        pred_context_pts3d=None, pred_context_pts3d_conf=None, static_render=False):
        images = data_dict["context_image"]
        b, t, v, c, h, w = images.size()

        step = 20

        if data_dict["target_camtoworlds"].shape[1] <= step:
            # Rendering the results of context frame aggregation.
            render_results = self.forward_renderer(gs_params, data_dict, feats=pred_feat, time_step=time_step, static_render=static_render)

            # Rendering every context frame (only itself, no aggregation) in target view.
            if not self.training and self.render_context_frame_contribution:
                for idx in range(t):
                    context_render_results = self.forward_renderer(gs_params, data_dict, idx=idx, time_step=time_step, static_render=static_render)
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
                chunk_render_results = self.forward_renderer(gs_params, chunk_data_dict, feats=pred_feat, time_step=time_step, static_render=static_render)
                if chunk_start == 0:
                    render_results = chunk_render_results
                else:
                    for key, value in chunk_render_results.items():
                        render_results[key] = torch.cat([render_results[key], value], dim=1)

                # Rendering every context frame (only itself, no aggregation) in target view.
                if not self.training and self.render_context_frame_contribution:
                    for idx in range(t):
                        chunk_context_render_results = self.forward_renderer(gs_params, chunk_data_dict, idx=idx, time_step=time_step, static_render=static_render)
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
                    target_dirs = target_ray_dict["dirs"][:, chunk_start:chunk_start + step]
                    chunk_target_sky = self.sky_head(target_dirs, sky_token)
                    images[:, chunk_start:chunk_start + step] += (
                        1 - opacities[:, chunk_start:chunk_start + step][..., None]
                    ) * chunk_target_sky
                    # per-context in target view
                    if not self.training and self.render_context_frame_contribution:
                        for idx in range(t):
                            render_results[f'context_{idx}_rendered_image'][:, chunk_start:chunk_start + step] += (1 - render_results[f'context_{idx}_rendered_alpha'][:, chunk_start:chunk_start + step][..., None]) * chunk_target_sky
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
                        context_dirs = context_ray_dict["dirs"][:, chunk_start:chunk_start + step]
                        chunk_context_sky = self.sky_head(context_dirs, sky_token)
                        context_images[:, chunk_start:chunk_start + step] += (
                            1 - context_opacities[:, chunk_start:chunk_start + step][..., None]
                        ) * chunk_context_sky
            if "target_sky" not in gs_params.keys() or gs_params["target_sky"] is None:
                gs_params["target_sky"] = 1 - opacities[..., None]
            if "sky_token" not in gs_params.keys() or gs_params["sky_token"] is None:
                gs_params["sky_token"] = sky_token

        if self.use_affine_token:
            affine = self.affine_linear(affine_tokens)  # b v (gs_dim * (gs_dim + 1))
            affine_matrix = rearrange(affine, "b v (p q) -> b v p q", p=self.gs_dim)
            linear_part = affine_matrix[..., :3]  # b, v, 3, 3
            translation_part = affine_matrix[..., 3]  # b, v, 3
            translation_part = translation_part.view(b, 1, v, 1, 1, 3)
            gs_params["images_without_affine"] = images.clone()  # TODO: whether to add regularization to keep it consistent with original image
            # apply linear and translation
            images = torch.einsum('btvhwi,bvij->btvhwj', images, linear_part) + translation_part
            if self.render_context_view:
                context_images = torch.einsum('btvhwi,bvij->btvhwj', context_images, linear_part) + translation_part
            if not self.training and self.render_context_frame_contribution:
                for idx in range(t):
                    render_results[f'context_{idx}_rendered_image'] = torch.einsum('btvhwi,bvij->btvhwj', \
                        render_results[f'context_{idx}_rendered_image'], linear_part) + translation_part
            if "affine" not in gs_params.keys() or gs_params["affine"] is None:
                gs_params["affine"] = {
                    'linear': linear_part,
                    'translation': translation_part,
                }

        render_results["rendered_image"] = images
        render_results = self.forward_decoder(render_results)

        if self.with_feat:
            if os.getenv("CONTEXT_FEAT"):
                render_results["rendered_feat"] = pred_feat
            # for i in range(len(self.feat_decoders)):
            #     render_results[f"rendered_feat_{i}"] = self.feat_decoders[i](render_results["rendered_feat"])
            # render_results['rendered_semantic'] = self.forward_renderer_target_view_feat(render_results, data_dict, self.feat_decoder(pred_feat))  # gpu context
            render_results["rendered_feat"] = self.feat_decoder(render_results["rendered_feat"])
            if os.getenv("CONTEXT_FEAT") and not self.training:
                render_results['rendered_semantic'] = self.forward_renderer_target_view_feat(render_results, data_dict, render_results["rendered_feat"])

            if self.save_gaussian:
                render_results["gs_decoded_feats"] = self.feat_decoder(pred_feat)
            if self.save_rendered_pc:
                render_results["gs_rendered_decoded_feat"] = render_results["rendered_feat"]

        output = dict(
            ray_dict=ray_dict,
            gs_params=gs_params,
            render_results=render_results,
            pred_feat=pred_feat,
            sky_token=sky_token,
            affine_tokens=affine_tokens
        )
        # context
        if self.render_context_view:
            output["rendered_context_image"] = context_images
            output["rendered_context_depth"] = context_depths
            output["rendered_context_alpha"] = context_opacities

        if self.pred_gs_conf:
            output['pred_gs_conf'] = rearrange(gs_params['confs'], 'b t v h w 1 -> b 1 (t v) h w')    # gs confidence
        if self.camera_head is not None:
            assert pose_enc_list is not None
            output['pred_context_camera_enc_list'] = pose_enc_list
        if self.depth_head is not None:
            assert pred_context_depth is not None and pred_context_depth_conf is not None
            output['pred_context_depth'] = pred_context_depth.squeeze(-1)
            output['pred_context_depth_conf'] = pred_context_depth_conf
        if self.point_head is not None:
            assert pred_context_pts3d is not None and pred_context_pts3d_conf is not None
            output['pred_context_pts3d'] = pred_context_pts3d
            output['pred_context_pts3d_conf'] = pred_context_pts3d_conf

        return output

    def apply_novelview_rt(self, data_dict,
                                 degree_x=0,
                                 degree_y=-10,
                                 degree_z=0,
                                 trans_x=-7,
                                 trans_y=-5,
                                 trans_z=2,
                                 fix_cam_pos=False):
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
        # fix_camera_t
        if fix_cam_pos:
            first_cam_pos = data_dict["target_camtoworlds"][:, 0:1, :, :3, 3]
            data_dict["target_camtoworlds"][..., :3, 3] = first_cam_pos

        # rotation
        data_dict["target_camtoworlds"][..., :3, :3] @= (R_z @ R_y @ R_x)

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

    def save_gs_params_to_ply(self, data_dict, render_results, target_sky, affine, \
                              opacity_threshold=0.1, save_path='output_gs'):
        input_image = data_dict['context_image']
        # first_idx = data_dict['context_frame_idx'][0].to(torch.int16).tolist()[0]
        target_frame_idxs = data_dict['target_frame_idx'][0].to(torch.int16).tolist()
        b, t, v, c, h, w = input_image.shape
        assert b == 1
        _, tgt_t, _, _ = render_results['gs_means'].shape  # [(b tgt_t), (t v h w), c]

        for t_idx in range(tgt_t):
            xyz = render_results['gs_means'][:, t_idx]
            color = render_results['gs_color'][:, t_idx]
            opacities = render_results['gs_opacities'][:, t_idx]
            scales = render_results['gs_scales'][:, t_idx]
            quats = render_results['gs_quats'][:, t_idx]

            # opacities mask
            mask = (opacities > opacity_threshold).squeeze(-1)
            # sky mask
            if target_sky is not None:
                context_frame_idx = (data_dict['context_frame_idx'][0]).to(torch.int16).tolist()
                context_frame_idx = [x - min(context_frame_idx) for x in context_frame_idx]
                context_sky = target_sky[:, context_frame_idx].squeeze(-1)
                context_sky_mask = rearrange(context_sky, 'b t v h w -> b (t v h w)') < opacity_threshold
                mask = mask & context_sky_mask

            # affine transform
            if affine is not None:
                color = rearrange(color, 'b (t v h w) c -> b t v h w c', t=t, v=v, h=h, w=w)
                color = torch.einsum('btvhwi,bvij->btvhwj', color, affine['linear']) + affine['translation']
                color = rearrange(color, 'b t v h w c -> b (t v h w) c')

            # filter gs
            xyz = xyz[mask]
            color = color[mask]
            opacities = opacities[mask]
            scales = scales[mask]
            quats = quats[mask]

            # align to the same coordinate system (first frame in first segment)
            if 'segment_to_ref' in data_dict:
                segment_to_ref = data_dict['segment_to_ref'][0].to(torch.float32).cpu().numpy()
                xyz = xyz.to(torch.float32).cpu().numpy()
                # xyz = xyz @ segment_to_ref[:3, :3].T + segment_to_ref[:3, 3]  # precision loss
                # xyz = ((segment_to_ref @ np.concatenate([xyz, np.ones_like(xyz)[:, :1]], axis=-1).T).T[:, :3]  # left multiply
                xyz = (np.concatenate([xyz, np.ones_like(xyz)[:, :1]], axis=-1) @ segment_to_ref.T)[:, :3]  # right multiply
                xyz = torch.from_numpy(xyz)

            # gs: use gs rgb
            gaussians_ply_format = torch.zeros(torch.Size([1, color.shape[0], 14]))
            gaussians_ply_format[:, :, 0:3] = xyz
            gaussians_ply_format[:, :, 3:6] = color
            gaussians_ply_format[:, :, 6:7] = opacities
            gaussians_ply_format[:, :, 7:10] = scales
            gaussians_ply_format[:, :, 10:14] = quats
            save_ply(gaussians_ply_format, os.path.join(save_path, f'gs_{target_frame_idxs[t_idx]}.ply'))

            # gs: use input rgb
            color = rearrange(input_image, 'b t v c h w -> b (t v h w) c')
            color = color[mask]
            gaussians_ply_format[:, :, 3:6] = color
            save_ply(gaussians_ply_format, os.path.join(save_path, f'gs_rgb_{target_frame_idxs[t_idx]}.ply'))

            if self.with_feat:
                feats = render_results['gs_decoded_feats']
                feats = rearrange(feats, "b t v h w c -> (b t v h w) c")
                semantic = feat2class(feats, get_text_label_feats(SEMANTIC_LABEL_LIST), similarity_probs_threshold=self.similarity_probs_threshold)
                semantic = semantic.view(1, -1)

                color = torch.zeros_like(render_results['gs_color'][:, t_idx])
                # color = render_results['gs_color'][:, t_idx]  # TODO: repeat does not deepcopy
                for class_idx in range(len(SEMANTIC_LABEL_LIST)):
                    color[semantic == class_idx] = torch.tensor(SEMANTIC_ID_TO_COLOR[class_idx]).to(color.device)
                color /= 255.

                # gs: use semantic rgb
                color = color[mask]
                gaussians_ply_format[:, :, 3:6] = color
                # concat semantic
                semantic = semantic[mask]
                semantic = semantic.to(gaussians_ply_format.device)
                semantic = semantic.unsqueeze(0).unsqueeze(-1)
                gaussians_ply_format = torch.concat([gaussians_ply_format, semantic], dim=2)
                # concat mask
                mask_indices = torch.nonzero(mask.view(-1)).squeeze()
                mask_indices = mask_indices.to(gaussians_ply_format.device)
                mask_indices = mask_indices.unsqueeze(0).unsqueeze(-1)
                gaussians_ply_format = torch.concat([gaussians_ply_format, mask_indices], dim=2)
                save_ply(gaussians_ply_format, os.path.join(save_path, f'gs_semantic_{target_frame_idxs[t_idx]}.ply'),
                         semantic_start_idx=14 if self.with_feat else None,
                         mask_indices_start_idx=15)
            print(f'Save frame_{target_frame_idxs[t_idx]} gs in {save_path}.')

    def save_rendered_pointcloud(self, data_dict, output, save_path, save_orig_results=False):
        rendered_image = output["render_results"]['rendered_image']
        rendered_depth = output["render_results"]['rendered_depth']
        rendered_flow = output["render_results"]['rendered_flow']
        b, tgt_t, v, h, w, c = rendered_image.shape
        assert b == 1
        # first_idx = data_dict['context_frame_idx'][0].to(torch.int16).tolist()[0]
        target_frame_idxs = data_dict['target_frame_idx'][0].to(torch.int16).tolist()
        c2ws = data_dict['target_camtoworlds'][0].view(-1, 4, 4)

        # save rendered rgb, depth, flow
        if save_orig_results:
            dir_path = os.path.join(save_path, 'orig_results')
            os.makedirs(dir_path, exist_ok=True)
            import cv2
            from src.visualization.visualization_tools import depth_visualizer, scene_flow_to_rgb
            for t_idx in range(tgt_t):
                rgb = output["render_results"]['rendered_image'][0, t_idx]
                rgb = rearrange(rgb, 'v h w c -> h (v w) c')
                cv2.imwrite(f'{dir_path}/frame_{target_frame_idxs[t_idx]}_rgb.png', rgb.to(torch.float16).detach().cpu().numpy()[:, :, [2, 1, 0]]*255)

                depth = output["render_results"]['rendered_depth'][0, t_idx]
                alpha = output["render_results"]['rendered_alpha'][0, t_idx]
                depth = depth.to(torch.float16).detach().cpu().numpy()
                alpha = alpha.to(torch.float16).detach().cpu().numpy()
                depth_image = depth_visualizer(depth, alpha)
                depth_image = rearrange(depth_image, 'v h w c -> h (v w) c')
                cv2.imwrite(f'{dir_path}/frame_{target_frame_idxs[t_idx]}_depth.png', depth_image[:, :, [2, 1, 0]]*255)

                flow = output["render_results"]['rendered_flow'][0, t_idx]
                flow = scene_flow_to_rgb(flow, flow_max_radius=15)
                flow = rearrange(flow, 'v h w c -> h (v w) c')
                cv2.imwrite(f'{dir_path}/frame_{target_frame_idxs[t_idx]}_flow.png', flow.to(torch.float16).detach().cpu().numpy()[:, :, [2, 1, 0]]*255)

                print(f'Save frame_{target_frame_idxs[t_idx]} rendered pc in {dir_path}.')

        target_ray_dict = self.plucker_embedder(
            data_dict["target_intrinsics"],
            data_dict["target_camtoworlds"],
            image_size=(data_dict["height"], data_dict["width"]),
        )
        xyzs = target_ray_dict['origins'] + target_ray_dict['dirs'] * rendered_depth.unsqueeze(-1)

        for t_idx in range(tgt_t):
            xyz = xyzs[0, t_idx].reshape(-1, 3)
            color = rendered_image[0, t_idx].reshape(-1, 3)
            flow = rendered_flow[0, t_idx].reshape(-1, 3)
            flow = flow * (1 / data_dict['fps'])  # to next frame

            # align to the same coordinate system (first frame in first segment)
            if 'segment_to_ref' in data_dict:
                segment_to_ref = data_dict['segment_to_ref'][0].to(torch.float32).cpu().numpy()
                xyz = xyz.to(torch.float32).cpu().numpy()
                # xyz = xyz @ segment_to_ref[:3, :3].T + segment_to_ref[:3, 3]  # precision loss
                # xyz = ((segment_to_ref @ np.concatenate([xyz, np.ones_like(xyz)[:, :1]], axis=-1).T).T[:, :3]  # left multiply
                xyz = (np.concatenate([xyz, np.ones_like(xyz)[:, :1]], axis=-1) @ segment_to_ref.T)[:, :3]  # right multiply
                xyz = torch.from_numpy(xyz)
                flow = flow.to(torch.float32).cpu().numpy()
                flow = flow @ segment_to_ref.T[:3, :3]  # flow only apply rotations, not translations.
                flow = torch.from_numpy(flow)

            gaussians_ply_format = torch.zeros(torch.Size([1, xyz.shape[0], 14]))
            gaussians_ply_format[:, :, 0:3] = xyz.unsqueeze(0)
            gaussians_ply_format[:, :, 3:6] = color.unsqueeze(0)
            gaussians_ply_format[:, :, 6:7] = torch.ones([1, xyz.shape[0], 1])
            gaussians_ply_format[:, :, 7:10] = torch.ones([1, xyz.shape[0], 3]) * 0.01
            gaussians_ply_format[:, :, 10:14] = torch.ones([1, xyz.shape[0], 4]) * torch.tensor([1, 0, 0, 0])

            # flow
            flow = flow.to(gaussians_ply_format.device)
            flow = flow.unsqueeze(0)
            gaussians_ply_format = torch.concat([gaussians_ply_format, flow], dim=2)

            # pc: rgb color
            if self.with_feat:
                if os.getenv("CONTEXT_FEAT"):
                    semantic = output["render_results"]["rendered_semantic"]
                    semantic = semantic[0, t_idx]
                    semantic = rearrange(semantic, "v h w -> (v h w)")
                else:
                    feats = output["render_results"]["gs_rendered_decoded_feat"]
                    feats = feats[0, t_idx]
                    feats = rearrange(feats, "v h w c -> (v h w) c")
                    semantic = feat2class(feats, get_text_label_feats(SEMANTIC_LABEL_LIST), similarity_probs_threshold=self.similarity_probs_threshold)
                semantic = semantic.to(gaussians_ply_format.device)
                semantic = semantic.unsqueeze(0).unsqueeze(-1)
                gaussians_ply_format = torch.concat([gaussians_ply_format, semantic], dim=2)
            save_ply(gaussians_ply_format, os.path.join(save_path, f'pc_rgb_{target_frame_idxs[t_idx]}.ply'),
                     semantic_start_idx=17 if self.with_feat else None, flow_start_idx=14)

            # pc: semantic color
            if self.with_feat:
                color = torch.zeros_like(rendered_image[0, t_idx].reshape(-1, 3)).to(torch.float32)
                # color = render_results['gs_color'][:, t_idx]  # TODO: repeat does not deepcopy
                for class_idx in range(len(SEMANTIC_LABEL_LIST)):
                    color[semantic.squeeze(0).squeeze(-1) == class_idx] = torch.tensor(SEMANTIC_ID_TO_COLOR[class_idx]).to(color.device)
                color /= 255.
                gaussians_ply_format[:, :, 3:6] = color
                save_ply(gaussians_ply_format, os.path.join(save_path, f'pc_semantic_{target_frame_idxs[t_idx]}.ply'),
                         semantic_start_idx=17 if self.with_feat else None, flow_start_idx=14)

            # save camera pose
            filepath = os.path.join(save_path, f'camera_pose_{target_frame_idxs[t_idx]}.txt')
            with open(filepath, 'w') as f:
                for v_idx in range(v):
                    c2w = c2ws[t_idx*v + v_idx]
                    if 'segment_to_ref' in data_dict:
                        segment_to_ref = data_dict['segment_to_ref'][0].to(xyzs.device)
                        c2w = segment_to_ref @ c2w  # (row.T @ segment_to_ref.T).T
                    c2w_str = ' '.join(str(x) for x in (c2w).view(-1, 16).tolist())
                    f.write(c2w_str + '\n')
            print(f'Save frame_{target_frame_idxs[t_idx]} rendered pc in {save_path}.')

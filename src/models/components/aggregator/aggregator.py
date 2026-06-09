# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import logging
import torch
import torch.nn as nn
from typing import Tuple, List
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from src.models.components.layers import PatchEmbed
from src.models.components.layers.block import Block
from src.models.components.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from src.models.components.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from src.models.embedders import PluckerEmbedder, TimestepEmbedder
from src.models.components.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from src.models.components.layers.LinearRNN import DeepTemporalLinearRNN


logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class Aggregator(nn.Module):
    """
    An aggregator module that registers learnable special tokens (e.g., motion tokens, camera tokens)
    to capture high-level semantic cues related to dynamic content and viewpoint characteristics.
    These tokens are jointly embedded with visual patch features and participate in a two-stage
    attention process: first, frame-wise self-attention operates within each temporal frame to
    enable local token-patch interaction; second, global self-attention is applied across all
    frames and special tokens to model long-range dependencies and fuse information holistically.
    The output tokens (e.g., motion and camera tokens) thus aggregate spatiotemporal context from
    the entire input sequence through structured attention mechanisms.
    """
    def __init__(
        self,
        in_chans=3,
        num_cams=3,
        img_size=518,
        patch_size=14,
        decoder_type="dummy",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        num_motion_tokens=0,
        use_time_token=False,
        use_sky_token=False,
        use_affine_token=False,
        concat_plucker_embed=True,
        add_patch_plucker_embed=True,
        add_camera_embed=True,
        grad_checkpointing=True,
        use_rnn=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_type = decoder_type
        self.use_rnn = use_rnn
        self.concat_plucker_embed = concat_plucker_embed
        self.in_chans = in_chans
        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.num_cams = num_cams

        self.add_patch_plucker_embed = add_patch_plucker_embed
        self.add_camera_embed = add_camera_embed
        self.grad_checkpointing = grad_checkpointing

        if self.concat_plucker_embed:
            self.in_chans += 6

        if self.add_camera_embed:
            self.pose_encoding_mlp = nn.Linear(9, embed_dim)

        if self.add_patch_plucker_embed:
            self.patch_plucker_embed_mlp = nn.Linear(6, embed_dim)

        self.__build_patch_embed__(patch_embed, img_size, self.in_chans, patch_size, num_register_tokens, embed_dim=embed_dim)
        self.tlrnn = DeepTemporalLinearRNN(d_model=self.embed_dim, num_layers=4, expand=1, dropout=0.0) if self.use_rnn else None

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(self.depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(self.depth)
            ]
        )

        # ------- embedders -------
        self.plucker_embedder = PluckerEmbedder(img_size=img_size)
        self.time_embedder = TimestepEmbedder(embed_dim)

        if depth == 12:
            self.intermediate_layer_idx = [2, 5, 8, 11]
        elif depth == 24:
            self.intermediate_layer_idx = [4, 11, 17, 23]
        else:
            raise ValueError('only support depth layer 12 or 24!')

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # camera token
        # self.camera_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        nn.init.normal_(self.camera_token, std=1e-6)

        # register tokens
        # self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # ------- motion predictor -------
        self.num_motion_tokens = num_motion_tokens

        # ------- auxiliary tokens -------
        self.use_time_token = use_time_token
        self.use_sky_token = use_sky_token
        self.use_affine_token = use_affine_token

        if self.use_time_token:
            self.patch_start_idx += 1

        if self.num_motion_tokens > 0:
            self.patch_start_idx += num_motion_tokens
            self.motion_tokens = nn.Parameter(torch.randn(1, num_motion_tokens, embed_dim) * 0.0)
        else:
            self.motion_tokens = None

        if self.use_affine_token:
            self.patch_start_idx += self.num_cams
            self.affine_token = nn.Parameter(torch.randn(1, self.num_cams, embed_dim) * 0.0)

        if self.use_sky_token:
            self.patch_start_idx += 1
            self.sky_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.0)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 1, 3, 1, 1),
                persistent=False,
            )

    def forward(
        self,
        data_dict,
        mode: str = "causal",
        kv_cache_list: List[List[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            mode (str): Global attention mode, could be either "causal", "window" or "full"
            kv_cache_list (List[List[torch.Tensor]]): List of cached key-value pairs for
                each global attention layer of the aggregator

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        images = data_dict["context_image"]
        b, t, v, c, h, w = images.size()

        if c != 3:
            raise ValueError(f"Expected 3 input channels, got {c}")

        # GT camera pose
        _, ray_dict = self.get_ray_dict(data_dict)

        # Normalize images
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(b * t * v, c, h, w)

        if self.concat_plucker_embed:
            plucker_embeds = ray_dict["plucker"]
            plucker_embeds = rearrange(plucker_embeds, "b t v h w c-> (b t v) c h w")
            patch_tokens = self.patch_embed(torch.cat([images, plucker_embeds], dim=1))
        else:
            patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        # Add patch plucker_embed
        if self.add_patch_plucker_embed:
            patch_plucker_embeds = self.plucker_embedder(
                    data_dict["context_intrinsics"],
                    data_dict["context_camtoworlds"],
                    image_size=data_dict["context_image"].shape[-2:],
                    patch_size=self.patch_size,
                )['plucker']
            patch_plucker_embeds = rearrange(patch_plucker_embeds, "b t v h w c-> (b t v) (h w) c")
            patch_plucker_embeds_token = self.patch_plucker_embed_mlp(patch_plucker_embeds)
            patch_tokens += patch_plucker_embeds_token

        # Add time embedding
        patch_tokens = self._time_embed(patch_tokens, data_dict["context_time"], num_views=v)

        # Expand camera and register tokens to match batch size and sequence length
        # TODO: there's a concept of first frame here, need to handle correctly for multiview
        # TODO: predicting 20 frames, how to align with supervision frames
        is_anchor_exist = kv_cache_list is None or kv_cache_list[0][0] is None
        camera_token = slice_expand_and_flatten(self.camera_token, B=b, S=t*v, is_anchor_exist=is_anchor_exist)
        register_token = slice_expand_and_flatten(self.register_token, B=b, S=t*v, is_anchor_exist=is_anchor_exist)
        # camera_token = self.camera_token.expand(b, t*v, *self.camera_token.shape[2:]).view(b*t*v, *self.camera_token.shape[2:]).clone()
        # register_token = self.register_token.expand(b, t*v, *self.register_token.shape[2:]).view(b*t*v, *self.register_token.shape[2:]).clone()

        # Add pose embedding
        if self.add_camera_embed:
            intrinsic = rearrange(data_dict['context_intrinsics'], 'b t v ... -> b (t v) ...')
            extrinsic = rearrange(data_dict['context_camtoworlds'], 'b t v ... -> b (t v) ...').inverse()[..., :3, :4]
            pose_encoding = extri_intri_to_pose_encoding(extrinsic, intrinsic, images.shape[-2:])  # extrinsic is world to camera [b, s, 3, 4], intrinsic (no normalization) [b, s, 3, 3]
            pose_encoding = rearrange(pose_encoding, 'b tv ... -> (b tv) 1 ...')
            pose_emb = self.pose_encoding_mlp(pose_encoding)
            camera_token += pose_emb

        # Add time embedding
        camera_token = self._time_embed(camera_token, data_dict["context_time"], num_views=v)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token], dim=1)

        if self.use_time_token:
            time_token = self.time_embedder(data_dict["context_time"].flatten()).unsqueeze(1)
            tokens = torch.cat([tokens, time_token], dim=1)

        # NOTE: originally each clip has only one token, here copied t*v times to match above
        if self.num_motion_tokens > 0:
            motion_tokens = repeat(self.motion_tokens, "1 k d -> b k d", b=b*t*v)
            tokens = torch.cat([tokens, motion_tokens], dim=1)
        if self.use_affine_token:
            affine_token = repeat(self.affine_token, "1 k d -> b k d", b=b*t*v)
            tokens = torch.cat([tokens, affine_token], dim=1)
        if self.use_sky_token:
            sky_token = repeat(self.sky_token, "1 1 d -> b 1 d", b=b*t*v)
            tokens = torch.cat([tokens, sky_token], dim=1)

        tokens = torch.cat([tokens, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(b * t * v, h // self.patch_size, w // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(b * t * v, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        attn_mask = None
        if kv_cache_list is None:
            attn_mask = self._create_attn_mask(t, v * P, mode, tokens.dtype, tokens.device)

        frame_idx = 0
        global_idx = 0
        output_list = []

        for layer_idx in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, b, t * v, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if kv_cache_list is not None:
                        kv_cache = kv_cache_list[global_idx]
                        tokens, global_idx, global_intermediates, kv_cache = self._process_global_attention(
                            tokens, b, t * v, P, C, global_idx, pos=pos, attn_mask=attn_mask, kv_cache=kv_cache
                        )
                        kv_cache_list[global_idx-1] = kv_cache
                    else:
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, b, t * v, P, C, global_idx, pos=pos, attn_mask=attn_mask
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if self.tlrnn is not None:
                rnn_tokens, _ = self.tlrnn(tokens.view(b, t, v, P, C))
                rnn_tokens = rnn_tokens.view(b, t * v * P, C)
                # print(tokens.shape, rnn_tokens.shape)
                tokens = tokens + rnn_tokens

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                if layer_idx not in self.intermediate_layer_idx:
                    output_list.append(None)  # Reduce GPU usage
                else:
                    output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates

        if kv_cache_list is not None:
            return output_list, self.patch_start_idx, kv_cache_list
        else:
            return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training and self.grad_checkpointing:
                tokens = checkpoint(
                    self.frame_blocks[frame_idx],
                    tokens,
                    pos,
                    use_reentrant=False
                )
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, attn_mask=None, kv_cache=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S * P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if kv_cache is not None:
                if self.training and self.grad_checkpointing:
                    tokens, kv_cache = checkpoint(
                        self.global_blocks[global_idx],
                        tokens,
                        pos,
                        attn_mask,
                        kv_cache,
                        use_reentrant=False
                    )
                else:
                    tokens, kv_cache = self.global_blocks[global_idx](tokens, pos, attn_mask, kv_cache)
            else:
                if self.training and self.grad_checkpointing:
                    tokens = checkpoint(
                        self.global_blocks[global_idx],
                        tokens,
                        pos,
                        attn_mask,
                        kv_cache,
                        use_reentrant=False
                    )
                else:
                    tokens = self.global_blocks[global_idx](tokens, pos, attn_mask, kv_cache)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        if kv_cache is not None:
            return tokens, global_idx, intermediates, kv_cache

        return tokens, global_idx, intermediates

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        in_chans,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=518,
                in_chans=in_chans,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

            # NOTE: https://github.com/facebookresearch/vggt/issues/142
            pretrain = False if self.concat_plucker_embed else True
            if pretrain:
                url_dict = {
                    "dinov2_vitl14_reg": 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth',
                    "dinov2_vitb14_reg": 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth',
                    "dinov2_vits14_reg": 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth',
                    "dinov2_vitg2_reg": 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth',
                }
                state_dict = torch.hub.load_state_dict_from_url(url_dict[patch_embed], map_location="cpu")
                self.patch_embed.load_state_dict(state_dict, strict=True)

    def _create_attn_mask(self, S: int, P: int, mode: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        N = S * P
        mask = torch.zeros((N, N), dtype=dtype, device=device)

        if mode == "causal":
            for i in range(S):
                curr_view_start = i * P
                curr_view_end = (i + 1) * P
                mask[curr_view_start:curr_view_end, curr_view_end:] = float('-inf')
        elif "window" in mode:
            window_size = int(mode.split('_')[-1])
            for i in range(S):
                curr_view_start = i * P
                curr_view_end = (i + 1) * P
                if i < window_size:
                    # mask[curr_view_start:curr_view_end, P*window_size:] = float('-inf')
                    mask[curr_view_start:curr_view_end, (i+1)*P:] = float('-inf')
                else:
                    start_view = i - window_size + 1
                    mask[curr_view_start:curr_view_end, :start_view*P] = float('-inf')
                    mask[curr_view_start:curr_view_end, (i+1)*P:] = float('-inf')
        elif mode == "full":
            mask = None
        else:
            raise NotImplementedError(f"Unknown attention mode: {mode}")

        return mask

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


def slice_expand_and_flatten(token_tensor, B, S, is_anchor_exist=False):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    if is_anchor_exist:
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    else:
        query = token_tensor[:, 1:, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
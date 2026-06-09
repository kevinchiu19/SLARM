import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lseg.lseg_net import LSegNet
from tools import is_ascend_npu
if is_ascend_npu():
    import torch_npu
    import torchair
    backend = torchair.get_npu_backend()
else:
    backend = "inductor"  # "inductor", "cudagraphs"


def conditional_compile(use_compile, **compile_kwargs):
    """
    - use_compile: bool, enable torch.compile or not
    - compile_kwargs: args of torch.compile (mode, backend...)
    """
    def decorator(func):
        if use_compile:
            # Enable compilation, apply torch.compile and pass arguments
            return torch.compile(func, **compile_kwargs)
        else:
            # Disable compilation, return original function
            return func
    return decorator


class LSegFeatureExtractor:
    def __init__(self, lseg_model_pretrained_path, lseg_model_scratch_path, dtype=torch.float32):
        self.dtype = dtype
        lseg_net = LSegNet(
            labels=[],
            backbone='clip_vitl16_384',
            features=256,
            crop_size=480,
            arch_option=0,
            block_depth=0,
            activation='lrelu',
            dtype=self.dtype,
        )

        lseg_net.pretrained.model.patch_embed.img_size = (
            480,
            480,
        )

        self.model_pretrained = lseg_net.pretrained
        self.model_scratch = lseg_net.scratch

        print(f'Loading lseg_pretrained from {lseg_model_pretrained_path}.')
        state_dict = torch.load(lseg_model_pretrained_path)
        self.model_pretrained.load_state_dict(state_dict)
        print('lseg_pretrained loaded.')
        self.model_pretrained.eval()
        self.model_pretrained = self.model_pretrained.cuda()
        self.model_pretrained = self.model_pretrained.to(self.dtype)

        print(f'Loading lseg_scratch from {lseg_model_scratch_path}.')
        state_dict = torch.load(lseg_model_scratch_path)
        self.model_scratch.load_state_dict(state_dict)
        print('lseg_scratch loaded.')
        self.model_scratch.eval()
        self.model_scratch = self.model_scratch.cuda()
        self.model_scratch = self.model_scratch.to(self.dtype)

        # self.unflatten = nn.Sequential(
        #     nn.Unflatten(
        #         2,
        #         torch.Size(
        #             [
        #                 # 1280 // self.model_pretrained.model.patch_size[1],
        #                 # 1920 // self.model_pretrained.model.patch_size[0],
        #                 640 // self.model_pretrained.model.patch_size[1],
        #                 960 // self.model_pretrained.model.patch_size[0],
        #             ]
        #         ),
        #     )
        # )

    @torch.no_grad()
    def forward_vit(self, x):
        _, _, h, w = x.shape

        # encoder
        _ = self.model_pretrained.model.forward_flex(x)

        layer_1 = self.model_pretrained.activations["1"]
        layer_2 = self.model_pretrained.activations["2"]
        layer_3 = self.model_pretrained.activations["3"]
        layer_4 = self.model_pretrained.activations["4"]

        layer_1 = self.model_pretrained.act_postprocess1[0](layer_1)
        layer_2 = self.model_pretrained.act_postprocess2[0](layer_2)
        layer_3 = self.model_pretrained.act_postprocess3[0](layer_3)
        layer_4 = self.model_pretrained.act_postprocess4[0](layer_4)

        # Support dynamic shape when not using torch.compile
        unflatten = nn.Sequential(
            nn.Unflatten(
                2,
                torch.Size(
                    [
                        h // self.model_pretrained.model.patch_size[1],
                        w // self.model_pretrained.model.patch_size[0],
                    ]
                ),
            )
        )

        layer_1 = self.model_pretrained.act_postprocess1[1:3](layer_1)
        if layer_1.ndim == 3:
            # layer_1 = self.unflatten(layer_1)
            layer_1 = unflatten(layer_1)  # Support dynamic shape when not using torch.compile
        layer_1 = self.model_pretrained.act_postprocess1[3:](layer_1)

        layer_2 = self.model_pretrained.act_postprocess2[1:3](layer_2)
        if layer_2.ndim == 3:
            # layer_2 = self.unflatten(layer_2)
            layer_2 = unflatten(layer_2)  # Support dynamic shape when not using torch.compile
        layer_2 = self.model_pretrained.act_postprocess2[3:](layer_2)

        layer_3 = self.model_pretrained.act_postprocess3[1:](layer_3)
        if layer_3.ndim == 3:
            # layer_3 = self.unflatten(layer_3)
            layer_3 = unflatten(layer_3)  # Support dynamic shape when not using torch.compile

        layer_4 = self.model_pretrained.act_postprocess4[1:3](layer_4)
        if layer_4.ndim == 3:
            # layer_4 = self.unflatten(layer_4)
            layer_4 = unflatten(layer_4)  # Support dynamic shape when not using torch.compile
        layer_4 = self.model_pretrained.act_postprocess4[3:](layer_4)

        return layer_1, layer_2, layer_3, layer_4

    @torch.no_grad()
    @conditional_compile(
        os.getenv("COMPILE_FEAT_EXTRACTOR"),
        backend=backend,
        # mode="max-autotune",  # core dump, reason unknown
        dynamic=False,
        fullgraph=True,
    )
    def extract_lseg_feat(self, x, feat_size, interpolate_mode='nearest'):
        x = x.to(self.dtype)
        # print('x', x.dtype)
        feat_h, feat_w = feat_size

        layer_1, layer_2, layer_3, layer_4 = self.forward_vit(x)

        layer_1_rn = self.model_scratch.layer1_rn(layer_1)
        layer_2_rn = self.model_scratch.layer2_rn(layer_2)
        layer_3_rn = self.model_scratch.layer3_rn(layer_3)
        layer_4_rn = self.model_scratch.layer4_rn(layer_4)

        path_4 = self.model_scratch.refinenet4(layer_4_rn)
        path_3 = self.model_scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.model_scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.model_scratch.refinenet1(path_2, layer_1_rn)

        image_features = self.model_scratch.head1(path_1)  # torch.Size([1, 512, 640, 960])

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        image_features = image_features.to(self.dtype)

        if interpolate_mode == 'nearest':
            image_features = F.interpolate(
                image_features,
                size=(feat_h, feat_w),
                mode='nearest'
            )
        elif interpolate_mode == 'bilinear':
            image_features = F.interpolate(
                image_features,
                size=(feat_h, feat_w),
                mode='bilinear',  # Optional bilinear interpolation
                align_corners=True  # Bilinear interpolation align corners
            )
        else:
            raise ValueError

        return image_features

    @torch.no_grad()
    def extract_lseg_feat_one_by_one(self, x, feat_size, interpolate_mode='nearest'):
        # extract lseg feature one by one to avoid OOM
        # also can get better performance in conv operation
        image_features = []
        for i in range(len(x)):
            single_fram_feat = self.extract_lseg_feat(x[i].unsqueeze(0), feat_size, interpolate_mode)
            image_features.append(single_fram_feat)

        return torch.cat(image_features, dim=0)

    @torch.no_grad()
    def extract_lseg_feat_by_chunk(self, x, feat_size, interpolate_mode='nearest', step=1):
        # Avoid issue of interpolate exceeding max size
        if x.shape[0] <= step:
            return self.extract_lseg_feat(x, feat_size, interpolate_mode)
        else:
            for chunk_start in range(0, x.shape[0], step):
                chunk_end = min(chunk_start + step, x.shape[0])
                chunk_results = self.extract_lseg_feat(x[chunk_start:chunk_end], feat_size, interpolate_mode)
                if chunk_start == 0:
                    results = chunk_results
                else:
                    results = torch.cat([results, chunk_results], dim=0)
            return results

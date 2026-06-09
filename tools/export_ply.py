from plyfile import PlyData, PlyElement
import numpy as np
import os
import matplotlib
import torch


C0 = 0.28209479177387814


def SH2RGB(sh):
    return sh * C0 + 0.5

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def construct_dtypes(features_dc, features_rest, scale, rotation, use_fp16=False, enable_gs_viewer=True, sh_degree=0):
    if not use_fp16:
        l = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
        # All channels except the 3 DC
        if sh_degree > 0:
            for i in range(features_dc.shape[1] * features_dc.shape[2]):
                l.append((f"f_dc_{i}", "f4"))
        else:
            for i in range(features_dc.shape[1]):
                l.append((f"f_dc_{i}", "f4"))

        if enable_gs_viewer:
            assert sh_degree <= 3, "GS viewer only supports SH up to degree 3"
            if sh_degree > 0:
                sh_degree = 3
                for i in range(((sh_degree + 1) ** 2 - 1) * 3):
                    l.append((f"f_rest_{i}", "f4"))
        else:
            if sh_degree > 0:
                for i in range(
                    features_rest.shape[1] * features_rest.shape[2]
                ):
                    l.append((f"f_rest_{i}", "f4"))

        l.append(("opacity", "f4"))
        for i in range(scale.shape[1]):
            l.append((f"scale_{i}", "f4"))
        for i in range(rotation.shape[1]):
            l.append((f"rot_{i}", "f4"))
    else:
        l = [
            ("x", "f2"),
            ("y", "f2"),
            ("z", "f2"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
        # All channels except the 3 DC
        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append((f"f_dc_{i}", "f2"))

        if sh_degree > 0:
            for i in range(
                features_rest.shape[1] * features_rest.shape[2]
            ):
                l.append((f"f_rest_{i}", "f2"))
        l.append(("opacity", "f2"))
        for i in range(scale.shape[1]):
            l.append((f"scale_{i}", "f2"))
        for i in range(rotation.shape[1]):
            l.append((f"rot_{i}", "f2"))
    return l

def save_ply(gaussians, path, use_fp16=False, enable_gs_viewer=True, color_code=False, sh_degree=0,
             semantic_start_idx=None, flow_start_idx=None, mask_indices_start_idx=None):
    assert gaussians.shape[0] == 1, 'only support batch size 1'

    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = gaussians[0, :, 0:3].contiguous().float().detach().cpu().numpy()
    rgb = gaussians[0, :, 3:6].contiguous().float()
    # features_dc = torch.log(features_dc / (1 - features_dc + 1e-5) + 1e-5)
    features_dc = RGB2SH(rgb)
    opacities = gaussians[0, :, 6:7].contiguous().float().detach().cpu().numpy()
    scale = gaussians[0, :, 7:10].contiguous().float().detach().cpu().numpy()
    rotation = gaussians[0, :, 10:14].contiguous().float().detach().cpu().numpy()
    # Additional Gaussian properties
    if semantic_start_idx is not None:
        semantic = gaussians[0, :, semantic_start_idx:semantic_start_idx+1].contiguous().float().detach().cpu().numpy()
    if flow_start_idx is not None:
        flow = gaussians[0, :, flow_start_idx:flow_start_idx+3].contiguous().float().detach().cpu().numpy()
    if mask_indices_start_idx is not None:
        mask_indices = gaussians[0, :, mask_indices_start_idx:mask_indices_start_idx+1].contiguous().float().detach().cpu().numpy()

    if sh_degree > 0:
        features_rest = features_dc[:, 1:, :].contiguous()
    else:
        features_rest = None

    if sh_degree > 0:
        f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    else:
        f_dc = features_dc.detach().contiguous().cpu().numpy()
    if not color_code:
        rgb = (SH2RGB(f_dc) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    else:
        # use an color map to color code the index of points
        index = np.linspace(0, 1, xyz.shape[0])
        rgb = matplotlib.colormaps["viridis"](index)[..., :3]
        rgb = (rgb * 255.0).clip(0.0, 255.0).astype(np.uint8)

    # NOTE: Most visualization tools add a exp/sigmoid function during loading.
    # ref: https://github.com/antimatter15/splat/blob/main/convert.py
    scale = np.log(scale)
    opacities = np.log(opacities / (1 - opacities))

    dtype_full = construct_dtypes(features_dc, features_rest, scale, rotation,
                                use_fp16, enable_gs_viewer, sh_degree=sh_degree)
    if semantic_start_idx is not None:
        dtype_full.append(('semantic', 'u1'))
    if flow_start_idx is not None:
        dtype_full.extend([('flow_x', 'f4'), ('flow_y', 'f4'), ('flow_z', 'f4')])
    if mask_indices_start_idx is not None:
        dtype_full.append(('mask_indices', 'u4'))
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    f_rest = None
    if sh_degree > 0:
        f_rest = (
            features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )

    if enable_gs_viewer:
        if sh_degree > 0:
            sh_degree = 3
            if f_rest is None:
                f_rest = np.zeros((xyz.shape[0], 3*((sh_degree + 1) ** 2 - 1)), dtype=np.float32)
            elif f_rest.shape[1] < 3*((sh_degree + 1) ** 2 - 1):
                f_rest_pad = np.zeros((xyz.shape[0], 3*((sh_degree + 1) ** 2 - 1)), dtype=np.float32)
                f_rest_pad[:, :f_rest.shape[1]] = f_rest
                f_rest = f_rest_pad

    if f_rest is not None:
        attributes = np.concatenate(
            (xyz, rgb, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
    else:
        attributes = np.concatenate(
            (xyz, rgb, f_dc, opacities, scale, rotation), axis=1
        )
    if semantic_start_idx is not None:
        attributes = np.concatenate(
            (attributes, semantic), axis=1
        )
    if flow_start_idx is not None:
        attributes = np.concatenate(
            (attributes, flow), axis=1
        )
    if mask_indices_start_idx is not None:
        attributes = np.concatenate(
            (attributes, mask_indices), axis=1
        )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)

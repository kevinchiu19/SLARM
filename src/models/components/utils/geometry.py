# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def angular_velocity_to_quaternion(omega, dt):
    """
    Angular velocity -> Quaternion (rotation vector method)

    Args:
        omega: angular velocity (Tensor), shape (..., 3) [rad/s]
        dt: time step (float or Tensor)

    Returns:
        quat: quaternion (Tensor), shape (..., 4) [w, x, y, z]
    """
    rotvec = omega * dt  # rotation vector = angular velocity × time
    angle = torch.norm(rotvec, dim=-1, keepdim=True)  # rotation angle
    axis = rotvec / (angle + 1e-8)  # rotation axis (avoid division by zero)

    half_angle = angle * 0.5
    w = torch.cos(half_angle)
    xyz = torch.sin(half_angle) * axis

    quat = torch.cat([w, xyz], dim=-1)  # (..., 4)
    return quat


def quaternion_multiply(q1, q2):
    """
    Quaternion multiplication (q1 ⊗ q2)

    Args:
        q1: quaternion (Tensor), shape (..., 4) [w, x, y, z]
        q2: quaternion (Tensor), shape (..., 4) [w, x, y, z]

    Returns:
        product quaternion (Tensor), shape (..., 4) [w, x, y, z]
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Combine into new quaternions
    q3 = torch.stack([w, x, y, z], dim=-1)

    # Normalize the quaternions
    norm_q3 = q3 / torch.norm(q3, dim=-1, keepdim=True)

    return norm_q3


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    Convert an angle axis to a quaternion.
    Args:
        angle_axis (Tensor): Tensor of Nx3 representing the rotations
    Returns:
        quaternions (Tensor): Tensor of Nx4 representing the quaternions
    References:
        https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py

    Equations
    qx = ax * sin(angle/2)
    qy = ay * sin(angle/2)
    qz = az * sin(angle/2)
    qw = cos(angle/2)

    where:

    the axis is normalised so: ax*ax + ay*ay + az*az = 1
    the quaternion is also normalised so cos(angle/2)2 + ax*ax * sin(angle/2)2 + ay*ay * sin(angle/2)2+ az*az * sin(angle/2)2 = 1
    """
    angle = torch.norm(angle_axis, p=2, dim=-1, keepdim=True)  # N, 1
    half_angle = 0.5 * angle
    eps = 1e-6
    small_angle = angle.data.abs() < eps
    sin_half_angle = torch.sin(half_angle)
    cos_half_angle = torch.cos(half_angle)
    # for small angle, use taylor series
    sin_half_angle = torch.where(
        small_angle, half_angle - 0.5 * half_angle**3, sin_half_angle
    )
    cos_half_angle = torch.where(small_angle, 1 - 0.5 * half_angle**2, cos_half_angle)
    quaternions = torch.cat(
        [cos_half_angle, sin_half_angle * F.normalize(angle_axis, dim=-1)], dim=-1
    )
    return quaternions


def compute_normals_scales_torch(position_map: torch.Tensor):
    """
    Compute normal direction for each point in the 3D position map (supports 4D input, B×H×W×3)

    Args:
        position_map: torch.Tensor, shape (B, H, W, 3), representing 3D position of each pixel

    Returns:
        normals: torch.Tensor, shape (B, H, W, 3), unit normal vector for each point
        dx_scales: torch.Tensor, shape (B, H, W, 1), gradient magnitude in x direction
        dy_scales: torch.Tensor, shape (B, H, W, 1), gradient magnitude in y direction
    """
    if position_map.ndim == 3:
        # if input is (H, W, 3), add batch dimension
        position_map = position_map.unsqueeze(0)

    B, H, W, C = position_map.shape
    assert C == 3, "last dimension of input tensor must be 3 (x, y, z coordinates)"

    normals = torch.zeros_like(position_map)

    # Sobel operator (for computing gradients)
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=position_map.dtype, device=position_map.device) / 8.0

    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=position_map.dtype, device=position_map.device) / 8.0

    # expand Sobel operator to 4D tensor (1, 3, 3, 3)
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)#.repeat(1, 3, 1, 1)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)#.repeat(1, 3, 1, 1)

    # create Conv2d layer
    conv_x = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1, padding_mode="replicate")
    conv_y = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1, padding_mode="replicate")


    # set convolution kernel
    conv_x.weight.data = sobel_x
    conv_y.weight.data = sobel_y

    # convert position map from (B, H, W, 3) to (B, 3, H, W) for convolution
    position_map = position_map.permute(0, 3, 1, 2)

    # compute gradients in x and y directions
    dx = conv_x(position_map.reshape(-1,1,H,W)).reshape(B,C,H,W)
    dy = conv_y(position_map.reshape(-1,1,H,W)).reshape(B,C,H,W)

    # convert gradient tensors from (B, 3, H, W) to (B, H, W, 3)
    dx = dx.permute(0, 2, 3, 1)
    dy = dy.permute(0, 2, 3, 1)

    # compute normals (cross product dy × dx)
    normals_batch = torch.cross(dy, dx, dim=-1)
    # normalize
    norm = torch.norm(normals_batch, dim=-1, keepdim=True)
    norm = torch.where(norm == 0, torch.tensor(1e-10, device=norm.device), norm)  # avoid division by zero
    normals = normals_batch / norm

    dx_scales = torch.norm(dx, dim=-1, keepdim=True)
    dy_scales = torch.norm(dy, dim=-1, keepdim=True)

    if position_map.shape[0] == 1:
        normals = normals.squeeze(0)
        dx_scales = dx_scales.squeeze(0)
        dy_scales = dy_scales.squeeze(0)

    return normals, dx_scales, dy_scales, dx, dy


def compute_azimuth_tan(K):
    """
    Compute the tangent of azimuth angle in x direction for each pixel.
    """
    # extract focal length f_x
    f_x = K[:, 0, 0]  # shape [N, 1, 1]
    return 1/f_x


def scale_from_dxdy_torch(dx, dy, dz=1e-2):
    shape_target = list(dx.shape)
    shape_target[-1] = 3
    dxyz = torch.zeros(shape_target).to(dx.device)
    dxyz[..., 0:1] = dx
    dxyz[..., 1:2] = dy
    dxyz[..., 2:3] = dz
    return dxyz


def axes_to_quaternion_batch_torch(x_axes: torch.Tensor, y_axes: torch.Tensor, z_axes: torch.Tensor):
    """
    Convert batches of orthogonal axes to rotation quaternions.

    Parameters:
    x_axes (torch.Tensor): The x-axis directions after rotation, shape (N, 3).
    y_axes (torch.Tensor): The y-axis directions after rotation, shape (N, 3).
    z_axes (torch.Tensor): The z-axis directions after rotation, shape (N, 3).

    Returns:
    quats: A tensor of quaternions [w, x, y, z], shape (N, 4).
    """
    # Check input shapes
    if x_axes.ndim != 2 or y_axes.ndim != 2 or z_axes.ndim != 2:
        raise ValueError("Input axes must be 2D tensors of shape (N, 3)")
    if x_axes.shape[1] != 3 or y_axes.shape[1] != 3 or z_axes.shape[1] != 3:
        raise ValueError("Axes must be 3D vectors")
    N = x_axes.shape[0]
    if y_axes.shape[0] != N or z_axes.shape[0] != N:
        raise ValueError("All input axes must have the same number of points")

    # Normalize the axes
    x_axes = x_axes / torch.norm(x_axes, dim=1, keepdim=True)
    y_axes = y_axes / torch.norm(y_axes, dim=1, keepdim=True)
    z_axes = z_axes / torch.norm(z_axes, dim=1, keepdim=True)

    # Ensure orthogonality: adjust y_axes to be orthogonal to x_axes
    dot_xy = torch.sum(x_axes * y_axes, dim=1, keepdim=True)
    y_axes = y_axes - dot_xy * x_axes
    y_axes = y_axes / torch.norm(y_axes, dim=1, keepdim=True)

    # Recompute z_axes as the cross product of x_axes and y_axes
    z_axes = torch.cross(x_axes, y_axes, dim=1)
    z_axes = z_axes / torch.norm(z_axes, dim=1, keepdim=True)

    # Construct rotation matrices (N, 3, 3)
    rotation_matrices = torch.stack([x_axes, y_axes, z_axes], dim=2)

    # Ensure proper rotation matrices (det ~1)
    dets = torch.det(rotation_matrices)
    if not torch.allclose(dets, torch.tensor(1.0, device=dets.device), atol=1e-6):
        # Flip z-axis for matrices with det ~ -1
        mask = dets < 0
        rotation_matrices[mask, :, 2] *= -1

    # Convert rotation matrices to quaternions
    quats = matrix_to_quaternion(rotation_matrices)  # Shape (N, 4), [x, y, z, w]

    return quats


def compute_axes_from_normal_torch(point_normal: torch.Tensor, camera_up: torch.Tensor = None):
    """
    Compute right and up vectors for the given point normal direction
    """
    if camera_up is None:
        camera_up = torch.tensor([0, 1, 0], dtype=point_normal.dtype, device=point_normal.device)
        camera_up = camera_up.expand((1,) * (len(point_normal.shape) - 1) + (3,))
    right = torch.cross(camera_up, point_normal, dim=-1)
    up = torch.cross(point_normal, right, dim=-1)
    # normalize right
    norm_right = torch.norm(right, dim=-1, keepdim=True)
    norm_right = torch.where(norm_right == 0, torch.tensor(1e-10, device=norm_right.device), norm_right)  # avoid division by zero
    right = right / norm_right
    # normalize up
    norm_up = torch.norm(up, dim=-1, keepdim=True)
    norm_up = torch.where(norm_up == 0, torch.tensor(1e-10, device=norm_up.device), norm_up)  # avoid division by zero
    up = up / norm_up
    return right, up, point_normal


def rot_from_normals_torch(point_normal: torch.Tensor, up: torch.Tensor = None):
    origin_shape = list(point_normal.shape)
    point_normal=point_normal.reshape(-1,3)
    up = up.reshape(-1, 3) if up != None else None
    x_axes, y_axes, z_axes = compute_axes_from_normal_torch(point_normal, up)
    rot = axes_to_quaternion_batch_torch(x_axes=x_axes, y_axes=y_axes, z_axes=z_axes)
    origin_shape[-1] = 4
    return rot.reshape(origin_shape)


# from utils from pixel splat
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


# from utils from pixel splat
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


# from utils from pixel splat
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

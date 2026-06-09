import json
import yaml
import numpy as np
import open3d as o3d
import os
import cv2
import glob


def depth2xyz(depth, fxfycxcy, cam2world=None, return_pixel=False):
    h, w = depth.shape

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = (x + 0.5) / w
    y = (y + 0.5) / h
    x = (x - fxfycxcy[2:3]) / fxfycxcy[0:1]
    y = (y - fxfycxcy[3:4]) / fxfycxcy[1:2]
    z = np.ones_like(x)
    ray_d = np.stack([x, y, z], axis=2)  # [b*v, h*w, 3]

    # debug
    # cam2world = np.eye(4)
    if cam2world is not None:
        ray_d = np.matmul(ray_d, cam2world[:3, :3].T)  # [b*v, h*w, 3]  a @ b.T
        ray_o = cam2world[:3, 3]

    # ray_d = ray_d / np.linalg.norm(ray_d, axis=2, keepdims=True)  # Convert to Ray coordinate system

    xyz = ray_d * depth[..., None]

    if cam2world is not None:
        xyz = ray_o + ray_d * depth[..., None]

    if return_pixel:
        return xyz
    else:
        return xyz.reshape(-1, 3)


def load_cams(cam_file):
    c2w = []
    with open(cam_file, 'r') as f:
        line = f.readline()
        while line and line.strip():
            matrix = [float(x) for x in line.strip().split(' ')]
            matrix = np.array(matrix).reshape((4, 4))
            c2w.append(matrix)

            line = f.readline()

    c2w = np.stack(c2w, 0)  # [::-1]
    return c2w


data_root = 'data/SLARM_data/'
dataset_name = 'driving_sim'
datasets_path = 'training'
cam_list = ["front_left", "front", "front_right", "rear_left", "rear", "rear_right"]

source_path = os.path.join(data_root, 'datasets', dataset_name, datasets_path)
scene_list = os.listdir(source_path)

# scene = 'lidar_merge_0703_six_cam_v1'
for scene in scene_list:
    frame_nums = len(os.listdir(os.path.join(data_root, 'datasets', dataset_name, datasets_path , scene, 'front/vis/color')))
    # frame_nums = 300
    assert frame_nums > 0

    annotations = {}
    annotations['dataset'] = dataset_name
    annotations['scene_id'] = 1
    annotations['scene_name'] = scene
    annotations['num_timesteps'] = frame_nums
    annotations['camera_list'] = cam_list
    annotations['normalized_time'] = [i / 10 for i in range(frame_nums)]  # TODO: frame rate?

    normalized_intrinsics_dict = {}
    camera_to_world_dict = {}
    original_image_size_dict = {}
    relative_image_path_dict = {}
    for cam in cam_list:
        camera_paras_path = os.path.join(data_root, 'datasets', dataset_name, datasets_path, scene, cam, 'camera.yaml')
        with open(camera_paras_path, 'r') as f:
            cam_info = yaml.load(f, Loader=yaml.FullLoader)
            camera_paras = np.array(tuple(map(float, [cam_info['K']['data'][0], cam_info['K']['data'][4], cam_info['K']['data'][2], cam_info['K']['data'][5]])))
            W, H = cam_info['Imagesize']
            original_image_size_dict[cam] = [H, W]
            fxfycxcy = np.array([camera_paras[0] / W, camera_paras[1] / H, camera_paras[2] / W, camera_paras[3] / H])
            normalized_intrinsics_dict[cam] = fxfycxcy.tolist()

        # camera extrinsic
        trj_txt_filepath = glob.glob(os.path.join(data_root, 'datasets', dataset_name, datasets_path, scene, cam, 'trj_*'))
        trj_txt_filename = trj_txt_filepath[0].split('/')[-1]
        camera_trajs_path = os.path.join(data_root, 'datasets', dataset_name, datasets_path, scene, cam, trj_txt_filename)
        c2ws = load_cams(camera_trajs_path)

        tmp = []
        tmp2 = []
        for frame_idx in range(frame_nums):
            tmp.append(c2ws[frame_idx].tolist())
            tmp2.append(os.path.join(datasets_path, scene, cam, 'vis', 'color_downsample', f'{frame_idx:08d}_color_vis.png'))
        camera_to_world_dict[cam] = tmp
        relative_image_path_dict[cam] = tmp2

        '''
        c2w = c2ws[frame_idx]
        rgb_path = os.path.join(data_root, 'datasets', dataset_name, datasets_path, scene, cam, 'vis', 'color_downsample', f'{frame_idx:08d}_color_vis.png')
        rgb_image = cv2.imread(rgb_path)
        rgb_image = rgb_image / 255.0

        depth_path = os.path.join(data_root, 'datasets', dataset_name, datasets_path, scene, cam, 'vis', 'depth_downsample', f'{frame_idx:08d}_depth.tif')
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # Filter points that are too far away
        depth_image[depth_image > 200] = 0.0

        xyz = depth2xyz(depth_image, fxfycxcy, c2w)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb_image.reshape(-1, 3))
        o3d.io.write_point_cloud((f"cam{cam}_{frame_idx}_pts3d.ply"), pcd)
        '''

    annotations['normalized_intrinsics'] = normalized_intrinsics_dict
    annotations['camera_to_world'] = camera_to_world_dict
    annotations['original_image_size'] = original_image_size_dict
    annotations['relative_image_path'] = relative_image_path_dict

    # TODO: camera_to_ego, ego_to_world for flow
    annotations['fps'] = 10

    with open(f"{data_root}/annotations/{dataset_name}/{datasets_path}/{scene}.json", "w") as f:
        json.dump(annotations, f)

    print('Saving: ', f"{data_root}/annotations/{dataset_name}/{datasets_path}/{scene}.json")


print('done')

# sky mask
# find data/SLARM_data/datasets/driving_sim/training/*/*/vis/color -name "*.png" > file_list.txt
# CUDA_VISIBLE_DEVICES=7 python extract_sky.py --file_list ./file_list.txt --rgb_type png --rgb_dir color

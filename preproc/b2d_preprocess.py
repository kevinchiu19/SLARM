import json
import os
import glob
import yaml
import numpy as np
import open3d as o3d #in docker
import cv2
import gzip


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

    # ray_d = ray_d / np.linalg.norm(ray_d, axis=2, keepdims=True)  # convert to Ray coordinate system

    xyz = ray_d * depth[..., None]

    if cam2world is not None:
        xyz = ray_o + ray_d * depth[..., None]

    if return_pixel:
        return xyz
    else:
        return xyz.reshape(-1, 3)

stand_to_ue4_rotate = np.array([[ 0, 0, 1, 0],
                                [ 1, 0, 0, 0],
                                [ 0,-1, 0, 0],
                                [ 0, 0, 0, 1]])

left2right = np.eye(4)
left2right[1, 1] = -1

def convert_extrinsic_4x4_left_to_right(T_left):
    return np.linalg.inv(stand_to_ue4_rotate) @ T_left @ left2right

def load_json_gz(file):
    with gzip.open(file, 'rt', encoding='utf-8') as gz_file:
        anno = json.load(gz_file)
        gz_file.close()
    return anno

b2d_camera_tags2path_name = {
    'CAM_FRONT': 'rgb_front',
    'CAM_FRONT_LEFT': 'rgb_front_left',
    'CAM_FRONT_RIGHT': 'rgb_front_right',
    'CAM_BACK': 'rgb_back',
    'CAM_BACK_LEFT': 'rgb_back_left',
    'CAM_BACK_RIGHT': 'rgb_back_right',
}


data_root = 'data/SLARM_data'
dataset_name = 'b2d'
datasets_path = 'training'
cam_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
# cam_list = ["rgb_front_left", "rgb_front", "rgb_front_right", "rgb_back_right", "rgb_back", "rgb_back_left"]
dataset_frequency = 10

source_path = os.path.join(data_root, 'datasets', dataset_name, datasets_path)
scene_list = os.listdir(source_path)
scene_list = [s for s in scene_list if not s.endswith('.tar')]  # there are tar files in the directory

for scene_idx, scene in enumerate(scene_list):
    frame_nums = len(os.listdir(os.path.join(source_path, scene, "anno")))

    if frame_nums == 0:
        continue

    annotations = {}
    annotations['dataset'] = dataset_name
    annotations['scene_id'] = scene_idx
    annotations['scene_name'] = scene
    annotations['num_timesteps'] = frame_nums
    annotations['camera_list'] = cam_list
    annotations['normalized_time'] = [i / dataset_frequency for i in range(frame_nums)]

    normalized_intrinsics_dict = {}
    camera_to_world_dict = {}
    camera_to_ego_dict = {}
    original_image_size_dict = {}
    relative_image_path_dict = {}
    for cam in cam_list:
        frame_idx_str = "{:05d}".format(0)  # first frame
        anno_path = os.path.join(source_path, scene, "anno", f"{frame_idx_str}.json.gz")
        anno = load_json_gz(anno_path)
        intrinsic = np.array(anno['sensors'][cam]['intrinsic'])
        img_size_x = anno['sensors'][cam]['image_size_x']
        img_size_y = anno['sensors'][cam]['image_size_y']
        normalized_intrinsic = [
            intrinsic[0, 0] / img_size_x,
            intrinsic[1, 1] / img_size_y,
            intrinsic[0, 2] / img_size_x,
            intrinsic[1, 2] / img_size_y
        ]
        original_image_size_dict[cam] = [img_size_y, img_size_x]
        normalized_intrinsics_dict[cam] = normalized_intrinsic
        # camera_to_ego
        camera_to_ego = np.array(anno['sensors'][cam]['cam2ego']).reshape(4, 4)
        ego_to_camera = np.linalg.inv(camera_to_ego)
        ego_to_camera = convert_extrinsic_4x4_left_to_right(ego_to_camera)
        camera_to_ego_dict[cam] = np.linalg.inv(ego_to_camera).tolist()

        # camera extrinsic
        tmp = []
        tmp2 = []
        for frame_idx in range(frame_nums):
            frame_idx_str = "{:05d}".format(frame_idx)
            anno_path = os.path.join(source_path, scene, "anno", f"{frame_idx_str}.json.gz")
            anno = load_json_gz(anno_path)
            world2cam = np.array(anno['sensors'][cam]['world2cam'])
            world2cam = convert_extrinsic_4x4_left_to_right(world2cam)
            c2w = np.linalg.inv(world2cam)
            tmp.append(c2w.tolist())
            relative_image_path = os.path.join(datasets_path, scene, "camera", b2d_camera_tags2path_name[cam], f"{frame_idx_str}.jpg")
            tmp2.append(relative_image_path)
        camera_to_world_dict[cam] = tmp
        relative_image_path_dict[cam] = tmp2

        '''
        # Generate colored PLY point clouds from depth maps, RGB images, and camera intrinsic/extrinsic parameters
        fxfycxcy = np.array(normalized_intrinsic)
        rgb_path = os.path.join(data_root, 'datasets', dataset_name, relative_image_path)
        rgb_image = cv2.imread(rgb_path)
        rgb_image = rgb_image / 255.0

        depth_path = rgb_path.replace("rgb", "depth").replace(".jpg", ".png")
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.01

        # Filter points that are too far away
        depth_image[depth_image > 400] = 0.0

        xyz = depth2xyz(depth_image, fxfycxcy, c2w)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb_image.reshape(-1, 3))
        o3d.io.write_point_cloud((f"{cam}_{scene}_pts3d.ply"), pcd)
        '''

    annotations['normalized_intrinsics'] = normalized_intrinsics_dict
    annotations['camera_to_world'] = camera_to_world_dict
    annotations['camera_to_ego'] = camera_to_ego_dict
    annotations['original_image_size'] = original_image_size_dict
    annotations['relative_image_path'] = relative_image_path_dict
    annotations['fps'] = dataset_frequency

    save_dir = os.path.join(data_root, 'annotations', dataset_name, datasets_path)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{scene}.json')
    with open(save_path, "w") as f:
        json.dump(annotations, f)
    print(f'Saving: {save_path}')

print('done')

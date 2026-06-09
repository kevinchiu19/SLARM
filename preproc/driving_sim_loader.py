import os
import numpy as np
import cv2
from argparse import ArgumentParser


def load_rgb_data(file_path):
    data = cv2.imread(file_path)
    print(f"rgb: shape {data.shape}, min {data.min()}, max {data.max()}")
    return data

def load_depth_data(file_path):
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    data_valid_mask = data < 65535  # data is clipped at 65535, i.e. 655.35m
    data = data.astype(np.float32) / 100.
    print(f"depth: shape {data.shape}, min {data.min()}, max {data.max()}")
    return data, data_valid_mask

def load_normal_data(file_path):
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    data = data.astype(np.float32)
    # data = data / 100 - 1.
    data = data / 255 * 2 - 1
    print(f"normal: shape {data.shape}, min {data.min()}, max {data.max()}")
    return data

def load_semantic_data(file_path):
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    print(f"semantic: shape {data.shape}, min {data.min()}, max {data.max()}")
    return data

def load_instance_data(file_path):
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    cls_data = data[..., 0]
    inst_data = data[..., 1:].astype(np.uint16)
    inst_data = inst_data[..., 0] * 256 + inst_data[..., 1]
    print(f"semantic (from instance): shape {cls_data.shape}, min {cls_data.min()}, max {cls_data.max()}")
    print(f"instance: shape {inst_data.shape}, min {inst_data.min()}, max {inst_data.max()}")
    return cls_data, inst_data

def load_scene_flow_data(file_path):
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    data = data.astype(np.float32)
    data = (data / 65535 * 2 - 1) * 5  # Unit: meters
    data = data * 20  # 20Hz -> Unit: m/s
    # (+x: front, +y: left, +z: up)
    print(f"scene flow: shape {data.shape}, min {data.min()}, max {data.max()}")
    print(f"X-direction velocity: min = {data[..., 0].min():.4f} m/s, max = {data[..., 0].max():.4f} m/s")
    print(f"Y-direction velocity: min = {data[..., 1].min():.4f} m/s, max = {data[..., 1].max():.4f} m/s")
    print(f"Z-direction velocity: min = {data[..., 2].min():.4f} m/s, max = {data[..., 2].max():.4f} m/s")
    return data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--data_folder', type=str,
                        default="xxx/processed_data/datasets/driving_sim/training/040")
    args = parser.parse_args()
    data_folder = args.data_folder

    rgb_path = os.path.join(data_folder, "rgb_front", f"{0:05d}.jpg")
    load_rgb_data(rgb_path)

    depth_path = os.path.join(data_folder, "depth_front", f"{0:05d}.png")
    load_depth_data(depth_path)

    normal_path = os.path.join(data_folder, "normal_front", f"{0:05d}.png")
    load_normal_data(normal_path)

    semantic_path = os.path.join(data_folder, "semantic_front", f"{0:05d}.png")
    load_semantic_data(semantic_path)

    instance_path = os.path.join(data_folder, "instance_front", f"{0:05d}.png")
    load_instance_data(instance_path)

    scene_flow_path = os.path.join(data_folder, "scene_flow_front",f"00001.png")
    load_scene_flow_data(scene_flow_path)


'''
# check waymo flow & depth
flow_path = "xxx/SLARM_data/datasets/waymo/training/089/depth_flows_4/198_0.npy"

depth_and_flow = np.load(flow_path)
flow_np = depth_and_flow[..., 1:]
depth_np = depth_and_flow[..., 0]

print(f"flow shape: {flow_np.shape}")  # (h, w, 3)
print(f"X-direction velocity: min = {flow_np[..., 0].min():.4f} m/s, max = {flow_np[..., 0].max():.4f} m/s")
print(f"Y-direction velocity: min = {flow_np[..., 1].min():.4f} m/s, max = {flow_np[..., 1].max():.4f} m/s")
print(f"Z-direction velocity: min = {flow_np[..., 2].min():.4f} m/s, max = {flow_np[..., 2].max():.4f} m/s")

print(f"depth shape: {depth_np.shape}, min {depth_np.min()}, max {depth_np.max()}")
'''

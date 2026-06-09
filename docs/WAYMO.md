# Preparing Waymo Dataset
> Note: This document is modified from [GaussianSTORM](https://github.com/NVlabs/GaussianSTORM/blob/main/docs/WAYMO.md).

## 1. Register on Waymo Open Dataset

#### Sign Up for a Waymo Open Dataset Account and Install gcloud SDK

To download the Waymo dataset, you need to register an account at [Waymo Open Dataset](https://waymo.com/open/). You also need to install gcloud SDK and authenticate your account. Please refer to [this page](https://cloud.google.com/sdk/docs/install) for more details.

#### Set Up the Data Directory

Once you've registered and installed the gcloud SDK, create a directory to house the raw data:

```bash
# create the data directory or create a symbolic link to the data directory
# You may need to download two original datasets because the scene flow annotations and semantic annotations in the Waymo dataset are not in the same file. You can control the version you download by specifying the version parameter.
mkdir -p ./data/waymo/raw_scene_flow
mkdir -p ./data/waymo/raw_1_4_3
```

## 2. Environment

We highly recommend setting up another environment for data processing as the TensorFlow dependencies often conflict with our main environment.
```bash
conda create -n data_process python=3.10
conda activate data_process
pip install -r requirements_data_preprocess.txt
```

## 3. Download the Raw Data
For the Waymo Open Dataset, we first organize the scene names alphabetically and store them in `data/dataset_scene_list/waymo_train_list.txt` and `data/dataset_scene_list/waymo_val_list.txt`. The scene index is then determined by the line number minus one.

For example, you can download 3 sequences from the dataset by:

```bash
python preproc/waymo_download.py \
    --target_dir ./data/waymo/raw_scene_flow/training \
    --split_file data/dataset_scene_list/waymo_train_list.txt \
    --scene_ids 700 754 23
```
If you wish to run experiments on different scenes, please specify your own list of scenes.

You can also omit the `scene_ids` to download all scenes specified in the `split_file`:

```bash
# training set (for scene flow)
python preproc/waymo_download.py \
    --target_dir ./data/waymo/raw_scene_flow/training \
    --split_file data/dataset_scene_list/waymo_train_list.txt \
    --version waymo_open_dataset_scene_flow

# training set (for semantic)
python preproc/waymo_download.py \
    --target_dir ./data/waymo/raw_1_4_3/training \
    --split_file data/dataset_scene_list/waymo_train_list.txt \
    --version waymo_open_dataset_v_1_4_3

# validation set (for scene flow)
python preproc/waymo_download.py \
    --target_dir ./data/waymo/raw_scene_flow/validation \
    --split_file data/dataset_scene_list/waymo_val_list.txt \
    --version waymo_open_dataset_scene_flow

# validation set (for semantic)
python preproc/waymo_download.py \
    --target_dir ./data/waymo/raw_1_4_3/validation \
    --split_file data/dataset_scene_list/waymo_val_list.txt \
    --version waymo_open_dataset_v_1_4_3

```

<details>
<summary>If this script doesn't work due to network issues, consider manual download:</summary>

Download the [scene flow version](https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) of Waymo.

Download the [sematic version](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_3;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) of Waymo.

![Waymo Dataset Download Page](https://github.com/user-attachments/assets/a1737699-e792-4fa0-bb68-0ab1813f1088)

> **Note**: Ensure you're downloading the scene flow version to avoid errors.

</details>

## 4. Preprocess the Data
After downloading the raw dataset, you'll need to preprocess this compressed data to extract and organize various components.

#### Run the Preprocessing Script
To preprocess specific scenes of the dataset, use the following command:
```bash
python preprocess.py \
    --data_root data/waymo/raw_scene_flow/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --scene_list_file data/dataset_scene_list/waymo_train_list.txt \
    --scene_ids 700 754 23 \
    --num_workers 8 \
    --process_keys images lidar calib pose dynamic_masks ground \
    --json_folder_to_save data/SLARM_data/annotations/waymo 
```
Alternatively, preprocess a batch of scenes by providing the split file:
```bash
# training set (for scene flow)
python preprocess.py \
    --data_root data/waymo/raw_scene_flow/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --scene_list_file data/dataset_scene_list/waymo_train_list.txt \
    --num_scenes 798 \
    --num_workers 8 \
    --process_keys images lidar calib pose dynamic_masks ground \
    --json_folder_to_save data/SLARM_data/annotations/waymo 

# training set (for semantic)
python preprocess.py \
    --data_root data/waymo/raw_1_4_3/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --scene_list_file data/dataset_scene_list/waymo_train_list.txt \
    --num_scenes 798 \
    --num_workers 8 \
    --process_keys panoptic_segs \
    --json_folder_to_save data/SLARM_data/annotations/waymo \
    --only_process_semantic

# validation set (for scene flow)
python preprocess.py \
    --data_root data/waymo/raw_scene_flow/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split validation \
    --scene_list_file data/dataset_scene_list/waymo_val_list.txt \
    --num_scenes 202 \
    --num_workers 8 \
    --process_keys images lidar calib pose dynamic_masks ground \
    --json_folder_to_save data/SLARM_data/annotations/waymo 

# validation set (for semantic)
python preprocess.py \
    --data_root data/waymo/raw_1_4_3/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split validation \
    --scene_list_file data/dataset_scene_list/waymo_val_list.txt \
    --num_scenes 202 \
    --num_workers 8 \
    --process_keys panoptic_segs \
    --json_folder_to_save data/SLARM_data/annotations/waymo \
    --only_process_semantic 
```
The extracted data will be stored in the `data/waymo/processed` directory.

## 5. Extract Sky Masks

- We originally used `ViT-adapater`, following [EmerNeRF](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md#sky-mask), to extract sky masks. We later realized the non-trivial dependency of installing `mmcv` properly.

- Therefore, we recommend using `DepthAnything-v2` to extract the sky masks. Specifically, the predicted depth map from DepthAnything-v2 is a relative depth map, where lower values indicate farther regions, and zero values indicate infinitely far regions, which represent the sky.

- To do this:

    1. Use whatever command you prefer to generate the file list to all image files, e.g., `find data/waymo/processed/training/*/images -name "*.jpg" > file_list.txt`
    2. Download DpethAnything-v2 checkpoint, e.g., `mkdir ckpts && wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -O ckpts/depth_anything_v2_vitl.pth`
    2. Use `extract_sky_mask.py` to extract sky masks for these images by `python extract_sky.py --file_list ./file_list.txt`

## 6. Data Structure
After completing all preprocessing steps, the project files should be organized according to the following structure:
```bash
ProjectPath/data/
  └── waymo/
    ├── raw_scene_flow/
    │    ├── training/
    │    │    ├── segment-9654060644653474834_3905_000_3925_000_with_camera_labels.tfrecord
    │    │    └── ...
    │    └── validation/
    │         ├── segment-967082162553397800_5102_900_5122_900_with_camera_labels.tfrecord
    │         └── ...
    ├── raw_1_4_3/
    │    ├── training/
    │    │    ├── segment-9509506420470671704_4049_100_4069_100_with_camera_labels.tfrecord
    │    │    └── ...
    │    └── validation/
    │         ├── segment-967082162553397800_5102_900_5122_900_with_camera_labels.tfrecord
    │         └── ...
    └── processed/
         └── training/
              ├── 000/
              │  ├──cam_to_ego/         # camera to ego-vehicle transformations: {cam_id}.txt
              │  ├──cam_to_world/       # camera to world transformations: {timestep:03d}_{cam_id}.txt
              │  ├──depth_flows_4/      # downsampled (1/4) depth flow maps: {timestep:03d}_{cam_id}.npy
              │  ├──dynamic_masks/      # bounding-box-generated dynamic masks: {timestep:03d}_{cam_id}.png
              │  ├──ego_to_world/       # ego-vehicle to world transformations: {timestep:03d}.txt
              │  ├──ground_label_4/     # downsampled (1/4) ground labels extracted from point cloud, used for flow evaluation only: {timestep:03d}.txt
              │  ├──images/             # original camera images: {timestep:03d}_{cam_id}.jpg
              │  ├──images_4/           # downsampled (1/4) camera images: {timestep:03d}_{cam_id}.jpg
              │  ├──instance_segs/      # original instance label: {timestep:03d}_{cam_id}.npy
              │  ├──instance_segs_4/    # downsampled (1/4) instance label: {timestep:03d}_{cam_id}.npy
              │  ├──intrinsics/         # camera intrinsics: {cam_id}.txt
              │  ├──lidar/              # lidar data: {timestep:03d}.bin
              │  ├──semantic_segs/      # original semantic label: {timestep:03d}_{cam_id}.npy
              │  ├──semantic_segs_4/    # downsampled (1/4) semantic label: {timestep:03d}_{cam_id}.npy
              │  ├──sky_masks/          # sky masks: {timestep:03d}_{cam_id}.png
              ├── 001/
              ├── ...
```

## 7. Final Preparations

1. Create a symbolic link to the directory  (Change the xxx to an absolute path)
mkdir -p data/SLARM_data/datasets
ln -s xxx/data/waymo/processed data/SLARM_data/datasets/waymo

2. Generate a txt file specifying the scene during training/validation.
a. create file
mkdir -p data/SLARM_data/scene_list/
touch data/SLARM_data/scene_list/waymo_train.txt
touch data/SLARM_data/scene_list/waymo_val.txt

b. Use Python to read parsed data.  (The same goes for validation)
python -c 'import os; [print(os.path.join("annotations/waymo/training/", i)) for i in os.listdir("data/SLARM_data/annotations/waymo/training")]'

c. and then copy to data/SLARM_data/scene_list/waymo_train.txt.

This data/SLARM_data/scene_list/waymo_train.txt is like:
annotations/waymo/training/segment-10275144660749673822_5755_561_5775_561_with_camera_labels.json
annotations/waymo/training/segment-9320169289978396279_1040_000_1060_000_with_camera_labels.json
annotations/waymo/training/segment-17902907331132202998_1564_000_1584_000_with_camera_labels.json
...

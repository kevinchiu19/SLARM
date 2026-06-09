import os


# fpath = 'xxx/SLARM_data/datasets/waymo/validation'
# scenes = 202
fpath = 'xxx/SLARM_data/datasets/waymo/training'
scenes = 798

num_scenes_all = 0
num_samples_all = 0
for scene_id in range(scenes):
    sub_fpath = os.path.join(fpath, str(scene_id).zfill(3), 'semantic_segs')
    labels_num = len(os.listdir(sub_fpath))
    if labels_num > 0:
        num_scenes_all += 1
        num_samples_all += labels_num

print(num_scenes_all)
print(num_samples_all)

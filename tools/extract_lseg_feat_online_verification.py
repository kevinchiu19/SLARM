import os
import torch
import torchvision.transforms as transforms
import numpy as np
from glob import glob
from PIL import Image
from sklearn.decomposition import PCA

from utils.lseg_feat_extractor import LSegFeatureExtractor
from src.dataset.constants import MEAN, STD


IMG_H, IMG_W = 1280, 1920  # Input image height and width, extract features on original image
FEAT_H, FEAT_W = 168, 252  # Saved feature height and width, corresponding to training image dimensions

# Scene ID and image ID for verification
SCENE_ID = '089'
IMG_ID = 0


data_path = f'xxx/SLARM_data/datasets/waymo/training/{SCENE_ID}/images/*_0.jpg'  # _0 indicates front camera
image_paths = glob(data_path)
image_paths.sort()

saved_feat_path = f'xxx/SLARM_data/datasets/waymo/training/{SCENE_ID}/features/lseg'

pretrained_path = "/home/tangzerui/luotongan/playground/lang-seg/lseg_model_pretrained.pth"
scratch_path = "/home/tangzerui/luotongan/playground/lang-seg/lseg_model_scratch.pth"
feat_extractor = LSegFeatureExtractor(pretrained_path, scratch_path)


image_path = image_paths[IMG_ID]

image = Image.open(image_path).convert("RGB")
img_to_extract_feat_transformation = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=MEAN, std=STD),  # larger deviation from offline features
    ]
)
img_tensor = img_to_extract_feat_transformation(image).cuda()
img_tensor = img_tensor.unsqueeze(0)  # torch.Size([1, 3, 1280, 1920])

# Ensure using original image
assert img_tensor.shape[-2] == IMG_H and img_tensor.shape[-1] == IMG_W

with torch.no_grad():
    feat = feat_extractor.extract_lseg_feat(img_tensor, (FEAT_H, FEAT_W))


# Visualization verification
_, c, h, w = feat.shape
feat_to_vis = feat.permute(0,2,3,1).reshape(-1, c)
pca = PCA(n_components=3)  # Visualize dimensionality reduction to 3
feat_to_vis = pca.fit_transform(feat_to_vis.cpu())
feat_to_vis = (feat_to_vis - feat_to_vis.min()) / (feat_to_vis.max() - feat_to_vis.min())  # normalization
feat_to_vis = feat_to_vis.reshape(h, w, 3)
feat_img = Image.fromarray(np.uint8(255 * feat_to_vis)).convert("RGBA")
feat_img.save("feat_img.png")

# Compare with offline extracted features for verification
feat_name = image_path.split('/')[-1].strip().replace('jpg', 'npz')
saved_feat_path = os.path.join(saved_feat_path, feat_name)
feat_offline = torch.from_numpy(np.load(saved_feat_path)['x'])
feat = feat.squeeze(0).cpu()  # torch.Size([1, 512, 168, 252]) -> torch.Size([512, 168, 252])
if torch.allclose(feat, feat_offline, atol=1e-6, rtol=1e-6):
    print('Online feature is the same as offline feature.')
else:
    print('Online feature is different from offline feature.')
    feat_min = torch.min(feat).item()
    feat_offline_min = torch.min(feat_offline).item()
    feat_max = torch.max(feat).item()
    feat_offline_max = torch.max(feat_offline).item()
    print(f'The span of online feature is from {feat_min:.3f} to {feat_max:.3f}.')
    print(f'The span of offline feature is from {feat_offline_min:.3f} to {feat_offline_max:.3f}.')
    diff = torch.abs(feat - feat_offline)
    print(f'The maximum deviation value between online feature and offline feature is {torch.max(diff).item():.3f}.')
    diff_ratio = diff / (feat_max - feat_min + 1e-6)
    diff_ratio_max = torch.max(diff_ratio).item() * 100
    print(f'The maximum deviation ratio between online feature and offline feature is {diff_ratio_max:.3f}%.')


'''
Reference scene:
SCENE_ID = '089'
IMG_ID = 0

w.o. img norm
Online feature is different from offline feature.
The span of online feature is from -0.2282 to 0.6293.
The span of offline feature is from -0.2277 to 0.6291.
The maximum deviation value between online feature and offline feature is 0.0296.
The maximum deviation ratio between online feature and offline feature is 3.4568%.
The mean deviation ratio between online feature and offline feature is 0.1555%.

w. img norm
Online feature is different from offline feature.
The span of online feature is from -0.2268 to 0.6248.
The span of offline feature is from -0.2277 to 0.6291.
The maximum deviation value between online feature and offline feature is 0.1048.
The maximum deviation ratio between online feature and offline feature is 12.3012%.
The mean deviation ratio between online feature and offline feature is 1.0894%.
'''

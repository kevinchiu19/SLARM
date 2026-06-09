'''
LSeg feature extraction script, requires LSeg open source code setup: https://github.com/isl-org/lang-seg
Configuration process reference: https://wiki.huawei.com/domains/96125/wiki/198166/WIKI202507187553074
Place this script in the LSeg code root directory and run it
'''


import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from modules.lseg_module import LSegModule
# from additional_utils.models import LSeg_MultiEvalModule
from modules.models.lseg_blocks import forward_vit


FEAT_H, FEAT_W = 168, 252  # Saved feature height and width, corresponding to training image dimensions


def load_model():
    torch.manual_seed(1)  # seed = 1, consistent with open source demo

    # pytorch_lightning.LightningModule.load_from_checkpoint:
    # The LSegModule class used for loading must have exactly the same class structure as when the checkpoint was saved
    # (layer structure, method names, etc. cannot be modified), otherwise weight loading will fail
    model = LSegModule.load_from_checkpoint(
        checkpoint_path='checkpoints/demo_e200.ckpt',
        data_path='../datasets/',
        dataset='ade20k',
        backbone='clip_vitl16_384',
        aux=False,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=255,
        dropout=0.0,
        scale_inv=False,
        augment=False,
        no_batchnorm=False,
        widehead=True,
        widehead_hr=False,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )

    model.eval()

    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]

    # May be needed when extracting fine local features
    # # evaluator
    # scales = (
    #     [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    # )
    # model = model.cpu()
    # evaluator = LSeg_MultiEvalModule(
    #     model, scales=scales, flip=True
    # ).cuda()
    # evaluator.eval()
    # # transform for evaluator
    # transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #         transforms.Resize([360,480]),
    #     ]
    # )

    # return evaluator, transform

    return model.net  # <class 'modules.models.lseg_net.LSegNet'> same as evaluator.module.net


def extract_lseg_feat(lseg_model, x, save_path):
    layer_1, layer_2, layer_3, layer_4 = forward_vit(lseg_model.pretrained, x)

    layer_1_rn = lseg_model.scratch.layer1_rn(layer_1)
    layer_2_rn = lseg_model.scratch.layer2_rn(layer_2)
    layer_3_rn = lseg_model.scratch.layer3_rn(layer_3)
    layer_4_rn = lseg_model.scratch.layer4_rn(layer_4)

    path_4 = lseg_model.scratch.refinenet4(layer_4_rn)
    path_3 = lseg_model.scratch.refinenet3(path_4, layer_3_rn)
    path_2 = lseg_model.scratch.refinenet2(path_3, layer_2_rn)
    path_1 = lseg_model.scratch.refinenet1(path_2, layer_1_rn)

    image_features = lseg_model.scratch.head1(path_1)  # torch.Size([1, 512, 640, 960])

    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    image_features = F.interpolate(
        image_features,
        size=(FEAT_H, FEAT_W),
        mode='nearest'
        # mode='bilinear',  # Optional bilinear interpolation
        # align_corners=True  # Bilinear interpolation align corners
    )

    np.savez(save_path, x=image_features.squeeze(0).cpu().numpy())  # 512, 168, 252


lseg_model = load_model().cuda()

# use 089 scene currently
data_path = 'xxx/SLARM_data/datasets/waymo/training/089/images/*_0.jpg'  # _0 indicates front camera
image_paths = glob(data_path)
image_paths.sort()
print(len(image_paths))

save_feat_path = 'xxx/SLARM_data/datasets/waymo/training/089/features/lseg'

for image_path in tqdm(image_paths):
    image = Image.open(image_path)
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(image).cuda()
    img_tensor = img_tensor.unsqueeze(0)  # torch.Size([1, 3, 1280, 1920])
    feat_name = image_path.split('/')[-1].strip().replace('jpg', 'npz')
    save_path = os.path.join(save_feat_path, feat_name)
    with torch.no_grad():
        extract_lseg_feat(lseg_model, img_tensor, save_path)

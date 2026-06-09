'''
LSeg model conversion script, requires LSeg open source code setup: https://github.com/isl-org/lang-seg
Place this script in the LSeg code root directory and run it
Converts the open source pytorch_lightning-based model to a pytorch model and saves weights needed for LSeg feature extraction
Subsequent model loading does not require extra code or open source environment
'''


import os
import torch
from third_party.lang_seg.modules.lseg_module import LSegModule


SAVE_PATH = "ckpts/lseg"  # path to save converted model
os.makedirs(SAVE_PATH, exist_ok=True)

torch.manual_seed(1)  # seed = 1, consistent with open source demo

# pytorch_lightning.LightningModule.load_from_checkpoint:
# The LSegModule class used for loading must have exactly the same class structure as when the checkpoint was saved
# (layer structure, method names, etc. cannot be modified), otherwise weight loading will fail
model = LSegModule.load_from_checkpoint(
    checkpoint_path='ckpts/demo_e200.ckpt',
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
    weights_only=False,
)

model.eval()

model.mean = [0.5, 0.5, 0.5]
model.std = [0.5, 0.5, 0.5]

print('pytorch_lightning model loaded.')

# print(f'saving lseg_model.pretrained to {os.path.join(SAVE_PATH, "lseg_model_pretrained.pth")}.')
# torch.save(model.net.pretrained.state_dict(), os.path.join(SAVE_PATH, "lseg_model_pretrained.pth"))
# print('lseg_model.pretrained saved.')

print(f'saving lseg_model.scratch to {os.path.join(SAVE_PATH, "lseg_model_scratch.pth")}.')
torch.save(model.net.scratch.state_dict(), os.path.join(SAVE_PATH, "lseg_model_scratch.pth"))
print('lseg_model.scratch saved.')

# Convert 1x1 conv to linear for better torch.compile support
print('converting 1x1 conv to linear in lseg_model.pretrained.')
lseg_model_pretrained_state_dict = model.net.pretrained.state_dict()
lseg_model_pretrained_state_dict["act_postprocess1.1.weight"] = lseg_model_pretrained_state_dict["act_postprocess1.3.weight"].squeeze()
lseg_model_pretrained_state_dict["act_postprocess1.1.bias"] = lseg_model_pretrained_state_dict["act_postprocess1.3.bias"]
lseg_model_pretrained_state_dict["act_postprocess2.1.weight"] = lseg_model_pretrained_state_dict["act_postprocess2.3.weight"].squeeze()
lseg_model_pretrained_state_dict["act_postprocess2.1.bias"] = lseg_model_pretrained_state_dict["act_postprocess2.3.bias"]
lseg_model_pretrained_state_dict["act_postprocess3.1.weight"] = lseg_model_pretrained_state_dict["act_postprocess3.3.weight"].squeeze()
del lseg_model_pretrained_state_dict["act_postprocess3.3.weight"]
lseg_model_pretrained_state_dict["act_postprocess3.1.bias"] = lseg_model_pretrained_state_dict["act_postprocess3.3.bias"]
del lseg_model_pretrained_state_dict["act_postprocess3.3.bias"]
lseg_model_pretrained_state_dict["act_postprocess4.1.weight"] = lseg_model_pretrained_state_dict["act_postprocess4.3.weight"].squeeze()
lseg_model_pretrained_state_dict["act_postprocess4.1.bias"] = lseg_model_pretrained_state_dict["act_postprocess4.3.bias"]

lseg_model_pretrained_state_dict["act_postprocess1.3.weight"] = lseg_model_pretrained_state_dict["act_postprocess1.4.weight"]
del lseg_model_pretrained_state_dict["act_postprocess1.4.weight"]
lseg_model_pretrained_state_dict["act_postprocess1.3.bias"] = lseg_model_pretrained_state_dict["act_postprocess1.4.bias"]
del lseg_model_pretrained_state_dict["act_postprocess1.4.bias"]
lseg_model_pretrained_state_dict["act_postprocess2.3.weight"] = lseg_model_pretrained_state_dict["act_postprocess2.4.weight"]
del lseg_model_pretrained_state_dict["act_postprocess2.4.weight"]
lseg_model_pretrained_state_dict["act_postprocess2.3.bias"] = lseg_model_pretrained_state_dict["act_postprocess2.4.bias"]
del lseg_model_pretrained_state_dict["act_postprocess2.4.bias"]
lseg_model_pretrained_state_dict["act_postprocess4.3.weight"] = lseg_model_pretrained_state_dict["act_postprocess4.4.weight"]
del lseg_model_pretrained_state_dict["act_postprocess4.4.weight"]
lseg_model_pretrained_state_dict["act_postprocess4.3.bias"] = lseg_model_pretrained_state_dict["act_postprocess4.4.bias"]
del lseg_model_pretrained_state_dict["act_postprocess4.4.bias"]

print(f'saving converted lseg_model.pretrained to {os.path.join(SAVE_PATH, "lseg_model_pretrained_replace_1x1conv_with_linear.pth")}.')
torch.save(lseg_model_pretrained_state_dict, os.path.join(SAVE_PATH, "lseg_model_pretrained_replace_1x1conv_with_linear.pth"))
print('converted lseg_model.pretrained saved.')

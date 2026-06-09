import torch
import os
import clip  # install using 'pip install git+https://github.com/openai/CLIP.git'

from src.dataset.constants import SEMANTIC_LABEL_LIST
from functools import lru_cache


@lru_cache(maxsize=8)
def _load_text_features_cached(semantic_label_list, device_str):
    """  Get semantic class text embeddings  """
    device = torch.device(device_str)
    dtype = torch.float16 if os.environ.get('DISABLE_BFLOAT') else torch.bfloat16
    clip_pretrained, _ = clip.load("ViT-B/32", device=device, jit=False)
    text_label_inputs = clip.tokenize(semantic_label_list)
    text_label_inputs = text_label_inputs.cuda()
    with torch.autocast(device.type, dtype=dtype), torch.no_grad():
        text_label_feats = clip_pretrained.encode_text(text_label_inputs)
        text_label_feats = text_label_feats / text_label_feats.norm(dim=-1, keepdim=True)
    return text_label_feats.float().detach()


def get_text_label_feats(semantic_label_list):
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device_str = f"cuda:{current_device_id}"
    else:
        device_str = "cpu"
    return _load_text_features_cached(tuple(semantic_label_list), device_str)


def feat2class(feats, text_label_feats, similarity_probs_threshold=0.2, return_probs=False):
    similarity = feats @ text_label_feats.T / 0.07  # 0.07 is temperature aligned with LSeg
    probs = torch.softmax(similarity, dim=-1)
    if similarity_probs_threshold > 0.0:
        for i in range(1, probs.shape[-1]):  # NOTE: 'others' in class idx 0
            p = probs[:, i]
            p[p < similarity_probs_threshold] = 0.0
    if return_probs:
        return probs
    semantic = torch.argmax(probs, dim=-1)
    semantic = semantic.long()
    return semantic


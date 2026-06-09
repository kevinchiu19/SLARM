import logging
import os
import copy

import torch
from torch.nn.functional import cosine_similarity
import torch.backends.cudnn as cudnn
from einops import rearrange
from PIL import Image
import numpy as np

from engine_tools import build_model
import src.models as models
import src.utils.misc as misc
from src.dataset.constants import DATASET_DICT, FEAT_TYPES
from src.dataset.data_utils import prepare_inputs_and_targets, to_batch_tensor
from src.dataset.datasets import PerceptualModelDataset
from src.utils.logging import setup_logging
from src.visualization.video_maker import make_video
from src.dataset.constants import SEMANTIC_ID_TO_COLOR
from tools.lseg_feat_extractor import LSegFeatureExtractor
from main_slarm import get_args_parser
if os.getenv("FEAT_DIST"):
    from tools.feats_tools import get_text_label_feats, feat2class


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True


def forward_renderer(model, gs_params, data_dict):
    render_results = model.forward_renderer(gs_params, data_dict)
    images, opacities = render_results["rendered_image"], render_results["rendered_alpha"]
    if model.use_sky_token:
        target_ray_dict = model.plucker_embedder(
            data_dict["target_intrinsics"],
            data_dict["target_camtoworlds"],
            image_size=(data_dict["height"], data_dict["width"]),
        )

        sky_token = gs_params["sky_token"]
        sky = model.sky_head(target_ray_dict["dirs"], sky_token)
        images = images + (1 - opacities[..., None]) * sky

    if model.use_affine_token:
        # apply linear and translation
        images = torch.einsum('btvhwi,bvij->btvhwj', images, gs_params['affine']['linear'].float()) + gs_params['affine']['translation'].float()

    render_results["rendered_image"] = images
    render_results = model.forward_decoder(render_results)

    return render_results


def main(args):

    global logger
    args.exp_name = args.model.replace("/", "-") if args.exp_name is None else args.exp_name
    log_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    video_dir = os.path.join(log_dir, "videos")
    args.log_dir, args.video_dir = log_dir, video_dir

    [os.makedirs(d, exist_ok=True) for d in [log_dir, video_dir]]

    device = torch.device(args.device)
    misc.fix_random_seeds(args.seed)

    dtype = torch.float16 if os.environ.get('DISABLE_BFLOAT') else torch.bfloat16

    # set up logging
    setup_logging(output=log_dir, level=logging.INFO)
    logger = logging.getLogger("SLARM")
    logger.info(f"hostname: {os.uname().nodename}\n")
    logger.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    logger.info(f"Logging to {log_dir}")

    # # get feature types
    # feat_types = args.load_feat_types.split(',')
    # feat_types = [feat_type.strip() for feat_type in feat_types]
    # assert all(feat_type in FEAT_TYPES for feat_type in feat_types), \
    #     f"The feat types {feat_types} should be in {FEAT_TYPES}. "

    # build dataset
    dataset_meta = DATASET_DICT[args.dataset[0]]
    dataset_annotation = dataset_meta["annotation_txt_file_train"]
    assert dataset_annotation is not None
    if args.dataset == "nuscenes":
        dataset_annotation = f"data/dataset_scene_list/nuscenes_train.txt"
    else:
        dataset_annotation = f"{args.data_root}/{dataset_annotation}"

    dataset_vis = PerceptualModelDataset(
        data_root=args.data_root,
        annotation_txt_file_list=dataset_annotation,
        target_size=args.input_size,
        num_context_timesteps=args.num_context_timesteps,
        num_target_timesteps=args.num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=args.load_depth,
        load_flow=args.load_flow,
        # load_feat_types=args.load_feat_types.split(',') if args.load_feat_types else [],
        online_feat=args.online_feat,
        img_norm_for_online_feat=args.img_norm_for_online_feat,
        skip_sky_mask=args.skip_sky_mask,
    )

    logger.info(f"Dataset: {args.dataset}, annotation: {dataset_annotation}")
    logger.info(f"Dataset contains {len(dataset_vis):,} sequences using {dataset_annotation}.")

    # build model
    assert args.model == 'slarm'
    model = build_model(args)

    logger.info(f"Model = {str(model)}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Parameters: {n_params / 1e6:.2f}M ({n_params:,})")
    model.to(device)
    misc.load_model(args, model)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Trainable Parameters: {num_trainable_params / 1e6:.2f}M")
    model.eval()

    # Load feature extractor
    # if args.online_feat and args.load_feat_types is not None:
    if args.online_feat:
        logger.info("Using online feature, loading feature extractor.")
        feat_extractor = LSegFeatureExtractor(args.lseg_model_pretrained_path, args.lseg_model_scratch_path, dtype=torch.float16 if os.environ.get('DISABLE_BFLOAT') else torch.bfloat16)
        logger.info("Feature extractor loaded.")
    else:
        feat_extractor = None

    # use specific sample
    data_dict = dataset_vis.__getitem__(args.scene_id, args.sample_index, return_all=True)
    data_dict = to_batch_tensor(data_dict)
    if not isinstance(data_dict['num_max_cams'], int):
        num_max_cams = int(data_dict['num_max_cams'][0])
    else:
        num_max_cams = data_dict['num_max_cams']
    input_dict, target_dict = prepare_inputs_and_targets(data_dict, device, v=num_max_cams, feat_extractor=feat_extractor)

    # get query text
    text_list = args.text_list.split(',')
    # NOTE: in feat2class, the filtering order requires 'others' to be first.
    text_list = ['others'] + [text.strip() for text in text_list]
    assert len(text_list) > 0

    # extract text embedding for different type of features
    # text_feats_all = []
    # for feat_type in feat_types:
    #     if feat_type == 'pe3r':
    #         siglip = AutoModel.from_pretrained("text_query_visualize/google/siglip-large-patch16-256", device_map=device)
    #         siglip_tokenizer = AutoTokenizer.from_pretrained("text_query_visualize/google/siglip-large-patch16-256")
    #         text_inputs = siglip_tokenizer(text=text_list, padding="max_length", return_tensors="pt")
    #         text_inputs = {key: value.to("cuda") for key, value in text_inputs.items()}
    #         with torch.autocast("cuda", dtype=dtype):
    #             with torch.no_grad():
    #                 text_feats =siglip.get_text_features(**text_inputs)
    #                 text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    #     elif feat_type == 'lseg':
    #         clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
    #         text_inputs = clip.tokenize(text_list)
    #         text_inputs = text_inputs.cuda()
    #         with torch.autocast("cuda", dtype=dtype):
    #             with torch.no_grad():
    #                 text_feats = clip_pretrained.encode_text(text_inputs)
    #                 text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    #     else:
    #         raise ValueError
    #     text_feats_all.append(text_feats)
    text_feats = get_text_label_feats(text_list)
    with torch.autocast("cuda", dtype=dtype):
        with torch.no_grad():
            pred_dict = model(input_dict)

    if os.getenv("CONTEXT_FEAT"):
        context_idx = list(range(len(input_dict["context_frame_idx"][0])))
    else:
        start_frame_idx = int(target_dict["target_frame_idx"][0][0])
        context_idx = [int(idx) - start_frame_idx for idx in input_dict["context_frame_idx"][0]]

    # target feat for debug
    # target_feat = target_dict["target_feat"]
    # target_feat = rearrange(target_feat, "b t v c h w -> b t v h w c")
    rendered_feat = pred_dict["render_results"]["rendered_feat"]

    if args.text_query_mode == "activate_one":
        for j in range(len(text_feats)):
            # calculate similarity on CPU to avoid OOM
            # similarity_map = cosine_similarity(target_feat[:, context_idx, :, :, :].to("cpu"),
            #                                    text_feats[j].to("cpu"), dim=-1)  # target_feat for debug
            similarity_map = cosine_similarity(rendered_feat[:, context_idx, :, :, :].to("cpu"),
                                            text_feats[j].to("cpu"), dim=-1)

            # Normalize Similarities
            min_sim, max_sim = similarity_map.min(), similarity_map.max()
            similarity_map = (similarity_map - min_sim) / (max_sim - min_sim)
            # response over threshold
            threshold = args.similarity_probs_threshold
            response_mask = similarity_map >= threshold
            gs_params = copy.deepcopy(pred_dict["gs_params"])

            # high light using red color
            # TODO: Compute similarity using rendered semantic results and project back to RGB
            gs_params["colors"][..., 0][response_mask] = 0.99

            # high light using high light
            # gs_params["colors"] /= 4.0
            # gs_params["colors"][response_mask] *= 4.0

            # change rendered image
            render_results = forward_renderer(model, gs_params, input_dict)
            pred_dict["render_results"]["rendered_image"] = render_results["rendered_image"]

            # make video
            out_path = f"{args.video_dir}/sample{args.sample_index}-{text_list[j]}.mp4"
            _ = make_video(
                    output_filename=out_path,
                    skip_plot_gt_depth_and_flow=False,
                    data_dict=data_dict,
                    input_dict=input_dict,
                    target_dict=target_dict,
                    pred_dict=pred_dict
                )
    elif args.text_query_mode == "segment_all":
        key_feats = rendered_feat[:, context_idx, :, :, :, :]
        b, t, v, h, w, _ = key_feats.shape
        key_feats = rearrange(key_feats, "b t v h w c -> (b t v h w) c")
        text_feats = text_feats.to(dtype)

        pred_semantic = feat2class(key_feats, text_feats, args.similarity_probs_threshold)
        # context should be in 0, 5, 10, 15
        pred_semantic = rearrange(pred_semantic, "(b t v h w) -> (b t v) h w ", b=b, t=t, v=v, h=h, w=w)

        rendered_image = pred_dict["render_results"]["rendered_image"]
        rendered_image = rearrange(rendered_image, "b t v h w c -> (b t v) h w c")

        # MEAN = [0.5, 0.5, 0.5]
        # STD = [0.5, 0.5, 0.5]
        # mean = torch.tensor([[MEAN]], device=rendered_image.device)
        # std = torch.tensor([[STD]], device=rendered_image.device)

        # rendered_image = (rendered_image * std + mean).clamp(0.0, 1.0)  # NOTE: RGB normalization
        rendered_image =  rendered_image.float().cpu().detach().numpy()
        pred_semantic = pred_semantic.cpu().detach().numpy()

        img_idx = 0  # view idx and frame idx
        img = rendered_image[img_idx].clip(0, 1) * 255
        pred = pred_semantic[img_idx]

        save_img = Image.fromarray(np.uint8(img)).convert("RGBA")
        save_img.save(f"input_{args.sample_index}.png")

        for class_idx in range(len(SEMANTIC_ID_TO_COLOR)):
            img[pred == class_idx] = np.array(SEMANTIC_ID_TO_COLOR[class_idx])

        rst_img = Image.fromarray(np.uint8(img)).convert("RGBA")
        rst_img.save(f"result_{args.sample_index}_{img_idx}.png")
        print('done.')
    else:
        raise ValueError


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--text_list", type=str, default='vehicle,bicycle,motorcycle,people,buildings,road,sidewalk,sky',
                        help="The input text query, input in comma-separated form")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument(
        "--text_query_mode",
        type=str,
        choices=["activate_one", "segment_all"],
        default="segment_all",
    )
    args = parser.parse_args()
    main(args)

#!/bin/bash


# Select platform
export DEVICE_TYPE="NPU"  # GPU or NPU

if [ "$DEVICE_TYPE" = "GPU" ]; then
    # run single-GPU inference demo
    export CUDA_VISIBLE_DEVICES=5  # 0,1,2,4,5,6,7
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    export NCCL_DEBUG=INFO
    export NCCL_P2P_DISABLE=0
    export NCCL_P2P_LEVEL=NVL
fi

if [ "$DEVICE_TYPE" = "NPU" ]; then
    export DEVICE_TYPE="NPU"
    export ASCEND_RT_VISIBLE_DEVICES=13  # 0,1,2,4,5,6,7
    # NPU performance optimization
    export AVOID_AI_CPU=1  # Avoid generating AI_CPU Sin and Cos operators due to double data type.
    export USE_EQUAL_CROSS=1  # Equivalent replacement for torch.cross
    export TASK_QUEUE_ENABLE=2  # Speed ​​up host distribution
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  # Start virtual memory to save device memory
    export CONTEXT_FEAT=1  # No feature is rendered; feature is only supervised from the input viewpoint.
    export RENDER_OP_VERSION=1212  # Rendering operator version
    # export RENDER_PROJ_FWD_USE_FUSED_KERNEL=1  # Rendering projection uses the blending operator for forward propagation, and annotations use the small operator.

    # NPU Debugging and Logging
    # export ASCEND_SLOG_PRINT_TO_STDOUT=1
    # export ASCEND_GLOBAL_LOG_LEVEL=3
    # export ASCEND_LAUNCH_BLOCKING=1  # for debug

    # Replace the rendering preprocessing script in the container environment
    bash replace_meta_gauss_render.sh ${CONDA_DEFAULT_ENV} ${RENDER_OP_VERSION}
fi

# Set model configuration
export FEAT_DIST=1
export DATASET=waymo
export DATA_ROOT=data/SLARM_data
export OVERFIT_EXP=1
export SCENE_ID_WAYMO=80
# export PROFILING=1  # Printing takes time

export MASTER_PORT=16818
export DEVICE_NUM=1
export BS_PER_DEVICE=1
export PROJECT=slarm
export EXP_NAME=exp_0528
# export CKPT_PTH=xxx.pth


# python -m debugpy --listen 13688 --wait-for-client main_slarm.py \
torchrun --nproc_per_node=$DEVICE_NUM --master_port ${MASTER_PORT} main_slarm.py \
    --project ${PROJECT} \
    --exp_name ${EXP_NAME} \
    --dataset ${DATASET} \
    --data_root $DATA_ROOT \
    --batch_size $BS_PER_DEVICE --num_iterations 200000 --lr_sched constant \
    --vis_every_n_iters 500 \
    --eval_every_n_iters 10000 --keep_n_ckpts 30 --ckpt_every_n_iters 10000 \
    --enable_tensorboard \
    --model slarm \
    --load_depth --load_flow --load_ground \
    --load_semantic_label \
    --num_max_cameras 3 --use_affine_token \
    --sigmoid_rgb \
    --num_motion_tokens 0 \
    --use_sky_token \
    --embed_dim 768 --depth 12 --patch_embed conv --patch_size 8 \
    --use_ms3_motion \
    --use_last_token \
    --shortcut_rgb \
    --add_patch_plucker_embed \
    --enable_depth_loss --enable_sky_opacity_loss \
    --enable_flow_reg_loss --flow_reg_coeff 0.005 \
    --enable_perceptual_loss --perceptual_weight 0.05 --perceptual_loss_start_iter 5000 \
    --rgb_loss_coeff 1.0 \
    --similarity_probs_threshold 0.2 \
    --online_feat --img_norm_for_online_feat \
    $( [ "$DEVICE_TYPE" = "GPU" ] && echo "--enable_feat_loss" ) \
    $( [ "$DEVICE_TYPE" = "NPU" ] && echo "--enable_context_feat_loss" ) \
    --feat_loss_coeff 1.0 --feat_loss_type mse \
    --lseg_model_scratch_path ckpts/lseg/lseg_model_scratch.pth --lseg_model_pretrained_path ckpts/lseg/lseg_model_pretrained_replace_1x1conv_with_linear.pth \
    --auto_resume \
    # --load_from $CKPT_PTH \
    # --evaluate \

#!/bin/bash


DEVICE_TYPE="NPU"

WORKDIR="xxx"  # To be modified

# 环境变量设置
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT=29512

# 打印出参数的值
echo "NNODES: ${MA_NUM_HOSTS}"
echo "MASTER_HOST: ${VC_WORKER_HOSTS}"
echo "NODE_RANK: ${VC_TASK_INDEX}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NGPUS_PER_NODE: ${MA_NUM_GPUS}"
echo "NUM_PROCESSES: $((${MA_NUM_GPUS} * ${MA_NUM_HOSTS}))"

CONDA_ENV="xxx"  # To be modified
RENDER_OP_VERSION=1212

DATASET=waymo
PROJECT=slarm
EXP_NAME=exp_0527
DATA_ROOT=data/SLARM_data
BS_PER_DEVICE=1


sudo -i bash -i -c "
    export HCCL_CONNECT_TIMEOUT=6000 && \
    cd ${WORKDIR} && \
    conda activate ${CONDA_ENV} && \
    export AVOID_AI_CPU=1 && \
    export USE_EQUAL_CROSS=1 && \
    export TASK_QUEUE_ENABLE=2 && \
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True && \
    export FEAT_DIST=1 && \
    export CONTEXT_FEAT=1 && \
    export RENDER_OP_VERSION=${RENDER_OP_VERSION} && \
    bash replace_meta_gauss_render.sh ${CONDA_ENV} ${RENDER_OP_VERSION} && \
    bash cp_clip_weight.sh && \
    torchrun --nnodes=${MA_NUM_HOSTS} --nproc_per_node=${MA_NUM_GPUS} --node_rank=${VC_TASK_INDEX} \
        --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} main_slarm.py \
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
        --disable_grad_checkpointing > ${EXP_NAME}.txt
"

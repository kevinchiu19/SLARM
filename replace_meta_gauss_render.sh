#!/bin/bash

CONDA_ENV=$1
RENDER_OP_VERSION=$2

SOURCE_FILE="src/utils/projection_three_dims_gaussian_fused_$RENDER_OP_VERSION.py"
TARGET_FILE="/home/ma-user/anaconda3/envs/$CONDA_ENV/lib/python3.10/site-packages/meta_gauss_render/ops/projection_three_dims_gaussian_fused.py"


cp -f "$SOURCE_FILE" "$TARGET_FILE"

# 验证是否执行成功
if [ $? -eq 0 ]; then
    echo "成功：已用 $SOURCE_FILE 替换 $TARGET_FILE"
else
    echo "失败：替换操作执行出错"
    exit 1
fi
#!/bin/bash


SOURCE_FILE="xxx/ckpts/ViT-B-32.pt"
TARGET_DIR="/root/.cache/clip/"

if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"  # -p 递归创建，忽略已存在目录
    echo "多级目录不存在，已成功创建：$TARGET_DIR"
else
    echo "多级目录已存在，无需创建：$TARGET_DIR"
fi

cp -f "$SOURCE_FILE" "$TARGET_DIR"

# 验证是否执行成功
if [ $? -eq 0 ]; then
    echo "成功：已将 $SOURCE_FILE 复制到 $TARGET_DIR"
else
    echo "失败：替换操作执行出错"
    exit 1
fi

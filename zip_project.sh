#!/bin/bash

# 获取当前目录的父目录
cd "$(pwd)/.."

# 使用 rsync 排除不需要的文件，并创建 zip 文件
rsync -av --exclude='.git' --exclude='ckpts' --exclude='data' --exclude='path' --exclude='val' --exclude='maps' --exclude='bevformer_results.json' --exclude='output_results.json' --exclude='model_structure.txt' --exclude='test' --exclude='bug_fix' ./ ./temp_dir/

# 创建 zip 文件
zip -r FreeADProject.zip ./temp_dir/*

# 删除临时目录
rm -rf ./temp_dir

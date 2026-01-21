#!/bin/bash
# SafeEdit 安全性评估测试脚本

set -e

# 激活环境
eval "$(conda shell.bash hook)"
conda deactivate
conda activate SD

# 配置参数
MODEL_PATH="/mnt/sda/zjk/models/qwen3-4b"
CLASSIFIER_PATH="/mnt/sda/zjk/Second/DSCD/SafeEdit-Safety-Classifier"
DATA_PATH="/mnt/sda/zjk/Second/verl/data/safety_roleplay/train.json"
OUTPUT_DIR="/mnt/sda/zjk/Second/verl/safety_eval_results"
NUM_SAMPLES=100  # 评估样本数量，可以根据需要调整

# 运行评估
python3 test_safety_evaluation.py \
    --model_path ${MODEL_PATH} \
    --classifier_path ${CLASSIFIER_PATH} \
    --data_path ${DATA_PATH} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --device cuda

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================"

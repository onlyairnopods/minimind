#!/usr/bin/env bash
set -e
set -o xtrace

export TOKENIZERS_PARALLELISM=true
export HF_ENDPOINT='https://hf-mirror.com'
export CUDA_VISIBLE_DEVICES=7,3

SAVE_DIR="../out/MiniMind2"

mkdir -p $SAVE_DIR

torchrun --standalone --nproc_per_node=2 \
    train_grpo.py \
    --save_dir $SAVE_DIR \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate "8e-8" \
    --accumulation_steps 8 \
    --hidden_size 768 \
    --num_hidden_layers 16 \
    --max_seq_len 66 \
    --max_gen_len 1536 \
    --data_path "../dataset/minimind_dataset/rlaif-mini.jsonl" \
    --num_generations 8 \
    --beta 0.02 \
    --reasoning 0 \
    --reward_model_path "internlm/internlm2-1_8b-reward" \
    --use_wandb \
    --wandb_project "MiniMind-GRPO" \
    --use_compile 1 \
    2>&1 | tee -a $SAVE_DIR/train_grpo.log
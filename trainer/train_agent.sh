#!/usr/bin/env bash
set -e
set -o xtrace

export TOKENIZERS_PARALLELISM=true
export HF_ENDPOINT='https://hf-mirror.com'
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6}

TRAIN_GPUS=${TRAIN_GPUS:-2}
ROLLOUT_DEVICES=${ROLLOUT_DEVICES:-cuda:2}
REWARD_DEVICES=${REWARD_DEVICES:-cuda:2}
REF_DEVICES=${REF_DEVICES:-cuda:2}

SAVE_DIR="../out/MiniMind3"

mkdir -p $SAVE_DIR

torchrun --standalone --nproc_per_node=$TRAIN_GPUS \
    train_agent.py \
    --save_dir $SAVE_DIR \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate "3e-7" \
    --accumulation_steps 8 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_seq_len 1024 \
    --max_gen_len 1024 \
    --data_path "../../minimind_tmp/minimind_dataset/agent_rl.jsonl" \
    --num_generations 4 \
    --beta 0.02 \
    --loss_type "cispo" \
    --use_wandb \
    --wandb_project "MiniMind-Agent-RL" \
    --use_compile 1 \
    --debug_mode --debug_interval 100 \
    --thinking_ratio 0.3 \
    --reward_model_path "internlm/internlm2-1_8b-reward" \
    --rollout_device "$ROLLOUT_DEVICES" \
    --reward_device "$REWARD_DEVICES" \
    --ref_device "$REF_DEVICES" \
    --from_resume 1 \
    2>&1 | tee -a $SAVE_DIR/train_agent.log

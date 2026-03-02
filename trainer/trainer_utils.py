"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    """
    计算模型参数
    Total Params：模型占用的总显存参数。
    Active Params：推理时实际参与计算的参数（针对 MoE 模型，只统计被激活的专家参数）。
    """
    # 模型占用的总显存参数
    total = sum(p.numel() for p in model.parameters()) / 1e6 # 除以 1e6 转换为“百万(M)”单位
    
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0)) # 总共有多少个专家 (Routed Experts)，例如 64 个
    n_active = getattr(config, 'num_experts_per_tok', 0) # 每个 Token 实际激活选用的专家数 (Active Experts)，例如每次选 2 个
    n_shared = getattr(config, 'n_shared_experts', 0) # 共享专家数量 (Shared Experts)，这是 DeepSeek-MoE 等架构特有的，这些专家总是被激活
    
    # 计算单个专家的参数量
    # 通过筛选参数名中包含 'mlp.experts.0.' 的项，只统计“第0号专家”的大小
    # 假设所有专家的结构是一样的，算出一个就能代表所有
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6

    # 同理，计算单个“共享专家”的大小
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    
    # 计算“骨架”参数 (Base Params)
    # 骨架参数 = 总参数 - (单个路由专家参数 × 总数) - (单个共享专家参数 × 总数)
    # 这部分包含：Embedding, Attention, RMSNorm, OutputHead 等所有非 MLP 的公共部分
    base = total - (expert * n_routed) - (shared_expert * n_shared)

    # 计算激活参数 (Active Params)
    # 激活参数 = 骨架 + (单个路由专家参数 × 激活数) + (单个共享专家参数 × 总数)
    # 推理时，Token 会经过骨架部分，经过所有共享专家，但在路由专家中只走 n_active 条路
    active = base + (expert * n_active) + (shared_expert * n_shared)

    # 如果激活参数小于总参数，说明这是一个 MoE 模型
    if active < total:
        # 输出格式例如：Model Params: 1000.00M-A200.00M (总参数10亿，实际激活2亿)
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: 
        # 如果是 Dense 模型（稠密模型），active == total
        Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    # 余弦退火
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth' # 仅保存模型权重
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth' # 保存完整训练状态（模型+优化器+进度，用于断点续训）

    if model is not None:
        # 如果是 DDP 模型，取 .module；如果是 torch.compile 后的模型，取 ._orig_mod
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)

        state_dict = raw_model.state_dict()
        # 转为半精度 (half) 并移至 CPU，节省磁盘空间且不占用显存
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 安全保存模型权重 (使用 .tmp 中转防止保存过程中途崩溃导致文件损坏)
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 获取 WandB 的 run_id，确保重启训练后日志能接上
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        # 处理额外的需要保存的对象（如学习率调度器 scheduler）
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        # 显存清理
        del state_dict, resume_data
        torch.cuda.empty_cache()

    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')

            # [核心逻辑] 处理 GPU 数量变化后的 Step 转换
            # 例如：之前用 2 张卡跑了 100 step，现在换成 4 张卡，
            # 为了保持数据消耗量一致，step 需要调整（100 * 2 // 4 = 50）
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    断点续训采样器
    场景：假设你在第 3 个 Epoch 的第 1000 个 Step 训练中断了。
    作用：当你重启训练时，DataLoader 通常会从头开始加载数据。
    这个自定义采样器会跳过前 1000 个 Batch 的数据索引，
    直接从第 1001 个 Batch 开始提供数据，确保模型不重复训练已见过的数据。
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches # 需要跳过的 Batch 数量（通常从 checkpoint 中读取）

    def __iter__(self):
        batch = []       # 用于暂存当前的索引
        skipped = 0      # 记录已经跳过的 Batch 计数器

        # 遍历基础采样器产生的所有样本索引
        for idx in self.sampler:
            batch.append(idx)

            # 当收集的索引数量达到一个 Batch 大小时
            if len(batch) == self.batch_size:
                # 检查是否还需要继续跳过
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = [] # 清空当前 Batch，但不返回给 DataLoader
                    continue
                
                # 如果已经跳够了，则产出该 Batch
                yield batch
                batch = [] # 重置 Batch 准备收集下一个

        # [处理尾部数据]
        # 如果样本总数不能被 batch_size 整除，处理最后剩下的不足一个 Batch 的样本
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """
        计算跳过之后，剩余的总 Batch 数量
        """
        # 1. 计算原始数据总共能分成多少个 Batch (向上取整)
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        # 2. 返回剩余的 Batch 数量，确保不为负数
        return max(0, total_batches - self.skip_batches)
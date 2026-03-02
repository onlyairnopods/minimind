import os
import sys

# 设置包名为 trainer，方便内部引用
__package__ = "trainer"
# 将上级目录加入 sys.path，确保能 import model 和 dataset 文件夹下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    执行一个 Epoch 的训练流程。
    Args:
        epoch: 当前 Epoch 的索引。
        loader: 数据加载器 (DataLoader)。
        iters: 一个 Epoch 中包含的总 Iteration 数（步数）。
        start_step: 恢复训练时的起始步数。
        wandb: Weights & Biases 日志记录器对象。
    """
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 根据当前的总步数 (epoch * iters + step) 计算当前学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 手动更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward() # scaler 用于处理 FP16 下的梯度下溢问题

        # 梯度更新 (Optimizer Step) - 仅在达到累积步数时执行
        if (step + 1) % args.accumulation_steps == 0:
            # 先将梯度 unscale (反缩放) 回 FP32，以便进行梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新模型参数
            scaler.step(optimizer)
            # 更新 scaler 的缩放因子 (scale factor)
            scaler.update()

            # 清空梯度
            # set_to_none=True 比默认的 zero_grad() 更高效，因为它直接将梯度设为 None 而不是 0 张量
            optimizer.zero_grad(set_to_none=True)

        # 每隔 log_interval 步 或 在最后一步时记录日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time

            # 还原真实的 Loss 数值用于显示 (乘以累积步数)
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            
            current_lr = optimizer.param_groups[-1]['lr']

            # 计算 ETA (预计剩余时间)
            # 公式: (已用时间 / 当前步数) * 总步数 / 60 - 已用时间 / 60
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 模型保存 (Checkpointing)
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整的训练状态 (包含 optimizer, scaler, epoch 等)，用于断点续训 (Resume)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 显式删除当前步的变量，防止引用计数导致显存无法释放
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 检查是否有断点续训的需求（args.from_resume==1）
    # 如果有，尝试加载之前的 Checkpoint 数据（包含模型权重、优化器状态、step等）
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 创建自动混合精度（AMP）上下文管理器
    # 这将在后续的 forward 过程中自动将部分运算转为半精度以加速并节省显存
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 定义模型、数据、优化器 ==========
    # 初始化模型和分词器，如果指定了 args.from_weight，这里会加载预训练权重
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 定义采样器：如果是分布式训练，必须使用 DistributedSampler 来确保每张卡分到不同的数据
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 定义梯度缩放器（Scaler），用于 float16 训练防止梯度下溢（bfloat16 通常不需要，但这里为了兼容写上了）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 5. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 如果加载了 checkpoint，这里会将模型、优化器、Scaler 全部恢复到上次断掉的状态
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 6. DDP包模型 ==========
    if dist.is_initialized():
        # 忽略旋转位置编码（RoPE）的预计算缓存，因为它们是常量，不需要同步梯度
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 7. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 尝试从 Checkpoint 中恢复之前的 run_id，这样图表能接上
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        # 定义本次运行的名称，包含关键超参数
        if lm_config.use_moe:
            wandb_run_name = f"MiniMind-Pretrain-MoE-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        else:
            wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, logdir=args.save_dir, id=wandb_id, resume=resume)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # DDP 关键步骤：每个 epoch 设置不同的随机种子，确保数据 shuffle 顺序不同
        train_sampler and train_sampler.set_epoch(epoch)

        # 再次设置 Python 随机种子（为了后续 indices 生成），并生成随机索引
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()

        # 计算需要跳过的步数：只有在“恢复训练的那个 epoch”才需要跳过之前的 step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0

        # 使用自定义的 SkipBatchSampler，实现精确到 step 的断点续训
        # 它会快速空转跳过前 skip 个 batch，直接从断点处开始产出数据
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)

        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
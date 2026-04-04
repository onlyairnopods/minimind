"""
这个其实不是标准的PPO，而是MiniMind的简化版本
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')


# 自定义的Critic模型，继承自MiniMindLM
class CriticModel(MiniMindForCausalLM):
    # 其作用是评估当前生成状态（State）的价值 V(s)
    def __init__(self, params):
        super().__init__(params)
        # 将原有的语言模型输出头（lm_head，输出词表大小）替换为一个线性层
        # 该线性层将隐藏状态映射为单一的标量值（即该状态的价值）
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 1. 前向传播：使用基础的 Transformer 模型获取所有 token 的隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # outputs: hidden_states, past_key_values, aux_loss
        
        # 获取最后一层的隐藏状态并进行层归一化处理
        hidden_states = self.model.norm(outputs[0])
        # outputs[0] 获取到的是整个序列的隐藏状态，它的维度是 (Batch_size, SeqLen, Hidden_size)
        # 经过 self.model.norm 归一化后，维度保持不变
        
        # 2. 价值预测：将隐藏状态输入到 value_head 得到价值，并去掉最后一个维度 (B, SeqLen, 1) -> (B, SeqLen)
        values = self.value_head(hidden_states).squeeze(-1)
        # (Batch_size, SeqLen, 1): 这意味着线性层对序列中的每一个位置（token）都独立计算出了一个标量价值
        # (B, SeqLen): 输出的是每个 token 的 value（Token-level value），而不是直接输出一个代表整个回答的单一值
        # 第 t 个位置的 值 V_t 表示 Critic 预测从第 t 个 token 开始直到句子结束，模型能获得到的期望奖励
        return values


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
    def reasoning_model_reward(rewards):
        # 1. 格式奖励（仅针对训练推理模型时使用）
        # 检查模型输出是否严格符合 <think>...</think><answer>...</answer> 的规范格式
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$" # 允许中间有空行

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 2. 标记奖励（防止严格奖励稀疏，仅针对训练推理模型时使用）
        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    # 初始化每个 response 的基础奖励为 0，维度为 [Batch_size]
    rewards = torch.zeros(len(responses), device=args.device)

    # 格式奖励
    if args.reasoning == 1:
        # 如果是训练推理模型，先加上基于规则的格式奖励
        rewards = reasoning_model_reward(rewards)

    # 使用reward model计算整个response的奖励
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # 将当前模型生成的 response 作为 assistant 的回复拼接进去，构造完整的对话记录
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            # 调用奖励模型 API 计算得分。注意这里是对actor模型生成的整个句子进行打分。
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            # 限制得分在 [-3.0, 3.0] 范围内，防止异常大/小的 reward 导致训练崩溃
            scale = 3.0
            score = max(min(score, scale), -scale)

            # 当args.reasoning=1时，额外计算<answer>内容的奖励
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 对纯answer内容单独计算reward
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # 加权合并：整体得分占 40%，最终答案得分占 60%
                    score = score * 0.4 + answer_score * 0.6
            
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards # shape: [B]


def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # # 获取当前批次的 prompt 列表 (List[str], 长度为 B)
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, # 将文本 prompt 编码为 token，左侧填充（对于生成任务通常用左侧填充）
                       max_length=args.max_seq_len, padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        # P是Prompt长度。这里的attention_mask为1的部分是Prompt部分，为0的部分是Padding部分。
        prompt_length = enc.input_ids.shape[1]

        # ========== 1. Rollout（采样）阶段 ==========
        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            # Actor 模型根据 prompt 生成后续的 token
            gen_out = model_for_gen.generate( # 此处是 .generate 而不是 .forward
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id) # 生成结果形状：[B, Prompt_len + Response_len]

        # 将生成的 token 解码回纯文本 response (去除 prompt 部分)
        responses_text = [tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        

        # ========== 2. 奖励与优势计算 ==========        
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # 获取该 batch 生成结果的最终奖励分数 [B]

        # 生成 full_mask 以区分实际 token 和 padding token 。为1的位置是非padding部分，为0的位置是Padding部分。
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        
        # 1. 这里得到了所有 token 的 value，形状为 [B, P+R] (即上面分析的 Batch_size, SeqLen=Prompt_len+Response_len)
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
        # 2. 找到每个序列中“最后一个有效 token”的索引
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1) # shape: [B]
        # 3. 核心：只把最后一个 token 的 value 提取出来，形状变成了 [B]
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]
        # 此处是 两个序列索引
        # torch.arange(values_seq.size(0): 生成一个序列长度的范围的索引序列 [0, 1, 2, ..., seq_len-1] shape: [seq_len]
        # last_indices: 每个序列中最后一个有效 token 的索引 比如 [2, 5, 7, 9] shape: [B]
        # 两个序列索引，得到每个序列中最后一个有效 token 的 value，形状变成了 [B]，比如 [0.6190, 0.3721, 0.8380, 1.9829]
        # 如果是 values_seq[:, last_indices]，实际取的是每一行sample的 last_indices 索引下的元素，得到的结果形状是 [B, B]，这是不对的

        # 4. 用这个提取出的值，与整个 response 的总 Reward 计算优势 A = Reward - Baseline(Value)
        advantages = rewards - values.detach()  # [B]
        """
        注意，在标准的复杂 PPO 实现中（比如 HuggingFace 的 TRL 库），通常会利用完整的 (B, SeqLen) 输出去计算基于 Token 的广义优势估计 GAE (Generalized Advantage Estimation)。
        但 MiniMind 的代码为了极简，将整个生成过程视为一步（One-step MDP），所以只用了最后一个 token 的价值来对齐整体的 Reward。
        """

        # ========== 3. PPO 策略网络（Actor）的前向传播 ==========
        with autocast_ctx:
            # 将生成的完整序列输入当前 Actor 模型，获取 logits
            # 此处用的是 .forward 方法
            # 模型已经回答完了，现在要按 token 重新给这条回答打分，算出它在当前策略下的概率
            # 得到每个位置的 logits / logprobs → 对应 RL 里的 评估当前策略对这些动作的概率
            # 这里是将得到的回答 一次性并行地计算对数概率，效率高
            res = actor_model(input_ids=gen_out, attention_mask=full_mask)
            logits = res.logits  # [B, P+R, V] V: vocab size
            # 如果是 MoE 模型，获取辅助损失（用于负载均衡等），否则设为 0
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
        
        labels = gen_out[:, 1:].clone()  # 偏移一位作为目标标签 [B, P+R-1]
        # gen_out: [A, B, C, D, <pad>]
        # labels:  [B, C, D, <pad>]  (长度变成了 4)
        
        # 对每个位置，取出“模型给真实下一个 token 的 log 概率” log P(a|s)
        logp_tokens = F.log_softmax(
            logits[:, :-1], # 但因为我们只要预测到最后一个真实 token，所以最后一个位置没有对应标签，因此要去掉最后一位：logits[:, :-1]
            dim=-1          # 表示沿着词表维度做 softmax
        ) # shape: [B, P+R-1, V]
        # logp_tokens 里面每个位置都有整个词表的 log 概率，
        # 但我们不要整个词表，只想要该位置上，真实标签 token 对应的 log 概率
        # .gather(dim=2, index=...) 可以实现这个目的
        logp_tokens = logp_tokens.gather(
            2,
            labels.unsqueeze(-1)    # shape: [B, P+R-1, 1]
            # 因为 gather(dim=2, index=...) 要求 index 的 shape 在被 gather 的维度之外要对齐，
            # 而这里我们要沿词表维度 dim=2 取某一个 token 的 logprob
        )
        """
        举例：
        假设词表大小 V=5，某一个样本的 logp_tokens 如下：
        logp_tokens[0] =
        [
            [-2.0, -0.5, -3.0, -1.2, -4.0],   # 位置0预测下一个token的 logprob
            [-1.5, -2.2, -0.3, -3.1, -4.5],   # 位置1
            [-0.1, -3.0, -2.2, -4.0, -5.0],   # 位置2
            [-2.5, -0.7, -1.1, -3.2, -4.8]    # 位置3
        ]
        如果对应 labels[0] 是 [1, 2, 0, 1]
        那么gather后得到：[-0.5, -0.3, -0.1, -0.7]
        也就是：
        第 0 个位置取 token 1 的 logprob
        第 1 个位置取 token 2 的 logprob
        第 2 个位置取 token 0 的 logprob
        第 3 个位置取 token 1 的 logprob
        """        
        logp_tokens = logp_tokens.squeeze(-1)  # [B, P+R-1]
        # 最终得到每个位置上，模型对真实下一个 token 的 log 概率。
        
        seq_len = gen_out.size(1) - 1 # 对齐 labels/logp_tokens 的长度, P+R-1
        # 构造掩码，仅对 "生成的 Response 部分" 的 logP 进行计算
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_length - 1
        """
        假设 prompt 长度 prompt_length=2, 原序列 gen_out = [p0, p1, r0, r1, eos]
        那么 lables = [p1, r0, r1, eos], 从 labels 的角度， response开始的位置是 prompt_length - 1 = 1
        假如 seq_len = 4, 那么 [0, 1, 2, 3] >= 1 -> resp_mask = [False, True, True, True], 从第 prompt_length - 1 个位置开始，后面都是 response 部分
        """

        # 把必须是 response 的部分 并上 非padding的部分，表示只保留 response 部分且非 padding 的位置
        # labels.eq(pad) = 1 表示是 padding 的位置，所以取反 (~labels.eq(pad)) 表示非padding部分
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]

        """
        [-0.5, -0.3, -0.1, -0.7]
        [ F  ,  T  ,  T  ,  T  ]
        两个一乘，就是 [0.0, -0.3, -0.1, -0.7] 再求和
        表示：当前策略下，整个 response 的总对数概率分布和
        """
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]


        # ========== 4. Old Actor 和 Ref Model 的前向传播 ==========
        with torch.no_grad():
            # 旧策略模型（用于 PPO 的重要性采样比率计算）
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            # 参考模型（用于防止策略过度偏移的 KL 惩罚计算）
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        # ========== 5. PPO 损失计算 ==========
        # KL散度： Actor 与 Old Actor 的分布差异 (仅用于监控指标)
        kl = (actor_logp - old_logp).mean()  # scalar

        # KL散度： Actor 与 参考模型 的分布差异 (直接作为惩罚项加入 loss)
        kl_ref = (actor_logp - ref_logp).mean()  # scalar
        
        # 计算重要性采样比率 (Importance Sampling Ratio): \pi_\theta / \pi_{old}
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        
        # PPO Clip 目标函数
        surr1 = ratio * advantages  # [B]
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
        # Actor loss
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar
        
        # Critic Loss: 预测的价值 Value 与实际 Reward 之间的均方误差 MSE
        value_loss = F.mse_loss(values, rewards)  # scalar

        # 总损失 = 策略损失 + 价值损失(乘以系数) + KL散度惩罚(乘以系数) + MoE辅助损失
        # 除以梯度累积步数以做平均
        loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        # ========== 6. 参数更新 ==========
        # 当达到梯度累积步数时，更新一次网络权重
        if step % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # ========== 7. 日志打印 ==========
        if is_main_process():
            # 计算生成的平均长度 (寻找 eos_token)
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            # 取出各项指标的值用于日志打印
            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            current_aux_loss = aux_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            # 控制台输出日志
            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, "
                   f"Reward: {reward_val:.4f}, KL: {kl_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        # ========== 8. 模型状态同步 ==========
        # 定期用 Actor 的最新权重去覆盖更新 Old Actor 模型
        if (step + 1) % args.update_old_actor_freq == 0:
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            state_dict = raw_actor.state_dict()
            # 拷贝一份到 CPU 并加载至 old_actor，防止直接关联显存中的计算图
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        # ========== 9. 模型保存 ==========
        # 定期保存，或者到达当前 Epoch 最后一个 iter 时保存
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            # 转换成半精度存到硬盘，节省空间
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir=f'{args.save_dir}/checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model,
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train() # 切回训练模式
            del actor_state # 手动释放内存

        # ========== 10. 内存清理 ==========
        # 删除不再需要的局部变量，防止每步训练后显存累积导致 OOM（内存溢出）
        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="internlm/internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=f'{args.save_dir}/checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    # Actor模型 (使用SFT训练的模型初始化)
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')

    # Old Actor模型 (使用SFT训练的模型初始化)
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False) # 用于重要性采样，不计算梯度

    # Reference模型 (使用SFT训练的模型初始化)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False) # 监督基线，不计算梯度

    # Critic模型 (使用SFT训练的模型初始化)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)

    # Reward模型 (这里使用外部训练好的模型)
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False) # 不计算梯度
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True, local_files_only=True)

    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # Actor 和 Critic 分别使用独立的优化器
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    
    # 计算总迭代次数以初始化学习率调度器 (CosineAnnealingLR 余弦退火)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model = getattr(actor_model, '_orig_mod', actor_model)
        critic_model = getattr(critic_model, '_orig_mod', critic_model)
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        # 如果使用 RoPE (旋转位置编码)，需要忽略其缓存参数，防止 DDP 报错
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist() #  打乱数据

        # 判断是否需要跳过已经训练过的 batch（针对断点续训）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
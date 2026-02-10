"""
MiniMind 的 Transformer Block 架构设计不仅克制，更集成了当前大模型最前沿的工程化技巧：

- RoPE & YaRN (Dynamic Scaling)： 代码中不仅实现了标准的旋转位置编码（RoPE），更内嵌了 YaRN (Yet another RoPE extensioN) 算法。
通过动态调整频率（ramp 函数），使得模型能够在推理时突破训练长度限制（如从 2k 外推至 32k），实现了“训练短，推理长”的高效策略。

- Pre-Norm RMSNorm： 摒弃了传统 LayerNorm 的中心化操作，仅保留缩放，结合 Pre-Norm 结构显著提升了深层网络的训练稳定性与收敛速度。

- GQA + Flash Attention： 采用了分组查询注意力（GQA），大幅压缩了 KV Cache 的显存占用；
同时在底层自动适配 PyTorch 的 F.scaled_dot_product_attention，根据环境自动启用 Flash Attention 加速，实现了显存与计算的双重优化。

- SwiGLU / Hybrid MoE： 前馈网络不仅使用了 GLU 门控机制，更在 MoE 模式下采用了 Hybrid（混合）专家架构（n_shared_experts + n_routed_experts）。
这种“共享专家负责通用知识，路由专家负责垂类知识”的设计（类似 DeepSeek-MoE），配合 Aux Loss 负载均衡，极大地提升了模型的非线性表达能力与参数利用率。

- Weight Tying & Vocab Compression： 除了精简词表外，MiniMindForCausalLM 中显式执行了 embed_tokens.weight = lm_head.weight 的 权重绑定（Weight Tying）。
这一技巧让输入 Embedding 与输出 Head 共享参数，在小参数量模型中能显著减少冗余，确保每一分参数预算都用在“刀刃”上。
"""


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None, # FFN 中间层维度
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8, # Query 的头数
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2, # Key/Value 的头数 (涉及 GQA) 当此值小于 num_attention_heads 时，即开启了 GQA (分组查询注意力)。这里 8 个 Q 头共享 2 组 KV 头（4:1），能显著降低推理显存占用。
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2, # Top-K 路由数。每个 Token 在推理时实际会激活的路由专家数量。尽管总专家多，但每个 Token 只计算这 2 个，保证了推理速度。
            n_routed_experts: int = 4, # 路由专家总数。可供选择的专用专家总数量。
            n_shared_experts: int = 1, # 共享专家数量。无论路由结果如何，所有 Token 必然会经过的专家。用于捕捉通用知识（这是 DeepSeek-MoE 架构的典型特征）。
            scoring_func: str = 'softmax', # 门控评分函数。Router 网络使用 Softmax 来计算每个专家的权重概率。
            aux_loss_alpha: float = 0.01, # 辅助损失系数。训练时的负载均衡惩罚项权重。防止 Router 总是只选某几个专家（导致专家坍塌），强制让所有专家都“忙起来”。
            seq_aux: bool = True, # 序列级辅助损失。计算辅助损失的范围是在整个序列级别上统计，而非仅针对单个 Token。
            norm_topk_prob: bool = True, # 概率归一化。选出 Top-K 个专家后，是否将这 K 个专家的权重重新归一化（使其和为 1）。有助于数值稳定。
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # torch.rsqrt: 计算 1/sqrt，比先 sqrt 再除更快
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    # ========== 步骤 1：计算基础频率 ==========
    # RoPE 频率公式：f_i = 1 / (rope_base^(2i/dim))
    #   其中 i 是维度索引（0, 2, 4, ..., dim-2），只使用偶数索引
    #   频率随维度索引增加而递减，形成不同频率的旋转

    # ========== 步骤 2：应用 YaRN 外推（如果启用） ==========
    if rope_scaling is not None:
        # 获取 YaRN 配置参数
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)  # 训练时的最大长度
        factor = rope_scaling.get("factor", 16)  # 外推因子
        beta_fast = rope_scaling.get("beta_fast", 32.0)  # 快速频率调整参数
        beta_slow = rope_scaling.get("beta_slow", 1.0)  # 慢速频率调整参数
        attn_factor = rope_scaling.get("attention_factor", 1.0)  # 注意力缩放因子
        
        # 如果目标长度超过训练长度，应用 YaRN 外推
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            # YaRN 公式：f'(i) = f(i) * ((1-γ) + γ/s)
            #   其中 γ 是线性斜坡函数，s 是缩放因子（factor）
            #   对于低频维度（i < low），不进行缩放
            #   对于高频维度（i > high），完全缩放
            #   对于中间维度，线性插值

            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # 计算频率调整的边界维度
            # inv_dim(b) 返回频率为 b 的维度索引
            low = max(math.floor(inv_dim(beta_fast)), 0)  # 低频边界
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)  # 高频边界

            # 计算线性斜坡函数 γ
            #   对于维度 i：γ(i) = (i - low) / (high - low)，限制在 [0, 1]
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0, 1
            )
            
            # 应用 YaRN 缩放：f'(i) = f(i) * ((1-γ) + γ/s)
            freqs = freqs * (1 - ramp + ramp / factor)

    # ========== 步骤 3：计算所有位置的频率 ==========
    # 为每个位置计算频率：freqs[pos, dim] = pos * freqs[dim]  freqs.shape = [seq_len, dim // 2]
    t = torch.arange(end, device=freqs.device) # 位置索引 [0, 1, 2, ..., end-1]
    freqs = torch.outer(t, freqs).float() # 外积：[end, dim//2]

    # ========== 步骤 4：计算 cos 和 sin 值 ==========
    # 将频率转换为 cos 和 sin 值，用于旋转矩阵
    # 由于 RoPE 使用复数旋转，需要将 dim//2 的频率复制到完整的 dim 维度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor # [end, dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor # [end, dim]
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用旋转位置编码（RoPE）到 Query 和 Key
    
    RoPE 通过复数旋转将位置信息编码到 Q 和 K 中：
        R_θ(x) = [x_0 * cos(θ) - x_1 * sin(θ), x_0 * sin(θ) + x_1 * cos(θ)] （二维旋转矩阵形式）
               = [
                    x_0 * cos(θ) + (- x_1) * sin(θ),  （第 1 维）
                    x_1 * cos(θ) + (+ x_0) * sin(θ),  （第 2 维）
                ]

    其中 x_0 和 x_1 是输入向量的实部和虚部，θ 是位置索引与频率的乘积。
    
    在实现中，将复数旋转分解为实部和虚部的线性组合，使用 rotate_half 函数实现。
    (所谓“实部/虚部”是把 embedding 维度两两配对当作复数，并不是模型内部真的存复数。)

    合起来可以写成一个很常用的实现形式：
        rope(x) = x ⋅ cosθ + rotate_half(x) ⋅ sinθ
    前提是 rotate_half(x) 定义为 把 (x_0, x_1) 变成 (-x_1, x_0) 的函数。

    ---
    实现细节：rotate_half 的两种常见配对方式
    
    方式 A：相邻两维配对（直觉版）
        (x_0, x_1), (x_2, x_3), ..., (x_{d-2}, x_{d-1})

    roate_half(x) 把每对变成 (-x_{2k+1}, x_{2k})

    方式 B：前半/后半配对（工程版，LLaMA 系很常见）
    把向量切成两半：x1=x[..., :d/2], x2=x[..., d/2:], 
    然后 rotate_half(x) = concat(-x2, x1)

    这等价于把维度配对为：
        (x_0, x_{d/2}), (x_1, x_{d/2+1}), ..., (x_{d/2-1}, x_{d-1})

    仍然是同一个二维旋转，只是“谁和谁是一对”的索引方式不同。

    不管哪种，核心恒等式都一样：
        x * cos + rotate_half(x) * sin。
    ---

    Args:
        q: Query 张量 [batch, seq_len, num_heads, head_dim]
        k: Key 张量 [batch, seq_len, num_kv_heads, head_dim]
        cos: 预计算的 cos 值 [seq_len, head_dim]
        sin: 预计算的 sin 值 [seq_len, head_dim]
        position_ids: 位置索引（未使用，cos/sin 已包含位置信息）
        unsqueeze_dim: 在哪个维度插入新维度以匹配 q/k 的形状（默认 1）
        
    Returns:
        q_embed: 应用 RoPE 后的 Query [batch, seq_len, num_heads, head_dim]
        k_embed: 应用 RoPE 后的 Key [batch, seq_len, num_kv_heads, head_dim]
    """
    def rotate_half(x):
        """
        旋转向量的后半部分
        
        将向量分成两半，交换位置并取反后半部分：
            [a, b, c, d] -> [-c, -d, a, b]
        
        这实现了复数旋转的实部/虚部交换。
        
        Args:
            x: 输入张量，最后一个维度会被分成两半
            
        Returns:
            旋转后的张量，形状与输入相同
        """
        # 将最后一个维度分成两半，交换位置并取反后半部分
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用 RoPE 旋转
    """
    # 公式：R_θ(x) = x * cos(θ) + rotate_half(x) * sin(θ)
    #   这等价于复数旋转：x * e^(iθ) = x * (cos(θ) + i*sin(θ))
    #   其中 rotate_half 实现了虚部的操作
    """

    # 调整 cos 和 sin 的形状以匹配 q/k：[seq_len, head_dim] -> [1, seq_len, 1, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # 对 Query 和 Key 分别应用旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    """
    GQA 是一种注意力机制优化，使用较少的 KV heads 来匹配更多的 Query heads。
    例如：8 个 Query heads 对应 2 个 KV heads，每个 KV head 需要重复 4 次。
    
    这样可以减少 KV 缓存的大小，在推理时节省显存。

    Args:
        x: Key 或 Value 张量 [batch, seq_len, num_kv_heads, head_dim]
        n_rep: 每个 KV head 需要重复的次数（n_rep = num_heads / num_kv_heads）
        
    Returns:
        重复后的张量 [batch, seq_len, num_heads, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :] #  [B, L, num_kv_heads, 1, head_dim]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim) # [B, L, num_kv_heads, n_rep, head_dim]
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim) # [B, L, num_heads, head_dim]
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads # KV heads 数量（通常小于 Query heads）
        assert args.num_attention_heads % self.num_key_value_heads == 0 # 确保 Query heads 数量能被 KV heads 数量整除

        self.n_local_heads = args.num_attention_heads # Query heads 数量, 8 个 Q 头
        self.n_local_kv_heads = self.num_key_value_heads # KV heads 数量, 2 组 KV 头
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 每个 KV head 需要重复的次数, 每组 KV 头被多少个 Q 头共享 = 4
        self.head_dim = args.hidden_size // args.num_attention_heads # 每个头的维度
        
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 检查是否支持 Flash Attention（需要 PyTorch >= 2.0）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # past_key_value: 缓存的 KV 值，用于增量解码 [batch, past_len, num_kv_heads, head_dim]
                use_cache=False, # 是否返回 KV 缓存供下次使用
                attention_mask: Optional[torch.Tensor] = None # attention_mask: 注意力掩码 [batch, seq_len]，1 表示有效位置，0 表示掩码位置
                ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz,  seq_len,  self.n_local_heads,  self.head_dim)
        xk = xk.view(bsz,  seq_len,  self.n_local_kv_heads,  self.head_dim)
        xv = xv.view(bsz,  seq_len,  self.n_local_kv_heads,  self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cache实现
        if past_key_value is not None:
            # 如果有缓存的 KV 值（增量解码），将其与当前 KV 拼接
            # past_key_value[0] 是缓存的 K，past_key_value[1] 是缓存的 V
            # 在序列维度（dim=1）上拼接：[batch, past_len+seq_len, num_kv_heads, head_dim]
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        # 如果需要缓存，保存当前的 KV 值
        past_kv = (xk, xv) if use_cache else None

        # 调整维度顺序为 [batch, num_heads, seq_len, head_dim]（Flash Attention 格式）
        # 对于 KV，需要重复 heads 以匹配 Query heads 数量
        xq = xq.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # [batch, num_heads, kv_len, head_dim]
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)  # [batch, num_heads, kv_len, head_dim]

        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention主要优化的是训练和预填充（Prefill）阶段
            # 条件：序列长度 > 1，不需要存 KV cache，没有复杂掩码
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # [batch, num_heads, seq_len, kv_len]
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1) # 上三角矩阵掩码，防止看到未来的 token
            # M_causal = [[0, -inf, -inf, -inf],
            #             [0,   0, -inf, -inf],
            #             [0,   0,   0, -inf],
            #             [0,   0,   0,   0]]

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9 # 0 -> -inf, 1 -> 0
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv # [batch, num_heads, seq_len, head_dim]

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            # 标准比例：intermediate_size = hidden_size * 8/3
            #   例如：hidden_size=512 -> intermediate_size ≈ 1365
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 向上取整到 64 的倍数（优化 GPU 计算效率）
            #   例如：1365 -> 1408 (64 * 22)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    MoE (Mixture of Experts) 门控网络
    
    负责为每个 token 选择 top-k 个专家，并计算专家权重。
    使用辅助损失（auxiliary loss）来鼓励专家负载均衡，防止专家退化。
    
    工作流程：
        1. 计算每个专家对每个 token 的分数（logits）
        2. 使用 softmax 转换为概率
        3. 选择 top-k 个专家
        4. 计算辅助损失（训练时）
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok # 每个 token 选择的专家数量
        self.n_routed_experts = config.n_routed_experts # 专家总数

        self.scoring_func = config.scoring_func # 评分函数（'softmax'）
        self.alpha = config.aux_loss_alpha # 辅助损失权重
        self.seq_aux = config.seq_aux # 是否在序列级别计算辅助损失

        self.norm_topk_prob = config.norm_topk_prob # 是否标准化 top-k 概率
        self.gating_dim = config.hidden_size # 门控网络输入维度
        
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # 门控网络权重：[n_routed_experts, hidden_size]
        #   每一行对应一个专家的权重向量
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        前向传播：为每个 token 选择专家
        
        Args:
            hidden_states: 输入张量 [batch, seq_len, hidden_size]
            
        Returns:
            topk_idx: 选择的专家索引 [batch * seq_len, top_k]
            topk_weight: 专家权重 [batch * seq_len, top_k]
            aux_loss: 辅助损失（标量），用于鼓励负载均衡
        """

        # hidden_states: 输入数据。
        # 形状是 [batch(批次大小), seq_len(句子长度), h(隐藏层维度)]
        # 例如: [2, 10, 512] 表示 2 句话，每句 10 个词，每个词用 512 维向量表示。
        bsz, seq_len, h = hidden_states.shape
        
        # ========== 步骤 1：计算专家分数 ==========
            
        # .view(-1, h): 结果形状变为 [batch * seq_len, h]。
        # 含义：把所有句子的所有词平铺开，变成一个长长的列表，因为我们对每个词是独立处理的。
        hidden_states = hidden_states.view(-1, h)

        # F.linear(input, weight): 线性层计算，数学公式是 Y = XW^T。
        # hidden_states 形状 [Total_Tokens, h]
        # self.weight 形状 [n_routed_experts, h]
        # 结果 logits 形状 [Total_Tokens, n_routed_experts]
        # 含义：计算每个 Token 和每个 Expert 的匹配分数（原始分数，未归一化）。
        logits = F.linear(hidden_states, self.weight, None)

        # ========== 步骤 2：转换为概率 ==========
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1) # [batch * seq_len, n_routed_experts]
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # ========== 步骤 3：选择 top-k 专家 ==========
        # torch.topk: 寻找张量中最大的 k 个值。
        # scores: 来源张量。
        # k=self.top_k: 要选几个（比如 2 个）。
        # dim=-1: 在专家维度上选。
        # sorted=False: 不需要对选出来的结果排序（为了速度）。
        # 返回值：
        #   topk_weight: [batch * seq_len, top_k] 选中的那 k 个专家的概率值。
        #   topk_idx: [batch * seq_len, top_k] 选中的那 k 个专家的索引（ID 号）。
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # ========== 步骤 4：标准化 top-k 概率（可选） ==========
        if self.top_k > 1 and self.norm_topk_prob:
            # 将 top-k 个专家的权重标准化，使其和为 1
            #   这样确保每个 token 的专家权重分布是归一化的
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            # ========== 步骤 5：计算辅助损失（训练时） ==========
            # 辅助损失用于鼓励专家负载均衡，防止某些专家被过度使用或完全不用

            scores_for_aux = scores # 也就是所有专家原本的概率分布
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1) # [batch, seq_len * top_k]

            if self.seq_aux:
                # === 方案 A：序列级辅助损失 (DeepSeek-V2/V3 常用) ===
                # 这种计算方式更精细，在每条样本内部看负载均衡。
                
                # 变形回 [batch, seq_len, n_routed_experts]
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)

                # 计算每个专家的使用频率（期望负载）
                # 创建一个全 0 矩阵用来统计次数
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                
                # scatter_add_: 这是一个复杂的“散射加法”操作。
                # 形象理解：这是在“投票”。
                # topk_idx_for_aux_loss 里的值是专家 ID，它告诉我们每个 Token 投给了谁。
                # 这行代码统计：在这个 Batch 里，每个专家被选中了多少次。
                ce.scatter_add_(
                    1, # dim
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device) # [batch, seq_len * top_k]
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                # .div_(...): 除以期望的平均次数，将其归一化。
                # 如果 ce = 1，说明该专家被选中的频率正好等于平均水平。

                # 计算损失：(实际使用频率 * 专家平均概率得分)
                # 这种损失设计会迫使模型倾向于让所有专家的使用频率和平均得分趋于一致。
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # === 方案 B：Token 级辅助损失 (传统的 Switch Transformer 做法) ===
                # 这种是全局统计所有 Token。
                
                # F.one_hot: 独热编码。如果 ID 是 3，变成 [0, 0, 0, 1, 0...]
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # mask_ce: [batch * seq_len * top_k, n_routed_experts]
                # [
                #   [0, 0, 1, 0, 0, ..., 0 (n_routed_experts-1 列)], # 第一个 token 选了第 3 个专家
                #   [0, 1, 0, 0, 0, ..., 0 (n_routed_experts-1 列)], # 第二个 token 选了第 2 个专家
                #   ...
                #   [0, 0, 0, 1, 0, ..., 0 (n_routed_experts-1 列)], # 第 N 个 token 选了第 4 个专家
                # ]

                ce = mask_ce.float().mean(dim=0) # [n_routed_experts] - 统计每个专家的平均使用频率

                # 计算每个专家得到的平均分（模型“想”选它的程度）。
                Pi = scores_for_aux.mean(dim=0) # [n_routed_experts] - 每个专家的平均分数

                # 计算负载均衡分数
                fi = ce * self.n_routed_experts

                # 经典的负载均衡损失公式：
                # minimize (N * sum(Pi * fi))
                # 只有当概率分布是均匀分布时，这个点积最小。
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 如果不在训练，或者不需要辅助损失，损失为 0
            aux_loss = scores.new_zeros(1).squeeze()

        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    MoE (Mixture of Experts) 前馈网络
    
    使用多个专家（FeedForward）处理不同的 token，通过门控网络动态选择专家。
    支持 路由专家（routed experts）和 共享专家（shared experts）两种类型。
    
    工作流程：
        1. 门控网络为每个 token 选择 top-k 个路由专家
        2. 每个 token 被路由到选中的专家处理
        3. 专家输出按权重加权求和
        4. 共享专家处理所有 token 并添加到输出

    Output = 共享专家输出 + Σ(路由专家输出 * 门控权重)
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        # 路由专家：通过门控网络动态选择，每个 token 只使用 top-k 个专家
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])

        # 负责为每个 token 选择专家并计算权重
        self.gate = MoEGate(config)

        # 共享专家：处理所有 token，不经过门控网络
        #   用于提供通用特征，增强模型表达能力
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            
        Returns:
            输出张量 [batch, seq_len, hidden_size]
        """
        identity = x  # 保存原始输入，用于共享专家
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
         # ========== 步骤 1：门控网络选择专家 ==========
        # 为每个 token 选择 top-k 个专家并计算权重
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # topk_idx: [batch * seq_len, top_k] - 专家索引
        # topk_weight: [batch * seq_len, top_k] - 专家权重

        # ========== 步骤 2：路由到专家处理 ==========
        x = x.view(-1, x.shape[-1]) # [batch * seq_len, hidden_size]
        flat_topk_idx = topk_idx.view(-1) # [batch * seq_len * top_k] - 展平的专家索引
        
        if self.training:
            # 训练模式：
            """目标：必须构建一张完整的、正确的计算图 (Computational Graph)，以便梯度（Gradients）能够反向传播更新参数。"""
            # num_experts_per_tok: Top-K 路由数。每个 Token 在推理时实际会激活的路由专家数量。
            # 为每个 token 的每个选中专家复制输入
            #   例如：top_k=2，每个 token 需要处理 2 次
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0) # [batch * seq_len * top_k, hidden_size]
            """
            当 top_k > 1 时（例如每个 token 选 2 个专家），同一个 token 的向量需要被送入两个不同的专家。
            在训练中，我们使用 repeat_interleave 显式地把数据复制一份。
            为什么？ 这样做可以让 PyTorch 清楚地知道：专家 A 的梯度要传回给副本 1，专家 B 的梯度要传回给副本 2，最后在底层这两个梯度会自动加和（Accumulate）回原始的 token embedding。
            """

            # 输入是一个大矩阵，但我们需要把它拆得支离破碎，送进不同的专家，算完再拼回来。
            # 结果聚合, 显式索引赋值
            y = torch.empty_like(x, dtype=x.dtype) # [batch * seq_len * top_k, hidden_size]

            # 对每个专家，处理分配给它的 token
            for i, expert in enumerate(self.experts):
                # 找到分配给专家 i 的 token 索引，并处理这些 token
                expert_out = expert(x[flat_topk_idx == i])

                if expert_out.shape[0] > 0:
                    # 如果有 token 分配给该专家，保存输出
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    # 如果没有 token 分配给该专家，也需要创建空输出（保持梯度流）否则会导致 DDP 卡死（跟推理时不一样）
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
                    """
                    原因：在使用多卡分布式训练（DDP）时，如果某个专家在某张卡上恰好没有分配到任何数据（flat_topk_idx == i 全为 False），它的梯度就是 None。这会导致 DDP 在进程同步时卡死（Hang）。
                    解决：这行代码强行构造了一个“值为 0 但依赖于专家参数”的计算节点，确保梯度流不断，防止训练卡死。推理时不需要反向传播，自然不需要这个 hack。
                    """

            # 按权重加权求和：每个 token 的 top-k 个专家输出加权平均
            y = (
                y.view(*topk_weight.shape, -1) # [batch * seq_len, top_k, hidden_size]
                * topk_weight.unsqueeze(-1) # [batch * seq_len, top_k, 1]
            ).sum(dim=1) # [batch * seq_len, hidden_size]
            y = y.view(*orig_shape) # [batch, seq_len, hidden_size]

        else:
            # 推理模式：使用优化的推理函数
            """目标：不需要算梯度，只需要以前向传播最快的方式得到 Y"""
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # ========== 步骤 3：添加共享专家输出 ==========
        # 共享专家处理所有 token，输出直接添加到结果中
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity) # 残差连接
        
        # 保存辅助损失供后续使用
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        优化的 MoE 推理函数（仅推理时使用） 
        
        通过批量处理每个专家的所有 token，减少计算开销。
        工作流程：
            1. 按专家索引排序 token
            2. 统计每个专家处理的 token 数量
            3. 批量处理每个专家的所有 token
            4. 按权重加权并累加到输出缓存
        
        Args:
            x: 输入张量 [batch * seq_len, hidden_size]
            flat_expert_indices: 展平的专家索引 [batch * seq_len * top_k]
            flat_expert_weights: 展平的专家权重 [batch * seq_len * top_k, 1]
            
        Returns:
            输出张量 [batch * seq_len, hidden_size]
        """
        expert_cache = torch.zeros_like(x) # 输出缓存

        # ========== 步骤 1：按专家索引排序 ==========
        # 将 token 按专家索引排序，使同一专家的 token 聚集在一起
        idxs = flat_expert_indices.argsort()  # 排序后的索引

        # ========== 步骤 2：统计每个专家处理的 token 数量 ==========
        # bincount: 统计每个专家被选中的次数
        # cumsum: 累积和，得到每个专家的 token 范围
        # 一次性算出每个专家处理多少数据，以及数据在数组中的起止位置。
        #   例如：[6, 15, 20, 26] 表示：
        #     - 专家 0 处理前 6 个 token
        #     - 专家 1 处理第 6-15 个 token
        #     - 专家 2 处理第 15-20 个 token
        #     - 专家 3 处理第 20-26 个 token
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        
        # 计算每个 token 的原始索引（去除 top_k 的重复）
        token_idxs = idxs // self.config.num_experts_per_tok

        # ========== 步骤 3：批量处理每个专家 ==========
        # 当 tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且 token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味 token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置 token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]

            # 如果该专家没有处理的 token，跳过 达到推理加速的目的
            if start_idx == end_idx:
                continue

            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx] # 该专家需要处理的 token

            # 批量处理该专家的所有 token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 应用权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 累加到输出缓存（使用 scatter_add 处理同一 token 被多个专家处理的情况）
            expert_cache.scatter_add_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out
            )
            # 原子操作，直接把结果“累加”到输出缓冲区对应的位置

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    # Input IDs -> Embeddings -> [Transformer Blocks x L] -> RMSNorm -> Output Hidden States
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 在输出之前进行最后一次 RMSNorm，这是 LLaMA 架构的标准做法
        # 形状: [H]

        # 预先计算所有可能位置的 Cos 和 Sin 值，避免前向传播时重复计算
        # freqs_cos/sin 形状: [MaxPos, HD]
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads, # Head Dim
            end=config.max_position_embeddings, # 最大位置索引 (如 32768)
            rope_base=config.rope_theta, # RoPE 基频
            rope_scaling=config.rope_scaling
        )
        # 将频率表注册为 buffer
        # buffer 不会被视为模型参数 (parameter)，不参与梯度更新，但会随模型权重文件保存
        # persistent=False 表示这些值可以根据 config 动态重新计算，不强制依赖权重文件
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        Args:
            input_ids: 输入序列 [B, S]。
                       训练时 S 是整个句子长度；
                       推理 Decoding 阶段 S 通常为 1。
            attention_mask: 掩码 [B, S]。
            past_key_values: 历史 KV 缓存列表，缓存每一层 layer 的 KV。
                             List 长度为 L，每个元素是 (K, V) 元组，长度等于 layer 数量。
                             K/V 形状: [B, Past_Len, Num_KV_Heads, HD]。
            use_cache: 是否开启 KV Cache 加速 (推理时为 True)。

        Returns:
            hidden_states: [B, S, H] 模型输出特征
            presents: 新的 KV Cache 列表
            aux_loss: MoE 负载均衡辅助损失
        """
        
        # 注意：推理 Decoding 阶段，seq_length 始终为 1
        batch_size, seq_length = input_ids.shape

        # ========== KV Cache 兼容性处理 ==========
        # 如果传入的是 Hugging Face 新版的高级 Cache 对象 (含有 .layers 属性)
        # MiniMind 暂时不支持，为了防止报错，强制清空缓存 (安全降级)
        if hasattr(past_key_values, 'layers'):
            past_key_values = None

        # 初始化 past_key_values
        # 如果没有缓存 (Prefill 阶段或训练阶段)，初始化为全 None 的列表
        past_key_values = past_key_values or [None] * len(self.layers)

        # ========== 计算起始位置 (start_pos) ==========
        # 这里的逻辑是确定当前输入的 Token 在整篇文章中的绝对位置索引
        # 1. 如果有缓存 (past_key_values[0] 不为 None):
        #    说明是推理的 Decoding 阶段。
        #    past_key_values[0][0] 是第 0 层的 Key Tensor，形状 [B, Past_Len, H_kv, HD]
        #    .shape[1] 就是 Past_Len (历史已经处理过的 Token 数量)
        #    这也是当前新 Token 的起始索引。

        # 2. 如果没有缓存:
        #    说明是 Prefill 阶段或训练阶段，从第 0 个位置开始。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Token Embedding
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # ========== 提取位置编码 (RoPE Slicing) ==========
        # 根据绝对位置 start_pos 和当前长度 seq_length，从预计算的表中切片
        # 切片范围: [start_pos : start_pos + seq_length]
        # 
        # 场景 A (训练/Prefill): start_pos=0, seq_len=N -> 取出前 N 个位置编码
        # 场景 B (推理 Decoding): start_pos=N, seq_len=1 -> 仅取出第 N 个位置的编码 (长度为 1)
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # ========== 逐层前向传播 ==========
        presents = [] # 用于收集每一层新的 KV Cache

        # zip 组合：将 模型层对象 与 该层对应的历史缓存 配对
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 输入: hidden_states [B, S, H]
            # 输出: 
            #   hidden_states: 更新后的特征 [B, S, H]
            #   present: 当前层更新后的 KV Cache (包含历史+当前), 形状 [B, Past_Len+S, H_kv, HD]
            hidden_states, present = layer(
                hidden_states,
                position_embeddings, # 传入切片好的位置编码
                past_key_value=past_key_value, # 传入该层的历史缓存
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # ========== 最终归一化 ==========
        # 经过所有层后，进行最后一次 RMSNorm
        # [B, S, H] -> [B, S, H]
        hidden_states = self.norm(hidden_states)

        # ========== 汇总 MoE 辅助损失 ==========
        # 检查每一层，如果是 MoE 层 (MOEFeedForward)，提取其 aux_loss
        # 将所有层的 aux_loss 相加，用于训练时的反向传播
        # 如果没有使用 MoE，总 aux_loss 为 0
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    # Input IDs -> [MiniMindModel] -> Hidden States -> [LM Head] -> Logits
    """
    1. 权重共享 (Weight Tying): 输入 Embedding 和 输出 LM Head 共享同一份参数，显著减少显存。
    2. 推理优化 (Logits Slicing): 支持只计算最后一个 Token 的 Logits，避免全量计算。
    3. 训练并行 (Parallel Training): 利用 Mask 实现一次性计算所有 Token 的 Loss。
    """
    
    config_class = MiniMindConfig # 指定配置类，Hugging Face 框架自动加载机制需要

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        # 初始化父类 PreTrainedModel (负责权重加载、保存、下载等)
        super().__init__(self.config)

        # ========== 1. 骨干网络 (Backbone) ==========
        # 实例化纯 Transformer Decoder
        # 输入: [Batch, Seq_Len] -> 输出: [Batch, Seq_Len, Hidden_Size]
        self.model = MiniMindModel(self.config)

        # ========== 2. 语言模型头 (LM Head) ==========
        # 这是一个线性投影层 (Linear Layer)
        # 作用: 将高维特征向量 (Hidden State) 映射回词表空间 (Vocab Space)
        # 形状: [Hidden_Size] -> [Vocab_Size]
        # bias=False: 现代大模型 (LLaMA等) 通常不使用偏置项，以提升数值稳定性
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # ========== 3. 权重共享 (Weight Tying) ==========
        # [重要优化] 将 Input Embedding 的权重指针指向 LM Head 的权重
        # 物理意义: 语义上，“输入一个词”和“预测一个词”使用的是同一个语义空间。
        # 显存优势: 词表通常很大 (如 64k)，权重共享能节省大量参数 (Hidden * Vocab)。
        self.model.embed_tokens.weight = self.lm_head.weight # Weight Tying

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        前向传播 (支持 训练 和 推理 两种模式)
        
        Args:
            input_ids: 输入序列 [Batch, Seq_Len]。
                       - 训练时: 是一整句话 (Seq_Len = N)。
                       - 推理时(Decoding): 通常只是最新生成的那个词 (Seq_Len = 1)。
            
            attention_mask: 掩码 [Batch, Seq_Len] (1=有效, 0=padding)。
            
            labels: 标签序列 [Batch, Seq_Len]。
                    - 如果提供此参数，模型会计算 Loss (训练模式)。
                    - 如果为 None，只返回 Logits (推理模式)。
            
            past_key_values: KV Cache 列表。
                    - 用于存储每层的历史 Token 的 Key/Value，避免重复计算。
            
            use_cache: 是否返回更新后的 KV Cache (推理时开启)。
            
            logits_to_keep: 【性能优化参数】
                    - 0 (默认): 计算所有 Token 的 Logits (训练时必须选这个)。
                    - 1 (常用): 只计算最后一个 Token 的 Logits (推理生成时用)。
                    原理: 避免在 lm_head 上进行无用的矩阵乘法计算。
        
        Returns:
            CausalLMOutputWithPast: 包含 loss, logits, hidden_states, past_key_values, aux_loss
        """

        # ========== 步骤 1: 骨干网络特征提取 ==========
        # hidden_states: [Batch, Seq_Len, Hidden_Size]
        # past_key_values: 包含了当前步新生成的 KV Cache
        # aux_loss: 如果使用了 MoE，这里会返回负载均衡损失；否则为 0
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        # ========== 步骤 2: Logits 计算范围优化 (Logits Slicing) ==========
        # lm_head 的计算量是 O(Seq_Len * Hidden * Vocab)，非常巨大。
        # 在推理时，我们只需要最后一个词的预测结果，不需要前文的预测。

        if isinstance(logits_to_keep, int):
            # logits_to_keep 是整数
            # logits_to_keep = 1 -> slice(-1, None) -> 取最后 1 个
            # logits_to_keep = 0 -> slice(None)     -> 取全部 (训练时)
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            # logits_to_keep 是张量 (高级用法，指定特定位置)
            slice_indices = logits_to_keep

        # 对 Hidden States 进行切片，只保留需要计算的部分
        # 推理时: [Batch, 100, Hidden] -> [Batch, 1, Hidden]
        # 训练时: [Batch, 100, Hidden] -> [Batch, 100, Hidden]
        sliced_hidden_states = hidden_states[:, slice_indices, :]
        
        # ========== 步骤 3: 映射到词表 (Projection) ==========
        # 执行矩阵乘法: X @ W.T
        # logits 形状: [Batch, Sliced_Len, Vocab_Size]
        # 这里的 logits 是未归一化的概率 (Log-odds)
        logits = self.lm_head(sliced_hidden_states)

        # ========== 步骤 4: 计算损失 (仅训练模式) ==========
        loss = None
        if labels is not None:
            # 因果语言模型的核心逻辑: "Shift Prediction" (位移预测)
            # 目标: 第 t 个时间步的 Logit，应该预测第 t+1 个时间步的 Label。
            
            # [Input]:  A  B  C  | D (丢弃最后一个 Token，因为它没有对应的 Label)
            # [Target]: B  C  D  | E  (丢弃第一个 Label，因为它没有对应的 Logit)
            
            # shift_logits: 去掉最后一个 Logit (因为它预测的是 E，但 Input 只有到 D，没有对应的 label)
            # 形状: [Batch, Seq_Len-1, Vocab]
            shift_logits = logits[..., :-1, :].contiguous()
            
            # shift_labels: 去掉第一个 Label (因为 A 之前没有 Logit 预测它，没有对应的 logit)
            # 形状: [Batch, Seq_Len-1]
            shift_labels = labels[..., 1:].contiguous()

            # 计算交叉熵损失 (Cross Entropy)
            # ignore_index=-100: 忽略标签为 -100 (Padding) 的位置，不计算梯度
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), # view(-1): 将 Batch 和 Seq 维度展平，变成 [Total_Tokens, Vocab]
                shift_labels.view(-1), # [Total_Tokens]
                ignore_index=-100
            )

        # ========== 步骤 5: 封装输出 ==========
        # 使用 Hugging Face 标准格式返回，确保兼容性
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

        # [MoE 特有] 挂载辅助损失
        # 训练循环中通常写法: total_loss = output.loss + alpha * output.aux_loss
        output.aux_loss = aux_loss

        return output

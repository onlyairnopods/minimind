from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train') # 优势是 Lazy Loading（懒加载）和 Memory Mapping（内存映射）。

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        text_content = str(sample['text'])
        tokens = self.tokenizer(
            text_content, 
            add_special_tokens=False, 
            max_length=self.max_length - 2, # 预留 2 个位置给 BOS 和 EOS
            truncation=True
        ).input_ids
        
        # [添加特殊标记] 在文本前后分别加上“开始符”(BOS)和“结束符”(EOS)
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # [填充 Padding] 如果长度不足 max_length，在后面补齐 pad_token_id
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        # 将所有 Padding 部分的标签设为 -100, 以便使用PyTorch 的 CrossEntropyLoss 时，设置忽略 -100 的位置，不计算 Loss
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

        # 为了实现“只对 Assistant 的回复计算 Loss”，我们需要在长文本中找到回复的起始和结束位置。
        # 这里预先计算好“Assistant起始符”和“结束符”对应的 Token ID 序列。
        # 注意：add_special_tokens=False 很重要，因为我们不想要 BOS/EOS 再次包裹这些片段。

        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        将多轮对话列表转换为纯文本字符串。
        例如：[{"role": "user", "content": "hi"}] -> "<|im_start|>user\nhi<|im_end|>\n..."
        """
        messages = conversations.copy()
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, # 这里只拼成字符串，先不转 ID，方便后续统一截断和处理
            add_generation_prompt=False, # SFT 是训练已有对话，不需要像推理时那样自动添加 "assistant:" 引导头
            tools=tools
        )

    def generate_labels(self, input_ids):
        """
        这是 SFT 代码的灵魂。生成与 input_ids 等长的 labels 序列。
        规则：
        - User 的话 -> 设为 -100 (PyTorch CrossEntropyLoss 默认忽略 -100)
        - Assistant 的话 -> 设为原本的 Token ID (参与计算梯度)
        - Padding -> 设为 -100
        """
        labels = [-100] * len(input_ids) # 初始化全为 -100（默认全不学）
        i = 0

        # 线性扫描 input_ids，找到“assistant”回答的区间（介于 bos_token+assistant 和 eos_token 之间），
        # 将这部分的 Label 设为真实的 Token ID。
        while i < len(input_ids):
            # 判断当前位置是否匹配“Assistant起始符” (self.bos_id)
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id) # 找到了 Assistant 说话的开头
                end = start

                # 继续向后找，直到找到“结束符” (self.eos_id)
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # 将 start 到 end 之间的部分（即回复内容）从 -100 恢复为真实的 input_ids
                # 只有这一部分会产生梯度，更新模型参数
                # min(..., self.max_length) 是防止越界
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]

                # 移动指针 i 到当前回复结束的位置，继续找下一轮（多轮对话场景）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                # 如果没匹配到，指针后移一位
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)

        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 生成 Mask 后的标签
        labels = self.generate_labels(input_ids)
        # # === 调试打印 检查 Mask 是否正确 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     # 打印 Input Token 和 对应的 Label (注意这里模拟了 Next Token Prediction 的错位)
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # bos_id 这里特指 Assistant 回答的起始特征，例如 "<bos>assistant\n"
        # 只有检测到这个序列，才意味着接下来的内容是模型需要学习的回答
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 获取 chosen (好回答) 和 rejected (坏回答) 的对话列表
        # 格式通常是: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上

        # 将 list 格式的对话转换成纯文本字符串
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids) # 生成 mask (用于标记哪些 token 是助手的回答，需要计算 loss)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # 输入是 x (0 到 N-1)，目标是 y (1 到 N)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        # 只保留 Assistant 回答部分的 Loss，忽略 User 输入和 Padding。
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 1. 寻找 Assistant 回答的“开始标记” (例如 "assistant\n")
            # 检查当前位置 i 是否匹配 bos_id 序列
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 找到了开始标记，start 指向回答内容的第一个 token
                start = i + len(self.bos_id)
                end = start

                # 2. 寻找 Assistant 回答的“结束标记” (例如 "<eos>\n")
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                # 3. 将 Start 到 End 之间的部分 mask 设为 1
                # 这部分就是模型实际生成的回答，我们需要计算它的 Loss
                # min(..., max_length) 防止越界
                # end + len(self.eos_id) 是为了让模型学会生成结束符 (EOS) 本身
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                
                # 更新 i 指针，跳过这段已经处理完的回答，继续寻找下一轮对话（如果是多轮对话）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                # 如果没匹配到开始标记，指针后移一位继续找
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')
        # 获取特殊 token 的 ID。这里尝试获取 "<BOS>assistant" 对应的 input_ids
        # add_special_tokens=False 表示不要自动在首尾添加额外的特殊符
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        # 获取结束符 (EOS token) 对应的 input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            # 根据索引的奇偶性判断角色。偶数轮(0, 2, 4...)是 user，奇数轮(1, 3, 5...)是 assistant
            # 这种写法强制假设对话是 user 和 assistant 严格交替进行的
            role = 'user' if i % 2 == 0 else 'assistant'

            messages.append({"role": role, "content": turn['content']})

            # 不断覆盖 answer，当循环结束时，answer 保存的就是对话列表中的最后一句话
            # （通常这一句应该是 assistant 给出的目标回复）
            answer = turn['content']

        prompt = self.tokenizer.apply_chat_template(
            messages[:-1], # messages[:-1] 切片表示：取除了最后一句话之外的所有对话（即历史上下文作为 Prompt）
            tokenize=False,# False 表示返回字符串，而不是 token IDs
            add_generation_prompt=True  # 这里需要True
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        # 注意这里返回的是普通字符串字典，并没有将其转换为 Tensor
        return {
            'prompt': prompt,
            'answer': answer
        }

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model")
    
    # pretrainDataset = PretrainDataset(data_path='../dataset/sample_pretrain_hq.jsonl', tokenizer=tokenizer, max_length=500)
    # print(len(pretrainDataset[0]))
    # print(pretrainDataset[0])

    sftDataset = SFTDataset(data_path='../dataset/sample_sft_mini_512.jsonl', tokenizer=tokenizer, max_length=500)
    print(f"{sftDataset.bos_id=}")
    print(f"{sftDataset.eos_id=}")
    # for i in range(min(2, len(sftDataset))):
    #     sample = sftDataset.samples[i]
    #     prompt = sftDataset.create_chat_prompt(sample["conversations"])
    #     print("--- Sample {} ---\n{}\n".format(i, prompt))
    print(sftDataset[0])



    print("\nRLAIF dataset")
    rlaifDataset = RLAIFDataset('../dataset/minimind_dataset/rlaif-mini.jsonl', tokenizer, max_length=1024)
    for i in range(min(2, len(rlaifDataset))):
        print(f"{rlaifDataset[i]=}")

    
    print("\nDPO dataset")
    dpoDataset = DPODataset('../dataset/minimind_dataset/dpo.jsonl', tokenizer, max_length=1024)
    for i in range(min(1, len(dpoDataset))):
        out = dpoDataset[i]
        print("=" * 60)
        print("--- Sample {} ---".format(i))

        # ---------- chosen ----------
        print("\n[chosen]")
        x_chosen = out["x_chosen"]
        mask_chosen = out["mask_chosen"]
        chosen_ids = x_chosen.tolist()
        chosen_len = len(chosen_ids)
        if tokenizer.pad_token_id is not None:
            while chosen_len > 0 and chosen_ids[chosen_len - 1] == tokenizer.pad_token_id:
                chosen_len -= 1
        print("解码前 (input_ids, 去 pad 后长度 {}):".format(chosen_len))
        print(chosen_ids[:chosen_len])
        print("解码后 (文本):")
        print(tokenizer.decode(chosen_ids[:chosen_len], skip_special_tokens=False))
        print("mask_chosen (0=不计算 loss, 1=计算, 长度 {}):".format(mask_chosen.shape[0]))
        print(mask_chosen.tolist())
        print("mask_chosen 非零个数:", mask_chosen.sum().item())

        # ---------- rejected ----------
        print("\n[rejected]")
        x_rejected = out["x_rejected"]
        mask_rejected = out["mask_rejected"]
        rej_ids = x_rejected.tolist()
        rej_len = len(rej_ids)
        if tokenizer.pad_token_id is not None:
            while rej_len > 0 and rej_ids[rej_len - 1] == tokenizer.pad_token_id:
                rej_len -= 1
        print("解码前 (input_ids, 去 pad 后长度 {}):".format(rej_len))
        print(rej_ids[:rej_len])
        print("解码后 (文本):")
        print(tokenizer.decode(rej_ids[:rej_len], skip_special_tokens=False))
        print("mask_rejected (0=不计算 loss, 1=计算, 长度 {}):".format(mask_rejected.shape[0]))
        print(mask_rejected.tolist())
        print("mask_rejected 非零个数:", mask_rejected.sum().item())
        print()
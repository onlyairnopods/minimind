# 注：不建议再重复训练tokenizer（“词典”），MiniMind已自带，此脚本仅供学习和参考。基于不同词典训练的模型将导致输出完全不统一，降低社区的模型复用性
# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one. This script is for learning and reference only. Training models with different tokenizers will lead to inconsistent outputs and reduce model reusability in the community.
import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

DATA_PATH = '../dataset/pretrain_hq.jsonl'
TOKENIZER_DIR = '../model_learn_tokenizer/'
VOCAB_SIZE = 6400

def get_texts(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10000: break # 实验性，可只用前10000行测试
            data = json.loads(line)
            yield data['text']

def train_tokenizer(data_path, tokenizer_dir, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # add_prefix_space=False: 不在文本开头添加空格
    #   设置为 False 是因为中文等语言不需要在开头添加空格
    #   如果设置为 True，会在每个文本前添加空格（适合英文等需要空格分隔的语言）
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    texts = get_texts(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 强制验证特殊 Token 的 ID（确保顺序正确） 这些断言确保特殊 Token 的 ID 严格按照预期顺序
    # 后续的聊天模板和模型训练都依赖这些 ID，顺序错误会导致严重问题
    assert tokenizer.token_to_id("<|endoftext|>") == 0 # 必须是 0（用作 pad_token 和 unk_token）
    assert tokenizer.token_to_id("<|im_start|>") == 1 # 必须是 1（用作 bos_token）
    assert tokenizer.token_to_id("<|im_end|>") == 2 # 必须是 2（用作 eos_token）

    os.makedirs(tokenizer_dir, exist_ok=True)
    # ========== 保存分词器（关键：兼容 Hugging Face 格式） ==========
    # 保存的文件说明：
    #   1. tokenizer.json:
    #      - 分词器核心配置文件
    #      - 包含 BPE 模型参数、预处理/解码逻辑、完整词汇表 和 合并规则
    #      - 这是 Hugging Face tokenizers 库的标准格式
    #
    #   2. vocab.json + merges.txt:
    #      - vocab.json: BPE 词汇表，包含所有 token 及其 ID 映射
    #      - merges.txt: BPE 合并规则，记录训练过程中学到的字符对合并顺序
    #      - 这两个文件由 tokenizer.model.save() 生成，是 BPE 算法的核心数据
    #
    #   3. tokenizer_config.json:
    #      - Transformers 库的兼容配置文件
    #      - 包含聊天模板、特殊 Token 映射、模型最大长度等
    #      - 这个文件需要手动创建（见下方代码）
    #      - 关键字段：
    #        * chat_template: 聊天消息格式化模板（Jinja2 格式）
    #        * pad_token/eos_token/bos_token: 特殊 Token 映射
    #        * model_max_length: 模型支持的最大序列长度（32768）
    #        * tokenizer_class: 指定使用 PreTrainedTokenizerFast 类加载
    tokenizer.model.save(tokenizer_dir)                                 # 保存 BPE 模型（生成 vocab.json 和 merges.txt）
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))       # 保存核心分词器配置 tokenizer.json

    # ========== 手动创建配置文件 tokenizer_config.json ==========
    config = {
        "add_bos_token": False,
        # add_bos_token: 是否自动在输入开头添加 BOS（Begin of Sequence）token
        #   设置为 False，因为我们使用 <|im_start|> 手动标记消息开始
        "add_eos_token": False,
        # add_eos_token: 是否自动在输入结尾添加 EOS（End of Sequence）token
        #   设置为 False，因为我们使用 <|im_end|> 手动标记消息结束
        "add_prefix_space": False,
        # add_prefix_space: 是否在文本前添加空格
        #   设置为 False，因为中文等语言不需要前缀空格
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False, # 解码时不在左侧去除空格
                "normalized": False, # 不进行 Unicode 规范化
                "rstrip": False, # 解码时不在右侧去除空格
                "single_word": False, # 不是单词语境（可以出现在词中间）
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        # bos_token: 序列开始符，映射到 <|im_start|>
        #   在生成任务中，模型会使用这个 token 作为序列的开始
        "clean_up_tokenization_spaces": False,
        # clean_up_tokenization_spaces: 是否清理分词后的空格
        #   设置为 False，保持原始格式
        "eos_token": "<|im_end|>",
        # eos_token: 序列结束符，映射到 <|im_end|>
        #   在生成任务中，模型生成这个 token 时表示序列结束
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        # pad_token: 填充符，映射到 <|endoftext|>
        #   在批处理时，较短的序列会用这个 token 填充到相同长度
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        # unk_token: 未知词 token，映射到 <|endoftext|>
        #   当遇到词汇表中不存在的词时使用（BPE 理论上不会有未知词，但保留此配置）
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
         # chat_template: 聊天消息格式化模板（Jinja2 语法）
        #   这是本分词器最核心的配置之一，定义了如何将多轮对话转换为模型输入
        #
        # 模板功能说明：
        #   1. 支持工具调用（tools）：如果提供了工具定义，会格式化工具调用提示
        #   2. 支持系统消息：处理 system role 的消息
        #   3. 支持多轮对话：格式化 user、assistant、tool 等不同角色的消息
        #   4. 消息格式：<|im_start|>role\ncontent<|im_end|>\n
        #   5. 生成提示：如果 add_generation_prompt=True，会在末尾添加 <|im_start|>assistant\n
        #
        # 使用示例：
        #   messages = [
        #       {"role": "system", "content": "你是一个助手"},
        #       {"role": "user", "content": "你好"},
        #       {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
        #   ]
        #   prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        #   输出：
        #   <|im_start|>system
        #   你是一个助手<|im_end|>
        #   <|im_start|>user
        #   你好<|im_end|>
        #   <|im_start|>assistant
        #   你好！有什么可以帮助你的？<|im_end|>
        #
        # 注意：这个模板是 Jinja2 格式，支持条件判断、循环等复杂逻辑
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        # 保存配置文件到 tokenizer_config.json
        #   这个文件是 Transformers 库加载分词器时必需的配置文件
        json.dump(config, f, ensure_ascii=False, indent=4) # ensure_ascii=False: 允许保存中文字符（不转义为 \uXXXX）
    print("Tokenizer training completed.")


def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    # 使用聊天模板格式化消息
    #   apply_chat_template 会使用 tokenizer_config.json 中的 chat_template
    #   将消息列表转换为模型输入格式
    print('-'*100)
    print(new_prompt)


    print('-'*100)
    print('tokenizer词表长度：', len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    # skip_special_tokens=False: 保留特殊 token（<|im_start|>, <|im_end|> 等）
    #   设置为 False 以便完整还原原始格式
    print('decoder一致性：', response == new_prompt, "\n")
    # 验证解码后的文本是否与原始格式化文本完全一致
    #   如果一致，说明分词器的编码和解码逻辑正确


    print('-'*100)
    print('流式解码（字节缓冲）测试：')
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []

if __name__ == '__main__':
    # train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    # eval_tokenizer(TOKENIZER_DIR)
    eval_tokenizer("../model")

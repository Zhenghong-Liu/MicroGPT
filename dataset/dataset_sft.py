import torch
from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import os
import json 
from tqdm import tqdm

# Dataset 类来自于：https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py

class SFTDataset(Dataset):
    """
    自定义数据集类，用于监督微调(Supervised Fine-Tuning)任务
    继承自PyTorch的Dataset类，用于加载和处理对话数据
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, cs):
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} Token Loss Mask (length: {len(input_ids)}) ---")
        # for i, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        #     token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        #     token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')  # 处理换行等不可见字符
        #     print(f"Token {i:3d}: {token_id:5d} -> '{token_str:10s}' | mask: {mask}")
        # print(f"--- End of Sample {index} ---")
        # # ================================
        return X, Y, loss_mask
    


if __name__ == "__main__":

    MAX_LEN = 513
    DATASET_FILE_NAME = "sft_512.jsonl" 
    # 包含您所有 .tar 文件的本地目录
    DATA_DIR = "/media/liuzh/data/DLData/minimind/" 

    # 导入词汇表
    tokenizer = AutoTokenizer.from_pretrained(DATA_DIR)
    VOCAB_SIZE = len(tokenizer)
    print(f"词汇表大小 (Vocab Size): {VOCAB_SIZE}")

    # 构造数据集
    full_dataset = SFTDataset(DATA_DIR + DATASET_FILE_NAME, tokenizer, max_length=MAX_LEN)
    print(f"数据集加载完成，总样本数: {len(full_dataset)}")

    ## 📦 创建 DataLoader
    full_dataloader = DataLoader(
        full_dataset,
        shuffle=True,  
        batch_size=3,
        pin_memory=True,  
        num_workers=4, # 提高数据加载速度
    )

    print(f"\n训练 DataLoader 创建完成，总批次数量: {len(full_dataloader)}")




    # --- 示例：检查 DataLoader 输出 ---
    for (input_ids, labels, loss_mask) in full_dataloader:
        print("\n--- 检查第一个 Batch 数据 ---")
        print(f"input_ids shape: {input_ids.shape}")  # 期望: [BATCH_SIZE, MAX_LEN]
        print(f"labels shape: {labels.shape}")        # 期望: [BATCH_SIZE, MAX_LEN]
        print(f"loss_mask shape: {loss_mask.shape}")        # 期望: [BATCH_SIZE, MAX_LEN]
        print(f"第一个序列的 input_ids (部分): {input_ids[0, :10]}")
        print(f"第一个序列的 labels (部分): {labels[0, :10]}")
        print(f"第一个序列的 loss_mask (部分): {loss_mask[0, :10]}")
        break # 只查看第一个批次


    sample_ids = input_ids[0]
    print(f"\n--- 检查第一个 Batch 序列的解码 ---")
    sample_text = tokenizer.decode(sample_ids, skip_special_tokens=False)

    print(f"采样数据: {sample_text}")
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.dataset_sft import SFTDataset
from model.microGPT import MicroGPT
from utils.utils import sample_output

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# ==============================================================
# 定义参数******************************************************=
# ==============================================================

# 最大序列长度
MAX_LEN = 513
# 训练批次大小
BATCH_SIZE = 32 

DATASET_FILE_NAME = "sft_512.jsonl" 
DATA_DIR = "/media/liuzh/data/DLData/minimind/"  #数据集路径，下载地址：https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files



#===============================================================
# 加载词汇表****************************************************=
# ==============================================================
tokenizer = AutoTokenizer.from_pretrained(DATA_DIR)
VOCAB_SIZE = len(tokenizer)
print(f"词汇表大小 (Vocab Size): {VOCAB_SIZE}")



#===============================================================
# 加载数据集****************************************************=
# ==============================================================
full_dataset = SFTDataset(DATA_DIR + DATASET_FILE_NAME, tokenizer, max_length=MAX_LEN)
print(f"数据集加载完成，总样本数: {len(full_dataset)}")

## 📦 创建 DataLoader
full_dataloader = DataLoader(
    full_dataset,
    shuffle=True,  
    batch_size=BATCH_SIZE,
    pin_memory=True,  
    num_workers=4, # 提高数据加载速度
)
print(f"\n训练 DataLoader 创建完成，总批次数量: {len(full_dataloader)}")



#===============================================================
# 定义模型******************************************************=
# ==============================================================
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 12
D_FF = D_MODEL * 4
DROPOUT = 0.0

micro_gpt = MicroGPT(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, D_FF, DROPOUT)
micro_gpt.load_state_dict(torch.load("./assert/micro_gpt_chat.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
micro_gpt = micro_gpt.to(device, dtype=torch.bfloat16)




#===============================================================
# 定义训练策略******************************************************=
# ==============================================================
EPOCHS = 1  #一般训练1轮，或者2-6轮
LEARNING_RATE = 5e-4 # 学习率

# 梯度累积配置
GA_STEPS = 4  # 每多少步更新一次梯度，相当于BATCH_SIZE *= GA_STEPS
ITER_STEP = 0 # 用于跟踪总的迭代次数
# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(micro_gpt.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))



# 🛠️ 关键改进 1: 学习率调度器
# 总的优化步数 (考虑梯度累积)
from torch.optim.lr_scheduler import CosineAnnealingLR # 导入调度器
# 学习率 Warmup 步数
WARMUP_STEPS = 500
TOTAL_TRAIN_STEPS = (len(full_dataloader) * EPOCHS) // GA_STEPS
# Cosine Annealing 调度器 (T_max 是周期，这里设为总步数)
scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_TRAIN_STEPS - WARMUP_STEPS, eta_min=1e-6) 
# Warmup 初始学习率
WARMUP_START_LR = 1e-7



#===============================================================
# 训练模型******************************************************=
# ==============================================================
def get_lr_warmup(step, max_lr, start_lr, warmup_steps):
    """计算 Warmup 阶段的学习率"""
    if step < warmup_steps:
        return start_lr + (max_lr - start_lr) * (step / warmup_steps)
    return max_lr

train_loss_history = []
for epoch in range(EPOCHS):
    micro_gpt.train()
    total_loss = 0
    iter_step = 0

    optimizer.zero_grad()
    for (input_ids, labels, loss_mask) in tqdm(full_dataloader):
        # ========================================================================================#
        # ===============构造输入输出数据============================================================#
        # ========================================================================================#
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        loss_mask = loss_mask.to(device)

        labels = labels.clone() # 创建 labels 的副本，不修改dataloader中的原始数据
        labels[loss_mask == 0] = -100  # 将mask位置置为-100，表示忽略这些位置的损失计算

        # 1) 构造 key_padding_mask：哪里不是 pad，就设 1
        key_padding_mask = (input_ids == tokenizer.pad_token_id).bool()


        # 🛠️ 关键改进 2: 学习率更新逻辑
        # 1. Warmup 阶段
        if ITER_STEP < WARMUP_STEPS:
            lr = get_lr_warmup(ITER_STEP, LEARNING_RATE, WARMUP_START_LR, WARMUP_STEPS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # 2. Cosine Annealing 阶段
        elif ITER_STEP % GA_STEPS == 0:
            # 只有在梯度更新时才调用 scheduler.step()
            pass # 调度器将在 optimizer.step() 之后调用


        # ========================================================================================#
        # ================训练模型，计算损失=========================================================#
        # ========================================================================================#
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # BF16 autocast, 混合精度训练
            outputs = micro_gpt(input_ids, key_padding_mask=key_padding_mask) # [batch, seq, vocab]
            # 交叉熵计算损失
            loss = loss_fn(outputs.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
            loss = loss / GA_STEPS
        
        # 反向传播（累积梯度）
        loss.backward()
        total_loss += loss.item() * GA_STEPS
        ITER_STEP += 1

        # 检查是否达到累积步数
        if ITER_STEP % GA_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(micro_gpt.parameters(), 1.0) # 梯度裁剪
            optimizer.step()
            optimizer.zero_grad()

            # 3. 在 Cosine 阶段更新调度器
            if ITER_STEP >= WARMUP_STEPS:
                scheduler.step()
            train_loss_history.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']


        # ========================================================================================#
        # ===============检查模型性能===============================================================#
        # ========================================================================================#
        iter_step += 1
        if iter_step % 2000 == 0:
            print(f"Epoch {epoch+1}, Iter {iter_step}, Loss: {total_loss/iter_step}")

            prompts = [
                "你有什么特长？",
                "为什么天空是蓝色的",
                "请用Python写一个二分查找的函数",
                '解释一下"光合作用"的基本过程',
            ]

            for prompt in prompts:
                generated_text = sample_output(prompt, micro_gpt, tokenizer, device)
                print(f"提示词: {prompt}")
                print(f"回答: {generated_text}")
                print("\n")

            torch.save(micro_gpt.state_dict(), f"micro_gpt_chat_{epoch}_ing.pth")
            micro_gpt.train() # 重新设置为训练模式


    # 保存模型
    torch.save(micro_gpt.state_dict(), f"micro_gpt_chat_{epoch}.pth")
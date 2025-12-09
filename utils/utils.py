import torch

# 生成输出
def sample_output(prompt, model, tokenizer, device, MAX_NEW_TOKENS=100, TEMPERATURE=0.8, TOP_K = 50):
    tools = None
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, # 建议设置为 True，以便模型知道何时开始生成
        tools=tools
    )
    # 编码为 Token IDs
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
    # 确定初始序列长度
    current_length = input_ids.shape[1]

    # 生成输出序列
    model.eval()
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            
            # 1. 模型前向传播：预测下一个 token
            # 注意: 如果序列过长，这里可能需要进行 K-V 缓存优化 (但在 MicroGPT 中先忽略)
            outputs = model(input_ids) 
            
            # 模型的输出 logits 形状是 [Batch Size, Seq Len, Vocab Size]
            # 我们只关心序列中最后一个位置 (即对下一个 token 的预测)
            next_token_logits = outputs[:, -1, :]
            
            # 2. 应用温度和 Top-K 采样
            
            # 调整 logits (降温，增加随机性)
            if TEMPERATURE > 0:
                next_token_logits = next_token_logits / TEMPERATURE
            
            # --- 核心修复点: 标准 Top-K 过滤 ---
            # 找到 Top-K 的值和索引
            v, i = torch.topk(next_token_logits, TOP_K)
            
            # 创建一个与 next_token_logits 形状相同、填充负无穷的张量
            # 只有 Top-K 的位置不会被屏蔽
            filtered_logits = torch.full_like(next_token_logits, float('-inf'))
            
            # 使用 scatter_() 将 Top-K 的值 (v) 散布到新的张量 (filtered_logits) 的对应位置 (i)
            filtered_logits = filtered_logits.scatter_(dim=-1, index=i, src=v)
            
            # 3. 采样下一个 token ID
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # ------------------------------------
            
            # 4. 检查停止条件
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # 5. 更新输入序列
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    # 3. 解码输出
    generated_ids = input_ids[0, current_length:].tolist() # 仅获取新生成的 IDs
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text.strip()
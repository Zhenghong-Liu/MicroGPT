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
        add_generation_prompt=False,
        tools=tools
    )

    # prompt_text += f"{tokenizer.bos_token}assistant\n"
    # 编码为 Token IDs
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
    # 确定初始序列长度
    current_length = input_ids.shape[1]

    # 生成输出序列
    model.eval()
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            
            # 1. 模型前向传播：预测下一个 token
            # 只传入当前完整的序列
            outputs = model(input_ids) 
            
            # 模型的输出 logits 形状是 [Batch Size, Seq Len, Vocab Size]
            # 我们只关心序列中最后一个位置 (即对下一个 token 的预测)
            next_token_logits = outputs[:, -1, :]
            
            # 2. 应用温度和 Top-K 采样
            # 调整 logits (降温，增加随机性)
            if TEMPERATURE > 0:
                next_token_logits = next_token_logits / TEMPERATURE
                
            v, i = torch.topk(next_token_logits, TOP_K)
            
            # ⬇️ 修正点 1: 确保 torch.arange 在相同的设备上 ⬇️
            # 创建一个与 next_token_logits 在同一设备上的张量
            device = next_token_logits.device
            vocab_indices = torch.arange(outputs.shape[-1], device=device)
            
            # 修正点 2: 使用正确的张量进行 isin 检查
            # i 已经是 [Batch_size, TOP_K] 形状
            
            # ⚠️ 注意: next_token_logits 的形状是 [1, Vocab Size]。i 的形状是 [1, TOP_K]。
            # isin 接受 [Vocab Size] 和 [TOP_K] 形状的张量
            
            # 将 i (top-k indices) 展平为一维向量，用于 isin 检查
            top_k_indices = i.flatten() 

            # 将不在 Top-K 范围内的 logits 设为负无穷
            # torch.isin 检查 vocab_indices 中的元素是否在 top_k_indices 中
            filter_mask = ~torch.isin(vocab_indices, top_k_indices)
            
            # 应用过滤（只针对批量中的第一个元素，因为 next_token_logits 是 [1, Vocab Size]）
            next_token_logits[0, filter_mask] = float('-inf')

            # 3. 采样下一个 token ID
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 4. 检查停止条件
            # 如果模型生成了 <EOS> 标记，则停止
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # 5. 更新输入序列
            # 将新生成的 token 追加到 input_ids 中
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    # 3. 解码输出
    generated_ids = input_ids[0, current_length:].tolist() # 仅获取新生成的 IDs
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text.strip()
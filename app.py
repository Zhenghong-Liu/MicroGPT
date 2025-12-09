import streamlit as st
import torch
from transformers import AutoTokenizer
import os
import sys
from model.microGPT import MicroGPT
from utils.utils import sample_output



# 模型权重文件路径
# MODEL_WEIGHTS_PATH = "./assert/micro_gpt_chat.pth"
MODEL_WEIGHTS_PATH = "./assert/micro_gpt_chat_0_ing.pth"
# MODEL_WEIGHTS_PATH = "./assert/micro_gpt_pretrain_1.pth"
# 分词器 (Tokenizer) 所在目录
# 注意：这应该与您训练时使用的 DATA_DIR 一致
DATA_DIR = "/media/liuzh/data/DLData/minimind/"

# microGPT 模型参数 (必须与训练时保持一致)
VOCAB_SIZE = 6400 # 假设这是您的词汇表大小，如果模型未在训练时使用特殊标记，通常是这个值
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 12
D_FF = D_MODEL * 4
DROPOUT = 0.0

# 设置设备
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")




@st.cache_resource
def load_model_and_tokenizer():
    """在应用启动时加载模型和分词器"""
    try:
        # 1. 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(DATA_DIR)

        # 2. 确定 VOCAB_SIZE
        global VOCAB_SIZE
        VOCAB_SIZE = len(tokenizer)
        
        # 3. 初始化模型
        model = MicroGPT(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, D_FF, DROPOUT)
        
        # 4. 加载权重
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # 5. 迁移到设备并设置为 bfloat16 (与训练时保持一致)
        model = model.to(DEVICE, dtype=torch.bfloat16)
        model.eval()

        st.success(f"✅ 模型 microGPT 和分词器加载成功，运行在 {DEVICE} 上。")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ 模型或分词器加载失败，请检查路径和模型参数！错误: {e}")
        st.stop()
        
# --- Streamlit 界面主体 ---
def main():
    st.set_page_config(
        page_title="MicroGPT Chatbot",
        page_icon="",
        layout="wide"
    )


    st.title("MicroGPT 对话界面")

    # 提示词和信息组织
    with st.expander("📝 示例提示词和使用说明", expanded=False):
        st.markdown("""
            欢迎使用 **MicroGPT** 模型聊天应用。
            
            - **模型参数:** 模型配置为 $D_{model}=512$, $N_{layers}=12$。
            - **使用技巧:** 尝试在侧边栏调整 **温度** 和 **Top-K** 参数来观察生成结果的变化。
            
            **推荐示例:**
            * “解释一下‘光合作用’的基本过程”
            * “请用 Python 写一个计算斐波那契数列的函数”
            * “如何才能更好地学习深度学习？”
        """)

    st.info(f"✨ 当前模型参数: **{D_MODEL}** 维度, **{NUM_LAYERS}** 层。")

    # 1. 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()

    # 2. 侧边栏：参数设置
    with st.sidebar:
        st.header("⚙️ 推理参数设置")
        
        # 温度 (Temperature) 滑块
        temperature = st.slider(
            "温度 (Temperature)",
            min_value=0.01,
            max_value=1.5,
            value=0.8,
            step=0.01,
            help="控制生成文本的随机性。温度越高，结果越多样化（越随机）。"
        )

        # Top-K 采样滑块
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=100, # 最大值为词汇表大小
            value=50,
            step=1,
            help="限制模型只从概率最高的 K 个词中采样。K 越小，生成越保守。"
        )

        # 最大生成长度
        max_new_tokens = st.slider(
            "最大生成长度 (Max New Tokens)",
            min_value=10,
            max_value=512,
            value=256,
            step=10,
            help="模型单次回答生成的最长 Token 数。"
        )
        
        st.info("💡 **提示:** 更改参数后，新一轮对话将使用新参数。")

    # 3. 对话历史初始化
    if "messages" not in st.session_state:
        # 初始化对话历史，包含一个系统提示
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "assistant", "content": "你好！我是 MicroGPT，有什么可以帮您的吗？"}
        ]
        
    # 4. 展示历史对话
    # 过滤掉 "system" 角色，只展示用户和助手的消息
    for message in st.session_state.messages:
        if message["role"] in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 5. 处理新的用户输入
    if prompt := st.chat_input("请在这里输入您的问题..."):
        # 将用户输入添加到历史记录
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 展示用户输入
        with st.chat_message("user"):
            st.markdown(prompt)

        # 获取模型回答
        with st.chat_message("assistant"):
            with st.spinner("🤖 MicroGPT 正在思考..."):
                # 构造用于传递给 sample_output 的 **当前完整对话上下文**
                # ⚠️ 注意: sample_output 默认是单轮对话，如果需要多轮，
                # 您需要修改 sample_output 使其接收并处理 st.session_state.messages
                # 而不是自己构造 messages 列表。
                
                # 当前版本的 sample_output 仅接受一个 "prompt" 字符串。
                # 简单起见，我们只把用户最新的 prompt 传入：
                
                # 确保 sample_output 函数可以处理您训练时的对话格式
                generated_text = sample_output(
                    prompt, 
                    model, 
                    tokenizer, 
                    DEVICE,
                    MAX_NEW_TOKENS=max_new_tokens,
                    TEMPERATURE=temperature,
                    TOP_K=top_k
                )

            # 展示模型回答
            st.markdown(generated_text)
            
            # 将模型回答添加到历史记录
            st.session_state.messages.append({"role": "assistant", "content": generated_text})

if __name__ == "__main__":
    main()
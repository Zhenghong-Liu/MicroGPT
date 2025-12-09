import streamlit as st
import torch
from transformers import AutoTokenizer
import os
import sys

# 确保项目根目录在 Python 路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.microGPT import MicroGPT
from utils.utils import sample_output

# ==================== 模型配置 ====================
MODEL_WEIGHTS_PATH = "./assert/micro_gpt_chat.pth"
DATA_DIR = "./dataset"

VOCAB_SIZE = 6400
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 12
D_FF = D_MODEL * 4
DROPOUT = 0.0

DEVICE = torch.device("cpu")

# ==================== 自定义 CSS 样式 ====================
st.markdown("""
<style>
    .main {
        background-color: #0a0a0a;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.2em;
        margin-bottom: 0.5em;
        
    }
    .subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #999999;
        margin-bottom: 1.5em;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
        max-width: 800px;
        margin: auto;
    }
    .message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    .user-message {
        flex-direction: row-reverse;
    }
    
    .message-text {
        color: #333;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        line-height: 1.5;
        word-wrap: break-word;
    }
    .user-message .message-text {
        background-color: #e6f4ff;
    }
    .footer {
        text-align: center;
        color: #777;
        font-size: 0.9em;
        margin-top: 2rem;
        opacity: 0.7;
    }
    .sidebar-title {
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .slider-label {
        color: #bbb;
        font-size: 0.9em;
    }
    .btn-primary {
        background-color: #4a4a4a;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.9em;
    }
    .btn-primary:hover {
        background-color: #666;
    }
            
    .avatar {
        width: 40px;
        height: 38px;
        border-radius: 50%;
        margin-right: 0.8rem;
        margin-left: 1.0rem;
        background-color: #000;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 1.0em;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 加载模型与分词器 ====================
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(DATA_DIR)
        global VOCAB_SIZE
        VOCAB_SIZE = len(tokenizer)

        model = MicroGPT(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, D_FF, DROPOUT)
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(DEVICE, dtype=torch.bfloat16)
        model.eval()

        # st.success(f"✅ 模型加载成功，运行在 {DEVICE} 上。")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ 加载失败: {e}")
        st.stop()

# ==================== 主函数 ====================
def main():
    st.set_page_config(
        page_title="MicroGPT",
        page_icon="🧠",
        layout="wide"
    )

    # === 页面顶部标题与提示 ===
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="title">你好，我是MicroGPT</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">内容完全由AI生成，请务必仔细甄别<br>Content AI-generated, please discern with care</div>', unsafe_allow_html=True)

    # === 侧边栏设置 ===
    with st.sidebar:
        st.markdown('<div class="sidebar-title">⚙️ 推理参数</div>', unsafe_allow_html=True)
        temperature = st.slider(
            "温度 (Temperature)",
            min_value=0.01,
            max_value=1.5,
            value=0.8,
            step=0.01,
            help="控制生成随机性：越高越自由，越低越保守。"
        )
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="从概率最高的 K 个词中采样。"
        )
        max_new_tokens = st.slider(
            "最大生成长度",
            min_value=10,
            max_value=512,
            value=256,
            step=10,
            help="模型一次最多生成多少 token。"
        )

    # === 初始化模型 ===
    if "model" not in st.session_state:
        st.session_state.model, st.session_state.tokenizer = load_model_and_tokenizer()

    # === 初始化对话历史 ===
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是一个大型语言模型，能够生成各种文本，包括故事、诗歌、代码、文章等。我的目标是帮助你解决问题、提供信息、娱乐等。"}
        ]

    # === 展示对话历史 ===
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            with st.container():
                st.markdown(f'<div class="message"><div class="avatar">Micro</div><div class="message-text">{msg["content"]}</div></div>', unsafe_allow_html=True)
        elif msg["role"] == "user":
            with st.container():
                st.markdown(f'<div class="message user-message"><div class="message-text">{msg["content"]}</div></div>', unsafe_allow_html=True)

    # === 用户输入 ===
    prompt = st.chat_input("请在这里输入你的问题...")

    if prompt:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.container():
            st.markdown(f'<div class="message user-message"><div class="message-text">{prompt}</div></div>', unsafe_allow_html=True)

        # 生成回复
        with st.spinner("🤖 正在思考..."):
            generated_text = sample_output(
                prompt,
                st.session_state.model,
                st.session_state.tokenizer,
                DEVICE,
                MAX_NEW_TOKENS=max_new_tokens,
                TEMPERATURE=temperature,
                TOP_K=top_k
            )
        
        # 添加助手回复
        st.session_state.messages.append({"role": "assistant", "content": generated_text})
        with st.container():
            st.markdown(f'<div class="message"><div class="avatar">Micro</div><div class="message-text">{generated_text}</div></div>', unsafe_allow_html=True)

    # === 默认示例按钮（可选）===
    if not st.session_state.messages[0]["content"].startswith("你好！我是一个大型语言模型"):
        st.session_state.messages[0]["content"] = "你好！我是一个大型语言模型，能够生成各种文本，包括故事、诗歌、代码、文章等。我的目标是帮助你解决问题、提供信息、娱乐等。"

    # === 添加一个示例按钮（可选）===
    if st.button("🎯 试试问我：'你有什么特长？'", key="example_button"):
        st.session_state.messages.append({"role": "user", "content": "你有什么特长？"})
        with st.container():
            st.markdown(f'<div class="message user-message"><div class="message-text">你有什么特长？</div></div>', unsafe_allow_html=True)

        with st.spinner("🤖 正在思考..."):
            example_response = sample_output(
                "你有什么特长？",
                st.session_state.model,
                st.session_state.tokenizer,
                DEVICE,
                MAX_NEW_TOKENS=128,
                TEMPERATURE=0.8,
                TOP_K=50
            )
        st.session_state.messages.append({"role": "assistant", "content": example_response})
        with st.container():
            st.markdown(f'<div class="message"><div class="avatar">Micro</div><div class="message-text">{example_response}</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # 关闭 chat-container

    # === 底部提示 ===
    st.markdown('<div class="footer">© 2025 MicroGPT | 内容由AI生成，请谨慎使用</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
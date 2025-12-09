# **🤖 MicroGPT：大模型入门的极简实践**

## **🌟 项目理念：揭秘大模型的奥秘**

在大型语言模型（LLM）日益流行的今天，我们往往感觉它们高不可攀，被复杂的工程优化和庞大的参数量所笼罩。

**MicroGPT** 的核心目标正是打破这种遥不可及感。

众所周知，现代的 GPT 架构本质上是 **Transformer 的 Decoder** 部分

![Transformer Decoder Architecture](./assert/image.jpeg)./assert/image.jpeg)

。相比完整的 Transformer，其实现更为简洁。本项目致力于：

1. **极简架构：** 剥离大型 LLM 因分布式训练和推理优化而带来的复杂性。
2. **核心代码：** 仅用最简洁、最易读的代码实现 GPT 的核心机制。
3. **入门首选：** 以一个仅有 **50M 参数量**的微型模型，为所有希望深入理解 LLM 内部代码原理的开发者和学习者提供一个清晰、无障碍的起点。

如果你渴望理解 LLM 的内在代码机制，MicroGPT 将是你理想的入门导师。

## **✨ 主要特性**

* **微小而强大：** 模型参数量仅 **50M**，易于在普通 CPU 或低配 GPU 上运行和调试。
* **纯粹的 GPT 结构：** 基于标准的 Transformer Decoder 块实现，代码高度简化。
* **完整的工程链：** 提供从模型定义、权重加载到 Streamlit 网页端聊天的完整流程。
* **易于部署：** 使用 Streamlit 快速搭建交互式聊天界面 (app.py)。

## **📁 架构概览**

本项目结构清晰，旨在实现最小的学习路径：

MicroGPT/
├── assert/
│   └── micro\_gpt\_chat.pth  \# 模型权重文件
├── model/
│   └── microGPT.py             \# \*\*GPT 模型核心定义 (重点学习)\*\*
├── utils/
│   └── utils.py                \# 辅助函数，如采样 (sample\_output)
├── app.py                      \# Streamlit 聊天前端界面
├── README.md                   \# 本文件
└── ...

## **🚀 快速开始**

### **1\. 环境准备**

本项目需要 Python 环境和以下依赖：

pip install torch transformers streamlit datasets

### **2\. 模型下载与配置**

请确保您已下载模型权重文件 (micro\_gpt\_chat.pth) 并将其放置在 assert/ 目录下。

同时，由于模型需要加载分词器（Tokenizer），请将您训练时使用的分词器文件（通常是 vocab.txt 等）下载到或配置在 DATA\_DIR 路径下。

### **3\. 运行 Web 界面**

通过 Streamlit 启动聊天应用：

streamlit run app.py

应用启动后，您可以在浏览器中访问提供的地址，与 MicroGPT 进行实时对话，并通过侧边栏调整推理参数，观察结果变化。

## **💡 深入代码学习**

对于希望学习 LLM 实现原理的同学，我们强烈建议从以下文件入手：

1. **model/microGPT.py：** 查看 MicroGPT 类的定义。这里包含了多头自注意力机制、前馈网络和残差连接等 GPT 的所有核心组件。
2. **app.py：** 学习如何加载模型、分词器，以及如何将推理参数（如 temperature, top\_k）集成到 Web 聊天界面中。
3. **utils/utils.py：** 理解模型推理（Generation）过程，特别是**采样策略**是如何影响文本生成结果的。

## **🔗 项目链接**

| 类型               | 链接                                                                           | 备注                           |
| :----------------- | :----------------------------------------------------------------------------- | :----------------------------- |
| **在线演示** | [MicroGPT](https://modelscope.cn/studios/SodaWaterWater/MicroGPT/summary "网站链接") | 欢迎试用 MicroGPT 的对话能力！ |

**感谢您的关注！让我们一起轻松入门 LLM！**

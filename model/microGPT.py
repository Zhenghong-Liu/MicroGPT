import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))
    
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True, # 确保输入是 [Batch, Seq, Feature] 格式
        )
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = FFN(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, key_padding_mask=None):
        # 1. 准备输入
        ln_x = self.ln1(x) 
        seq_len = ln_x.size(1)
        device = x.device

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=seq_len, 
            device=device 
        ).to(torch.bfloat16)

        # 3. 调用标准 MultiheadAttention
        attn_out, _ = self.attn(
            query=ln_x,
            key=ln_x,
            value=ln_x,
            attn_mask=causal_mask,          # 因果掩码 [Seq, Seq]
            key_padding_mask=key_padding_mask, # 填充掩码 [Batch, Seq]
            is_causal=True
        )

        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x


class MicroGPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, dropout=0.1):
        super().__init__()
        self.num_hiddens = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.gpt_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.gpt_blocks.append(DecoderBlock(d_model, nhead, d_ff, dropout))

        self.ln_f = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, X, key_padding_mask=None):
        X = self.embedding(X) * math.sqrt(self.num_hiddens)
        X = self.pos_encoding(X)
        for block in self.gpt_blocks:
            X = block(X, key_padding_mask=key_padding_mask)

        X = self.ln_f(X)
        return self.fc(X)
    

if __name__ == '__main__':
    VOCAB_SIZE = 6400
    D_MODEL = 512
    NHEAD = 8
    NUM_LAYERS = 12
    D_FF = D_MODEL * 4
    DROPOUT = 0.0

    micro_gpt = MicroGPT(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, D_FF, DROPOUT)
    


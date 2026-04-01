from turtle import forward
import torch
import torch.nn as nn
import math

"""
编码器 (Encoder) ：任务是“理解”输入的整个句子。
解码器 (Decoder) ：任务是“生成”目标句子。
"""


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    def __init__(self,d_model,dropout=0.1, max_len = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        # pe (positional encoding) 的大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        # 将 pe 注册为 buffer，这样它就不会被视为模型参数，但会随模型移动（例如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(1) 是当前输入的序列长度
        # 将位置编码加到输入向量上
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    """
    def __init__(self, d_model, num_heads) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 和输出的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 1. 计算注意力得分 (QK^T)
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        
        # 2. 应用掩码 (如果提供)
        if mask is not None:
            # 将掩码中为 0 的位置设置为一个非常小的负数，这样 softmax 后会接近 0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # dim=-1 (最后一个维度，即列)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # 4. 加权求和 (权重 * V)
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x:torch.Tensor):
        # 将输入 x 的形状从 (batch_size, seq_length, d_model)
        # 变换为 (batch_size, num_heads, seq_length, d_k)
        """
        一次处理batch_size个句子
        每个句子seq_length个词
        每个词用d_model维向量表示
        """
        batch_size, seq_length, d_model = x.size()
        #normal input LxD  output LxD_k
        #mulitiHead input: LxD ==> output: Lx(Head_num)D_k     (Head_num)D_k = D   D_k=D//Head_num
        #self.num_headsxself.d_k = d_model
        return x.view(batch_size, seq_length, self.num_heads,self.d_k).transpose(1,2)
    
    def combine_heads(self,x:torch.Tensor):
        # 将输入 x 的形状从 (batch_size, num_heads, seq_length, d_k)
        # 变回 (batch_size, seq_length, d_model)
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size,seq_length,self.d_model)
        

    def forward(self, Q, K, V, mask=None):
        # 1. 对 Q, K, V 进行线性变换
        # (L x d_model)*(d_model x d_model) = batch x L x d_model => Q (batch_size, num_heads, L, d_k)
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # 2. 计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 合并多头输出并进行最终的线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    """
    def __init__(self,d_model, d_ff, dropout=0.1) -> None:
        super(PositionWiseFeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # 最终输出形状: (batch_size, seq_len, d_model)
        return x


# --- 编码器核心层 ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. 多头自注意力
        attm_output = self.self_attn(x, x, x, mask)
        # 残差 + LN
        x = self.norm1(x + self.dropout(attm_output))

        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        # 残差 + LN
        x = self.norm2(x + self.dropout)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. 掩码多头自注意力 (对自己)
        att_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(att_output))

        # 2. 交叉注意力 (对编码器输出)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

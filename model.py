import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class RumorDetectModel(nn.Module):
    def __init__(
        self,
        embedding_size=300,  # 词向量大小
        origin_hsize=32,   # 原文语义大小
        comment_hsize=64,  # 评论语义大小
        attn_hsize=32,     # 注意力打分大小
        comment_dropout=0.5,  # 评论dropout
        fc_dropout=0.5,       # 分类隐层dropout
        output_size=2       # 类别数
    ):
        super().__init__()
        # 语义提取
        self.origin_bilstm = nn.LSTM(embedding_size, origin_hsize, batch_first=True, bidirectional=True)
        self.comment_lstm = nn.LSTM(embedding_size, comment_hsize, batch_first=True)
        # 注意力
        self.comment_dropout = nn.Dropout(comment_dropout)
        self.attn_U = nn.Linear(2*origin_hsize, attn_hsize, False)
        self.attn_W = nn.Linear(comment_hsize, attn_hsize, False)
        self.attn_v = nn.Linear(attn_hsize, 1, False)
        # 分类
        self.linear_dropout = nn.Dropout(fc_dropout)
        self.linear = nn.Linear(2*origin_hsize+comment_hsize, output_size)

    def forward(self, inputs):
        origin, comment = inputs  # 解包
        # origin: [l1*300, l2*300, l3*300, ...]
        # comment: [n1*300, n2*300, n2*300, ...] # 被压缩成300的了

        # 用bilstm提取原文语义
        origin = pack_sequence(origin, False)
        origin_vec, _ = self.origin_bilstm(origin)
        origin_vec, lengths = pad_packed_sequence(origin_vec, True)  # (B, maxL, 2H_o)
        origin_vec = origin_vec.view(origin_vec.size(0), origin_vec.size(1), 2, -1)
        origin_vec = torch.stack(
            [torch.cat([origin_vec[i, length-1, 0, :], origin_vec[i, 0, 1, :]])
             for i, length in enumerate(lengths)]
        )  # (B, 2H_o)
        origin_vec = torch.tanh(origin_vec)

        # 用lstm提取评论语义
        comment = pack_sequence(comment, False)
        comment_vec, _ = self.comment_lstm(comment)
        comment_vec, lengths = pad_packed_sequence(comment_vec, True)  # (B, maxN, H_c)
        comment_vec = torch.tanh(comment_vec)

        # 注意力机制
        outputs = []
        weights = []  # 用来保存非张量权重结果, 用于输出显示
        for i, length in enumerate(lengths):
            origin = origin_vec[i:i+1]  # (1, 2H)
            comment = self.comment_dropout(comment_vec[i, 0:length])  # (N, 2H) # 随机失活一部分评论
            # ((1, 2H) @ U(2H, H) + (N, 1, 2H) @ W(2H, H)) @ (H, 1)
            # ((1, H) + (N, 1, H)) @ v(H, 1)
            # (N, 1, 1)
            alpha = self.attn_v(torch.tanh(self.attn_U(origin) + self.attn_W(comment.unsqueeze(1))))
            weight = torch.softmax(torch.tanh(alpha.squeeze(-1)), dim=0)  # (N, 1) # 用tanh使得权重概率之间不会差很大
            comment = torch.sum(comment*weight, dim=0, keepdim=True)  # (1, 2H) # 加权平均
            outputs.append(torch.cat([origin, comment], dim=-1))
            weights.append(weight.view(-1).cpu().detach().numpy()*length.item()*0.5)  # 乘以数量放大比例

        outputs = torch.cat(outputs)  # (B, 4H)

        # 分类
        outputs = self.linear_dropout(outputs)
        outputs = self.linear(outputs)
        outputs = torch.softmax(outputs, dim=-1)

        return (outputs, weights)

import torch
import torch.nn as nn
from math import sqrt

"""
Модуль механизма внимания для трансформера (LLM)

Содержит два класса:
- HeadAttention: одноголовое скалярное внимание с маскированием (causal mask)
- MultiHeadAttention: многоголовое внимание с конкатенацией голов и проекцией Wo

Использование:
    multi_head_attn = MultiHeadAttention(
        num_heads=8,
        emb_size=512,
        head_size=64,
        max_seq_len=1024,
        dropout=0.1
    )
    output = multi_head_attn(x)  # x: (batch, seq_len, emb_size)
"""
class HeadAttention(nn.Module):

    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super(HeadAttention, self).__init__()

        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        self.Wk = nn.Linear(in_features=emb_size, out_features=head_size)
        self.Wq = nn.Linear(in_features=emb_size, out_features=head_size)
        self.Wv = nn.Linear(in_features=emb_size, out_features=head_size)


        # делаем маску, нижнюю треугольную матрицу
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.tensor):
        batch_size, seq_len, emb_size = x.shape

        K = self.Wk(x)
        Q = self.Wq(x)
        V = self.Wv(x)

        attention_matrix = Q @ K.transpose(-2, -1) # матрица внимания

        attention_matrix = attention_matrix / sqrt(self.head_size)

        mask_ = self.mask[:seq_len, :seq_len]

        attention_matrix = attention_matrix.masked_fill(
            mask_ == 0, float('-inf')
        )

        attention_weight = torch.softmax(attention_matrix, dim=2)

        output_matrix = torch.matmul(attention_weight, V)
        return output_matrix



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int,max_seq_len: int,
                 dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        # создаем список содержащий heads, в количестве определяемым параметром num_heads
        self.list_head = nn.ModuleList()
        for _ in  range(num_heads):
            self.list_head.append(HeadAttention(emb_size, head_size, max_seq_len))

        # создаем матрицу весов Wo которая будет перемножаться на сконкатенированный тензор внимания
        self.Wo = nn.Linear(head_size * num_heads, emb_size)

        # создаем слой Dropuot для борьбы с переобучением
        self.dropout = nn.Dropout(p = dropout)


    def forward(self, x: torch.tensor):

        # список содержащий матрицы Wo которые будут конкатенироваться
        list_Wo = []
        for head in self.list_head:
            list_Wo.append(head.forward(x))

        # конкатенируем матрицы полученные на выходе из всех head
        concatenated_matrix = torch.cat(list_Wo, dim=2)

        # прогоняем конкатенированную матрицу через полносвязный слой размерности
        # head_size*num_heads х emb_size
        out_matrix = self.Wo(concatenated_matrix)

        # прогоняет через слой dropout для борьбы с переобучением
        out_matrix_do = self.dropout(out_matrix)

        return out_matrix_do



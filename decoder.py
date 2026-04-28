import  torch
import  torch.nn as nn

from LLM_BPE import BPE
from Feed_Forward_Network import FeedForward
from attention import HeadAttention, MultiHeadAttention

"""
Модуль декодера трансформера для LLM

Содержит класс Decoder, который объединяет:
- MultiHeadAttention (механизм внимания)
- FeedForward (полносвязная сеть)
- LayerNorm (нормализация) и residual connections (сложение вход-выход)

Структура соответствует блоку декодера из архитектуры transformer
"""
class Decoder(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int,
                 max_seq_len: int, dropout = 0.1):
        super(Decoder, self).__init__()

        # создаем внутри поле содержащее MultiHeadAttention
        self.MultiHeadAttention = MultiHeadAttention(
            num_heads = num_heads,
            emb_size = emb_size,
            head_size = head_size,
            max_seq_len = max_seq_len,
            dropout = dropout
        )

        # создаем внутри поле содержащее FeedForward
        self.FeedForward = FeedForward(
            emb_size = emb_size,
            dropout = dropout
        )

        # создаем первый и второй слой нормализации
        self.norm_1 = nn.LayerNorm(emb_size)
        self.norm_2 = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor):

        # прогоняем тензор через Heads (блоки внимания), создаем тензор внимания
        o1 = self.MultiHeadAttention.forward(x)

        # складываем 2 тензора
        x_o1 = torch.add(x, o1)

        # нормализуем полученый тензор
        x_o1_norm = self.norm_1(x_o1)

        # прогоняем полученный тензор через полносвязные слои
        o2 = self.FeedForward.forward(x_o1_norm)

        # складываем полученный тензор, с входным тензором
        o2_plus = torch.add(x_o1_norm, o2)

        # нормализуем второй раз
        o2_plus_norm = self.norm_2(o2_plus)

        return o2_plus_norm









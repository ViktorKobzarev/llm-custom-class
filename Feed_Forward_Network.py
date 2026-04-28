import torch
import torch.nn as nn

"""
Модуль полносвязной сети для блока трансформера (FFN)

Содержит класс FeedForward — двухслойную сеть с расширением в 4 раза и ReLU.
Используется в декодере после механизма внимания.
"""
class FeedForward(nn.Module):

    def __init__(self, emb_size: int, dropout = 0.1):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(emb_size, 4*emb_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(4*emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor):

        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.dropout(x)

        return x




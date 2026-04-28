import torch
import torch.nn as nn

"""
Модуль эмбеддингов для GPT

Содержит два класса эмбеддингов:
- TokenEmbeddings: преобразование ID токенов в плотные векторы
- PositionalEmbeddings: добавление информации о позиции токена в последовательности
"""
class TokenEmbeddings(nn.Module):
    """
        Токен-эмбеддинги (стандартный Embedding слой)

        Преобразует целочисленные ID токенов в векторы фиксированного размера.

        Args:
            vocab_size (int): размер словаря (количество уникальных токенов)
            emb_size (int): размерность эмбеддинга

        Forward:
            x (torch.Tensor): тензор ID токенов формы (batch, seq_len)
            Returns: тензор эмбеддингов формы (batch, seq_len, emb_size)
        """
    def __init__(self, vocab_size: int, emb_size: int ):

        super(TokenEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor):

        return self.embeddings(x)


class PositionalEmbeddings(nn.Module):
    """
       Позиционные эмбеддинги (обучаемые)

       Добавляет информацию о позиции токена в последовательности.
       Использует обучаемую матрицу размером (max_seq_len, emb_size).

       Args:
           max_seq_len (int): максимальная длина последовательности
           emb_size (int): размерность эмбеддинга (должна совпадать с TokenEmbeddings)

       Forward:
           seq_len (int): текущая длина последовательности
           Returns: тензор позиционных эмбеддингов формы (seq_len, emb_size)

       Примечание:
           Возвращается срез матрицы весов [:seq_len], который будет
           поэлементно сложен с токен-эмбеддингами в модели GPT.
       """
    def __init__(self, max_seq_len: int, emb_size: int):

        super().__init__()

        self.position_embeddings_matrix = nn.Embedding(max_seq_len, emb_size)


    def forward(self, seq_len: int):
        matrix = self.position_embeddings_matrix.weight[:seq_len]
        return matrix

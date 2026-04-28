import torch
import torch.utils.data as data

"""
Модуль подготовки данных для обучения LLM

Содержит класс GetData, который наследует torch.utils.data.Dataset.
Используется DataLoader для формирования батчей последовательностей.

Формат данных: на входе список токенов, на выходе пары (x, y) для языкового моделирования
"""

"""данный класс GetData будет использовать DataLoader при формировании батчей"""
class GetData(data.Dataset):
    # инициализация
    def __init__(self, data: list, seq_len: int, device = "cpu"):
        super(GetData, self).__init__()

        self.data = data
        self.seq_len = seq_len
        self.device = device

    # возвращает длинну, запрашивается DataLoader-ром
    def __len__(self):
        return len(self.data) - self.seq_len - 1

    # также вызывается DataLoader-ром для формирования батчей
    def __getitem__(self, idx: int):
        x = torch.tensor(self.data[idx: idx + self.seq_len])
        y = torch.tensor(self.data[idx + 1 : idx + 1 + self.seq_len])

        return (x, y)

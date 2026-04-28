# импортируем инструменты (стандартные)
import torch
import torch.nn as nn
from math import sqrt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

# импортируем созданные модули
from token_embedings import TokenEmbeddings, PositionalEmbeddings
from attention import HeadAttention, MultiHeadAttention
from Feed_Forward_Network import FeedForward
from decoder import Decoder

"""
Модуль GPT (Generative Pre-trained Transformer)

Импорты:
- стандартные: torch, nn, DataLoader, Adam, F
- пользовательские: TokenEmbeddings, PositionalEmbeddings, attention, FeedForward, Decoder

Содержит основной класс GPT, который объединяет:
- токен-эмбеддинги + позиционные эмбеддинги
- стек декодеров (блоков внимания + FFN)
- выходной линейный слой для логитов
- методы генерации текста (с top-k / top-p семплированием)
- метод обучения fit() с валидацией и сохранением чекпоинтов
"""

# преходим к созданию класса GPT
class GPT(nn.Module):
    """
    Генеративная трансформер-модель (архитектура GPT)

    Args:
        vocab_size (int): размер словаря
        max_seq_len (int): максимальная длина последовательности
        emb_size (int): размер эмбеддингов
        num_heads (int): количество голов внимания
        head_size (int): размер каждой головы
        num_layers (int): количество слоёв (декодеров)
        dropout (float): вероятность dropout (по умолчанию 0.1)
        device (str): "cpu" или "cuda" (по умолчанию "cpu")

    Методы:
        forward(x): прямой проход, возвращает логиты
        generate(x, max_new_tokens, do_sample, temperature, top_k, top_p): генерация текста
        fit(train_loader, valid_loader, num_epoch, learning_rate): обучение модели
        top_k_funk(logits, top_k): фильтрация top-k (статический метод)
        top_p_funk(logits, top_p): фильтрация nucleus (top-p, статический метод)
    """
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 emb_size: int,
                 num_heads: int,
                 head_size: int,
                 num_layers: int,
                 dropout = 0.1,
                 device = "cpu"):

        super(GPT, self).__init__()
        self.max_seq_len = max_seq_len
        self.TokenEmbeddings = TokenEmbeddings(vocab_size, emb_size)
        self.PositionalEmbeddings = PositionalEmbeddings(max_seq_len, emb_size)
        self.dropout_1 = nn.Dropout(dropout)

        # создаем блоки декодера в колличестве num_layers
        """
        # в данном месте мы создаем все декодеры используя nn.ModuleList(),
        # но элегантнее использовать nn.Sequential, реализация ниже
        self.decoder_list = nn.ModuleList()
        for _ in range(num_layers):
            current_decoder = Decoder(num_heads=num_heads,
                    emb_size=emb_size,
                    head_size=head_size,
                    max_seq_len=max_seq_len,
                    dropout=dropout)
            self.decoder_list.append(current_decoder)
        """
        # создаем бол декодеров используя nn.Sequential() и с распаковкой списка
        self.decoder_block = nn.Sequential(*[
            Decoder(num_heads=num_heads,
                    emb_size=emb_size,
                    head_size=head_size,
                    max_seq_len=max_seq_len,
                    dropout=dropout) for _ in range(num_layers)
        ])

        self.linear_end = nn.Linear(emb_size, vocab_size)

    def forward(self, x: torch.Tensor):
        # создаем эмбеддинги, токенов и позиционные
        token_embeddings_tensor = self.TokenEmbeddings.forward(x)
        positional_embeddings_tensor = self.PositionalEmbeddings.forward(x.shape[1])

        # складываем эмбединги и применяем dropout
        full_embeddings = token_embeddings_tensor + positional_embeddings_tensor
        full_embeddings_dropout = self.dropout_1(full_embeddings)

        # пропускаем эмбединги через все слои декодера
        tensor_decodered = self.decoder_block(full_embeddings_dropout)

        # пропускаем выходные данные из блока декодера через линейный слой
        logits = self.linear_end(tensor_decodered)

        return(logits)


    def generate(self, x: torch.tensor, max_new_tokens: int, do_sample = False,
                 temperature = 1.0, top_k = None, top_p = None):
        for _ in range(max_new_tokens):

            # обрезаем последовательность (берем только последние токены)
            current_x = x[:, - self.max_seq_len: ]

            # прогоняем токены через нейросеть
            logits = self.forward(current_x)

            # берем последний логит у каждого батча
            end_logits = logits[:,-1, :]

            # применяем температуру
            if temperature != 0:
                end_logits = end_logits / temperature

            # применяем логику использования top_k и top_p
            # сначала применяем top_k
            if (top_k is not None) and do_sample == True:
                end_logits = GPT.top_k_funk(end_logits, top_k = top_k)

            # теперь применяем top_p
            if (top_p is not None) and do_sample == True and top_p != 1:
                end_logits = GPT.top_p_funk(end_logits, top_p)

            # прогоняем последние логиты через softmax
            end_logits_softmax = torch.softmax(end_logits, dim=-1)

            # срабатывает если не нужно применять семплирование, или если температура равна ==0
            if do_sample == False or temperature == 0:
                # берем индекс максимальрного логита, сохраняя размерность
                next_token = torch.argmax(end_logits_softmax, dim=-1, keepdim=True)

            else: # применяем случайное сэмплирование
                # для каждого батча (строки) семплируем 1 токен из распределения вероятностей
                next_token = torch.multinomial(end_logits_softmax, num_samples=1)
                # next_token уже имеет размерность (batch_size, 1)

            # добавляем последний логит в тензор
            x = torch.cat([x, next_token], dim = -1)

        return x

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader,
            num_epoch: int, learning_rate: float):

        # определяем device и помещаем модель на нужный девайс
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self.to(device)

        # создаем оптимизатор Adam
        optimizer = Adam(self.parameters(), lr = learning_rate)

        # начинаем цикл по эпохам
        for epoch in range(num_epoch):
            print(f"Epoch {epoch + 1}/{num_epoch}")

            self.train() # переводим в тренировочный режим

            train_losses = [] # для хранения лоссов на каждом батче

            # Цикл по тренировочному датасету
            for batch_idx, (inputs, targets) in enumerate(train_loader):

                # помещаем входные и выходные данные на device (их нужно помещать отдельноот самой модели)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # пропускаем входные данные через нейросеть
                logits_tensor = self.forward(inputs)

                # приводим тензоры с данными к нужной размерности
                logits_flatten = logits_tensor.flatten(0,1)
                targets_flatten = targets.flatten()

                # посчитаем функцию потерь кросс энтропию, и сохраним в переменную
                loss = F.cross_entropy(logits_flatten, targets_flatten)

                # сохраняем loss
                train_losses.append(loss.item())

                # обратный проход (backward pass)
                optimizer.zero_grad() # обнуляем градиенты
                loss.backward()       # вычисляем градиенты
                optimizer.step()      # шаг градиентного спуска

                # Выводим прогресс (локально)
                if (batch_idx + 1) % 50 == 0:
                    print(f"[Train] Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

                # Средний тренировочный лосс за эпоху
                avg_train_loss = sum(train_losses) / len(train_losses)
                print(f"\n[Train] Average Loss: {avg_train_loss:.4f}")


            # Переводим модель в режим оценки
            self.eval()

            valid_losses = [] # будем хранить лосы на валидации

            # отключаем вычисление градиента
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(valid_loader):

                    # переводим данные на device
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # пропускаем валидационные данные через модель
                    valid_logits = self.forward(inputs)

                    # меняем размерности тензоров
                    valid_flatten = valid_logits.flatten(0,1)
                    val_targets_flatten = targets.flatten()

                    # вычисляем лосс на валидационной подвыборке
                    valid_loss = F.cross_entropy(valid_flatten, val_targets_flatten)

                    # добавляем лосс в массив
                    valid_losses.append(valid_loss.item()) # item() - выделяет число из тензора

                    # Выводим прогресс (локально)
                    if (batch_idx + 1) % 50 == 0:
                        print(f"[Valid] Batch {batch_idx + 1}/{len(valid_loader)}, Loss: {loss.item():.4f}")



            # Средний валидационный лосс за эпоху
            avg_valid_loss = sum(valid_losses) / len(valid_losses)
            print(f"\n[Valid] Average Loss: {avg_valid_loss:.4f}")

            # Сохраняем текущую версию модели (локально)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'valid_loss': avg_valid_loss
            }
            torch.save(checkpoint, f'model_epoch_{epoch + 1}.pt')
            print(f"[Save] Model saved as 'model_epoch_{epoch + 1}.pt'")

            # Небольшая задержка между эпохами для читаемости вывода
            print()



    @staticmethod
    def top_k_funk(logits: torch.tensor, top_k):
        logits_copy = logits.clone()

        top_k_values, _ = torch.topk(logits_copy, top_k, dim = -1)
        # Добавляем unsqueeze(-1) чтобы получить форму [batch_size, 1]
        min_top_k = top_k_values[:, -1].unsqueeze(-1)  # или [:, -1:]

        logits_copy[logits_copy < min_top_k] = -float("inf")
        return logits_copy

    @staticmethod
    def top_p_funk(logits, top_p):
        logits_copy = logits.clone()

        # сортируем логиты по убыванию-(descending)
        sorted_logits, sorted_index = torch.sort(logits_copy, descending=True)

        # softmax для получения вероятностей
        sorted_probs = torch.softmax(sorted_logits, dim = 1)

        # Кумулятивная сумма
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Маска для токенов, которые оставляем (кумулятивная сумма <= top_p)
        # Всегда оставляем хотя бы один токен
        mask = cumulative_probs <= top_p

        mask[:, 0] = 1  # Всегда оставляем первый (самый вероятный)

        sorted_logits[~mask] = -float('inf')

        # востанавливаем исходный порядок
        logits_copy = torch.zeros_like(sorted_logits).scatter_(-1,
            sorted_index, sorted_logits )

        return logits_copy


    '''
    def generate(self, x: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            current_x = x[:, - self.max_seq_len:]
            logits = self.forward(current_x)
            logits_softmax = torch.softmax(logits, dim=-1)
            end_logits = logits_softmax[:, -1, :]
            next_token = torch.argmax(end_logits, dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)

        return x
    '''




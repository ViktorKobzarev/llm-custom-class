# llm-custom-class

# Пользовательский класс для создания и обучения LLM

## Описание

Реализация GPT-подобной модели с нуля на PyTorch. Включает:

- BPE токенизатор (обучение с нуля)
- Механизм внимания (одноголовое + многоголовое)
- Decoder блок (MultiHeadAttention + FeedForward)
- Обучаемые позиционные эмбеддинги
- Методы генерации с top-k и top-p семплированием

## Структура

| Файл | Назначение |
|------|------------|
| `GPT.py` | Основной класс GPT, forward(), generate(), fit() |
| `attention.py` | HeadAttention, MultiHeadAttention |
| `decoder.py` | Decoder (блок трансформера) |
| `Feed_Forward_Network.py` | FeedForward (FFN) |
| `token_embeddings.py` | TokenEmbeddings, PositionalEmbeddings |
| `LLM_BPE..py` | BPE токенизатор |
| `DataLoader.py` | GetData (Dataset для DataLoader) |

## Требования

- Python 3.8+
- Установка: `pip install -r requirements.txt`

## Краткий пример

```python
from gpt import GPT

# Создание модели
model = GPT(
    vocab_size=5000,
    max_seq_len=256,
    emb_size=256,
    num_heads=8,
    head_size=32,
    num_layers=6,
    dropout=0.1
)

# Генерация текста
input_ids = tokenizer.encode("Привет")
output = model.generate(input_ids, max_new_tokens=50)

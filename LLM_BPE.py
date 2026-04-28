
"""
Модуль BPE (Byte Pair Encoding) токенизатора

Содержит класс BPE для обучения токенизатора с нуля и кодирования/декодирования текста.
Реализует базовый алгоритм BPE без использования внешних библиотек.

Принцип работы:
1. Начинаем с отдельных символов
2. Итеративно находим самую частую пару соседних токенов
3. Объединяем её в новый токен
4. Повторяем до достижения целевого размера словаря
"""

class BPE:
    """
        Токенизатор на основе алгоритма Byte Pair Encoding

        Args:
            vocab_size (int): целевой размер словаря (количество токенов)

        Методы:
            fit(text_sample): обучение токенизатора на текстовом корпусе
            encode(text_sample): преобразование текста в список ID токенов
            decode(token_ids): преобразование списка ID токенов обратно в текст

        Атрибуты (создаются в процессе обучения):
            token2id (dict): токен -> ID
            id2token (dict): ID -> токен
            group_dict (dict, опционально): группировка токенов по первому символу для ускорения кодирования
        """
    # инициализация
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size # размер словаря

    # обучение токенизатора
    def fit(self, text_sample: str): # text - корпус текста для обучения

        unique_chars = set() # все уникальные символы
        for i in text_sample:
            unique_chars.add(i)

        # список всех уникальных символов в тексте отсортированый по алфавиту
        lst_unique_chars = list(sorted(unique_chars))

        text_simvoly = list(text_sample) # разбиваем текст на отдельные символы

        # Находим все токены в цикле
        while len(lst_unique_chars) < self.vocab_size:

            # считаем количество пар символов стоящих рядом
            slov_simvolov_rydom = dict()
            for j in range(len(text_simvoly) - 1):
                if "".join(text_simvoly[j: j+2]) in slov_simvolov_rydom:
                    slov_simvolov_rydom["".join(text_simvoly[j: j+2])] += 1
                else:
                    slov_simvolov_rydom["".join(text_simvoly[j: j+2])] = 1


            #set_new_token = set()
            max_count_simv = ["", 0]
            for sim, count in slov_simvolov_rydom.items():
                if count > max_count_simv[1]:
                    max_count_simv[0] = sim
                    max_count_simv[1] = count

            lst_unique_chars.append(max_count_simv[0])

            for i in range(len(text_simvoly) - 1):
                if "".join(text_simvoly[i: i+2]) == max_count_simv[0]:
                    text_simvoly[i: i+2]  = [max_count_simv[0]]


        # После того как мы получили все токены, создаем 2 словаря
        # кодирующий символы в токены, и обратный

        # начальный
        self.token2id = dict()
        numb = 0
        for simv in lst_unique_chars:
            self.token2id[simv] = numb
            numb += 1

        # обратный
        self.id2token = dict()
        for simv, numbr in self.token2id.items():
            self.id2token[numbr] = simv

    def encode(self, text_sample: str):

        # создаем словарь в котором сгрупированы ключи токенайзера для энкодера
        if hasattr(self, "group_dict"):
            pass
        else:
            self.group_dict = dict()

        for token in self.token2id:
            if token[0] in self.group_dict:
                self.group_dict[token[0]].append(token)
            else:
                self.group_dict[token[0]] = []
                self.group_dict[token[0]].append(token)


        # создали сгруппированый словарь, теперь ищем самый подходящий максимальный токен
        lst_text = list(text_sample)
        cur_ind = 0
        while True:
            if cur_ind >= len(lst_text):
                break

            mast_token = ""
            list_fit_token = self.group_dict[lst_text[cur_ind]]
            for tok in list_fit_token:
                srav = "".join(lst_text[cur_ind:  cur_ind + len(tok)])
                if len(tok) > len(mast_token) and tok == srav:
                    mast_token = tok

            lst_text[cur_ind: cur_ind + len(mast_token)] = [mast_token]
            cur_ind += 1

        finaly_token_list = []
        for i in lst_text:
            finaly_token_list.append(self.token2id[i])

        return finaly_token_list

    def decode(self, token_ids: list[int]):

        lst_token = []
        for i in token_ids:
            lst_token.append(self.id2token[i])

        return "".join(lst_token)


# Импортируем необходимые библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split

# Шаг 1: Создание подмножества из большого датасета
print("Шаг 1: Создание уменьшенного датасета")

# Путь к большому исходному файлу
large_file_path = r'C:\Users\User\Desktop\Get_n_mark_data\task9\sentiment140.csv'

# Считываем первые 1000 строк из большого файла
df = pd.read_csv(large_file_path, nrows=1000, encoding='ISO-8859-1', engine='python', on_bad_lines='skip', header=None)

# Задаем правильные имена столбцов вручную
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Оставляем только нужные столбцы для дальнейшей работы
df = df[['sentiment', 'text']]

# Преобразуем метку настроения в читаемый формат (0 = negative, 4 = positive)
df['sentiment'] = df['sentiment'].replace({0: 'negative', 4: 'positive'})

# Шаг 2: Разделение датасета на две части — для ручной и автоматической разметки
print("Шаг 2: Разделение на зоны автоматической и ручной разметки")

# Делим на два поднабора: 80% — для автоматической разметки, 20% — для ручной разметки
df_auto, df_manual = train_test_split(df, test_size=0.2, random_state=42)

# Сохраняем подмножество для ручной разметки
manual_file_path = r'C:\Users\User\Desktop\Get_n_mark_data\task9\sentiment140_manual.csv'
df_manual.to_csv(manual_file_path, index=False, encoding='utf-8')
print(f"Файл для ручной разметки сохранен: '{manual_file_path}'")

# Шаг 3: Автоматическая разметка на основе правил для подмножества
print("Шаг 3: Автоматическая разметка на основе правил")

# Определяем набор положительных и отрицательных слов для разметки
positive_words = {'good', 'great', 'fantastic', 'happy', 'love'}
negative_words = {'bad', 'terrible', 'sad', 'hate', 'angry'}

# Функция для разметки на основе ключевых слов
def rule_based_label(text):
    text = text.lower()
    if any(word in text for word in positive_words):
        return 'positive'
    elif any(word in text for word in negative_words):
        return 'negative'
    else:
        return 'neutral'

# Применение функции разметки на основе правил
df_auto['rule_based_sentiment'] = df_auto['text'].apply(rule_based_label)

# Сохранение автоматически размеченного датасета
auto_file_path = r'C:\Users\User\Desktop\Get_n_mark_data\task9\sentiment140_auto.csv'
df_auto.to_csv(auto_file_path, index=False, encoding='utf-8')

print(f"Файл с автоматически размеченными данными сохранен: '{auto_file_path}'")

# Финальный вывод
print("Подготовка данных завершена: \n- Файл для ручной разметки: sentiment140_manual.csv\n- Файл с автоматической разметкой: sentiment140_auto.csv")
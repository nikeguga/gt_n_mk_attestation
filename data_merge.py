import pandas as pd
import json

# 1. Пути к файлам
auto_csv_path = 'C:\\Users\\User\\Desktop\\Get_n_mark_data\\task9\\sentiment140_auto.csv'
manual_json_path = 'C:\\Users\\User\\Desktop\\Get_n_mark_data\\task9\\project-4-at-2024-10-01-19-43-405cae00.json'
output_file_path = 'C:\\Users\\User\\Desktop\\Get_n_mark_data\\task9\\final_merged_three_class_sentiment_dataset.csv'

# 2. Чтение и обработка JSON-файла (ручная разметка)
with open(manual_json_path, 'r', encoding='utf-8') as file:
    manual_data = json.load(file)

# Извлекаем данные из JSON и создаем DataFrame
manual_entries = []
for entry in manual_data:
    sentiment = entry['data']['sentiment']
    text = entry['data']['text']
    choices = entry['annotations'][0]['result'][0]['value']['choices'][0].lower()  # Преобразование в нижний регистр
    manual_entries.append({'sentiment': sentiment, 'text': text, 'label': choices})

# Создаем DataFrame из JSON данных
df_manual = pd.DataFrame(manual_entries)

# 3. Чтение и обработка CSV-файла с автоматической разметкой
df_auto = pd.read_csv(auto_csv_path, encoding='utf-8', engine='python')
df_auto = df_auto[['sentiment', 'text', 'rule_based_sentiment']].rename(columns={'rule_based_sentiment': 'label'})

# 4. Объединение двух DataFrame
merged_df = pd.concat([df_auto, df_manual], ignore_index=True)

# 5. Сохранение объединенного файла в CSV
merged_df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Объединенный файл успешно сохранен по пути: {output_file_path}")

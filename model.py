import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Загрузка данных
data_path = 'C:\\Users\\User\\Desktop\\Get_n_mark_data\\task9\\final_merged_three_class_sentiment_dataset.csv'
df = pd.read_csv(data_path, encoding='utf-8', engine='python')

# 2. Подготовка данных
# Проверяем, что нужные колонки присутствуют
if 'text' in df.columns and 'label' in df.columns:
    X = df['text']  # Тексты сообщений
    y = df['label'].astype(str)  # Метки разметок (positive, negative, neutral)

# 3. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Преобразование текстов в числовое представление
vectorizer = TfidfVectorizer(max_features=5000)  # Можно изменить параметр max_features для экспериментов
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Обучение модели
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train_vec, y_train)

# 6. Предсказание и оценка
y_pred = model.predict(X_test_vec)

# 7. Вывод результатов
print("Качество модели на тестовых данных:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Сохранение модели и векторизатора
import joblib
model_filename = 'C:\\Users\\User\\Desktop\\Get_n_mark_data\\task9\\sentiment_model.pkl'
vectorizer_filename = 'C:\\Users\\User\\Desktop\\Get_n_mark_data\\task9\\tfidf_vectorizer.pkl'
joblib.dump(model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)
print(f"Модель и векторизатор успешно сохранены в: {model_filename} и {vectorizer_filename}")

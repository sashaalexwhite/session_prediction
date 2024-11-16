import logging
import os
import dill
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    filename='predict.log',
                    filemode="w")

# Определение пути
path = os.environ.get("PROJECT_PATH", ".")

# Загрузка модели, энкодера и скалера
def load_model():
    model_dir = f'{path}/data/modele'
    model_files = sorted([file for file in os.listdir(model_dir) if file.endswith('.pkl') and 'model' in file])
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    latest_model_file = model_files[-1]
    latest_model_path = os.path.join(model_dir, latest_model_file)
    logging.info(f"Loading model from {latest_model_path}")
    with open(latest_model_path, 'rb') as file:
        model = dill.load(file)
    logging.info(f"Model loaded: {type(model)} - {latest_model_path}")

    return model

model = load_model()

encoder_path = f'{path}/data/modele/encoder.pkl'
scaler_path = f'{path}/data/modele/scaler.pkl'

encoder = joblib.load(encoder_path)
logging.info(f"Encoder loaded: {type(encoder)}")
scaler = joblib.load(scaler_path)
logging.info(f"Scaler loaded: {type(scaler)}")

# Функция для обработки данных
def process_data(file_path, encoder, scaler):
    data = pd.read_parquet(file_path)
    session_client_data = data[['session_id', 'client_id']].copy()
    if 'conversion_rate' in data.columns:
        data = data.drop(['session_id', 'client_id', 'conversion_rate'], axis=1, errors='ignore')
    else:
        data = data.drop(['session_id', 'client_id'], axis=1)

    if 'visit_number' in data.columns:
        data['visit_number'] = data['visit_number'].astype('int64')

    string_columns = data.select_dtypes(include=['object']).columns
    encoded_strings = encoder.transform(data[string_columns])
    data_numeric = data.drop(string_columns, axis=1)

    data_encoded = hstack([data_numeric, encoded_strings])
    data_encoded = csr_matrix(data_encoded)

    # Применение скалера
    data_encoded = scaler.transform(data_encoded)

    return data_encoded, session_client_data

# Функция предсказания
def predict():
    folder_path = f'{path}/data/common_data'  # Папка с файлами для предсказания
    output_folder = f'{path}/data/predictions'  # Папка для сохранения результатов предсказания

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.parquet'):
            file_path = os.path.join(folder_path, file_name)
            logging.info(f"Обработка файла: {file_path}")
            X, session_client_data = process_data(file_path, encoder, scaler)
            probabilities = model.predict_proba(X)[:, 1]
            predictions = model.predict(X)

            # Добавляем колонки с вероятностями и классификацией
            data = pd.read_parquet(file_path)
            data = data.drop('conversion_rate', axis=1, errors='ignore')

            # Условия классификации
            conditions = [
                (probabilities >= 0.70) & (predictions == 1),  # True Positive (TP)
                (probabilities >= 0.20) & (probabilities < 0.70) & (predictions == 0),  # False Positive (FP)
                (probabilities < 0.20) & (predictions == 0),  # True Negative (TN)
                (probabilities >= 0.20) & (probabilities < 0.70) & (predictions == 1)  # False Negative (FN)
            ]
            choices = ['TP', 'FP', 'TN', 'FN']

            data['predicted_class'] = np.select(conditions, choices, default='Unknown')

            # Восстанавливаем session_id и client_id
            data['session_id'] = session_client_data['session_id']
            data['client_id'] = session_client_data['client_id']

            # Сохраняем результат
            output_path = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}_with_predictions.parquet')
            data.to_parquet(output_path)
            logging.info(f"Предсказания сохранены: {output_path}")

if __name__ == "__main__":
    predict()





import sys
import sklearn
print(f"Scikit-learn version: {sklearn.__version__}")

import logging
import os
import dill
import pandas as pd
from datetime import datetime
import joblib
from scipy.sparse import hstack, csr_matrix, vstack
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

# Настройка логирования
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(message)s',
                    filename='pipeline.log',
                    filemode='w')

path = os.environ.get("PROJECT_PATH", ".")

def process_data(file_path, encoder=None):
    data = pd.read_parquet(file_path)
    session_client_data = data[['session_id', 'client_id']].copy()
    data = data.drop(['session_id', 'client_id'], axis=1)
    y = data['conversion_rate'].values  # Извлекаем целевую переменную
    data = data.drop(columns=['conversion_rate'])  # Удаляем её из признаков
    string_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns

    # Обработка категориальных признаков
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        encoded_strings = encoder.fit_transform(data[string_columns])
    else:
        encoded_strings = encoder.transform(data[string_columns])

    # Обработка временных признаков
    date_columns = data.select_dtypes(include=['datetime64[ns]']).columns
    for col in date_columns:
        data[col] = data[col].astype('int64') // 10 ** 9

    data_numeric = data[numeric_columns]
    data_encoded = hstack([data_numeric, encoded_strings])
    data_encoded = csr_matrix(data_encoded)
    X = data_encoded
    return X, y, session_client_data, date_columns, data, encoder


def train_model(X_train, y_train):
    models = {
        'XGBoost': XGBClassifier(tree_method='hist', objective='binary:logistic'),
        'Ridge Classifier': RidgeClassifier(random_state=42),
        'LGBMClassifier': LGBMClassifier(random_state=42, force_row_wise=True)
    }

    best_model = None
    best_score = 0
    best_model_name = ""

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_score = scores.mean()
        logging.info(f"{name} ROC-AUC: {mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_model_name = name

    best_model.fit(X_train, y_train)
    return best_model, best_model_name


def pipeline():
    logging.info('Начало выполнения функции pipeline')

    # Обучаем модель на первых трех частях данных и выбираем лучшую модель
    X_train, y_train, _, _, _, encoder = process_data(f'{path}/data/common_data/mod-0.parquet')
    X_train2, y_train2, _, _, _, _ = process_data(f'{path}/data/common_data/mod-1.parquet', encoder)
    X_train3, y_train3, _, _, _, _ = process_data(f'{path}/data/common_data/mod-2.parquet', encoder)

    # Объединяем данные для выбора лучшей модели
    X_train_full = vstack([X_train, X_train2, X_train3])
    y_train_full = np.concatenate([y_train, y_train2, y_train3])

    # Масштабирование данных
    scaler = StandardScaler(with_mean=False)
    X_train_full = scaler.fit_transform(X_train_full)

    # Выбираем лучшую модель на первых трех частях данных
    best_model, best_model_name = train_model(X_train_full, y_train_full)
    logging.info(f'Лучшая модель: {best_model_name}')

    # Поочередное дообучение модели на частях с 4-й по 19-ю
    for i in range(3, 19):
        file_path = f'{path}/data/common_data/mod-{i}.parquet'
        if not os.path.exists(file_path):
            logging.warning(f'Файл не найден: {file_path}')
            continue
        X, y, _, _, _, _ = process_data(file_path, encoder)
        X = scaler.transform(X)
        best_model.fit(X, y)

    # Определяем пути
    scaler_path = f"{path}/data/modele/scaler.pkl"
    encoder_path = f"{path}/data/modele/encoder.pkl"
    model_filename = f'{path}/data/modele/model_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'

    # Сохраняем масштабировщик
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)

    with open(model_filename, 'wb') as file:
        dill.dump(best_model, file)

    # Логируем информацию
    logging.info(f"Scaler сохранен в {scaler_path}")
    logging.info(f"Лучшая модель ({best_model_name}) сохранена в {model_filename}")
    logging.info(f"Энкодер сохранен в {encoder_path}")
    logging.info('Конец выполнения функции pipeline')


if __name__ == "__main__":
    pipeline()
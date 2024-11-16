import pandas as pd
import os
import json
import re
import logging
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename="union.log",
                    filemode="w")

path = os.environ.get("PROJECT_PATH", ".")

# Путь к папке с файлами
folder_path = f'{path}/data/n_file'
output_path = f'{path}/data/n_m_file'


# Функция для извлечения даты из имени файла
def extract_date(file_name):
    match = re.search(r'\d{4}-\d{2}-\d{2}', file_name)
    return match.group(0) if match else None


# Проверка, пуст ли файл
def is_empty(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return len(data) == 0


# Функция для обработки JSON данных в DataFrame
def json_to_dataframe(json_data):
    flattened_data = []

    def extract_data(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        flattened_data.append(item)
                elif isinstance(value, dict):
                    extract_data(value)

    extract_data(json_data)
    return pd.DataFrame(flattened_data)


# Функция для обработки файлов
def process_files(file1, file2, output_file):
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
        if data1 and data2:
            logging.info(f"Обработка файлов: {file1} и {file2}")
            # Преобразование JSON данных в DataFrame
            df1 = json_to_dataframe(data1)
            df2 = json_to_dataframe(data2)
            if 'session_id' not in df1.columns or 'session_id' not in df2.columns:
                logging.error(f"Колонка 'session_id' отсутствует в одном из файлов: {file1}, {file2}")
                return
            # Объединение файлов
            df12 = pd.merge(df1, df2, on='session_id', how='outer')
            # Заполнение NaN значением 'unknown' для строковых колонок и 0 для числовых колонок
            for col in df12.columns:
                if df12[col].dtype == 'int64' or df12[col].dtype == 'float64':
                    df12[col] = df12[col].fillna(0)
                else:
                    df12[col] = df12[col].fillna('unknown')
            df12 = df12.infer_objects(copy=False)
            if 'hit_number' in df12.columns:
                df12['hit_number'] = df12['hit_number'].astype('int64')
            if 'visit_number' in df12.columns:
                df12['visit_number'] = df12['visit_number'].astype('int64')

            # Удаление ненужных колонок, если они существуют
            columns_to_drop = [
                'hit_time', 'hit_type', 'hit_referer', 'event_value', 'date_x', 'visit_date',
                'utm_keyword', 'device_os', 'device_model', 'device_screen_resolution', 'date_y'
            ]
            df12 = df12.drop(columns=[col for col in columns_to_drop if col in df12.columns])

            # Категориальные колонки
            categorical_columns = [
                'hit_date', 'hit_page_path', 'event_category', 'event_action', 'event_label', 'visit_time',
                'utm_medium', 'device_category', 'device_browser', 'geo_country', 'geo_city', 'client_id',
                'utm_source', 'utm_campaign', 'utm_adcontent', 'device_brand'
            ]

            # Агрегация данных
            agg_dict = {col: lambda x: x.unique().tolist() for col in categorical_columns}
            agg_dict['hit_number'] = 'max'
            agg_dict['visit_number'] = 'max'

            # Группировка данных по session_id
            grouped_df = df12.groupby('session_id').agg(agg_dict).reset_index()

            # Преобразование списков в строки с удалением лишних запятых и пустых строк
            for col in categorical_columns:
                grouped_df[col] = grouped_df[col].apply(lambda x: ', '.join(filter(None, map(str.strip, map(str, x)))))

            # Заполнение пустых строк значением 'unknown'
            grouped_df.replace('', 'unknown', inplace=True)

            # Выстраивание колонок в заданном порядке
            column_order = [
                'session_id', 'hit_date', 'hit_number', 'hit_page_path', 'event_category', 'event_action',
                'event_label', 'client_id', 'visit_time', 'visit_number', 'utm_source', 'utm_medium', 'utm_campaign',
                'utm_adcontent', 'device_category', 'device_brand', 'device_browser', 'geo_country', 'geo_city'
            ]
            grouped_df = grouped_df[column_order]

            # Сохранение результата в формате Parquet
            grouped_df.to_parquet(output_file)
            logging.info(f"Файл успешно сохранен: {output_file}")
        else:
            logging.warning(f"Один или оба файла пусты: {file1}, {file2}")
    except Exception as e:
        logging.error(f"Ошибка при обработке файлов: {file1}, {file2}. Ошибка: {str(e)}")


# Функция для обработки всех файлов
def union():
    # Получаем список всех файлов в папке
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    # Группируем файлы по дате
    files_by_date = {}
    for file_name in all_files:
        date = extract_date(file_name)
        if date:
            if date not in files_by_date:
                files_by_date[date] = {'hits': None, 'sessions': None}
            if 'hits' in file_name:
                files_by_date[date]['hits'] = file_name
            elif 'sessions' in file_name:
                files_by_date[date]['sessions'] = file_name

    # Проверка структуры files_by_date
    logging.info(f"Файлы сгруппированы по дате")

    # Обработка всех пар файлов
    for date, files in files_by_date.items():
        if isinstance(files, dict) and files.get('hits') and files.get('sessions'):
            file1 = os.path.join(folder_path, files['hits'])
            file2 = os.path.join(folder_path, files['sessions'])
            if not is_empty(file1) and not is_empty(file2):
                output_file = os.path.join(output_path, f'combined_{date}.parquet')
                process_files(file1, file2, output_file)


if __name__ == '__main__':
    union()

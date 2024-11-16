import logging
from google.cloud import storage, bigquery
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='adding_to_the_db.log',
                    filemode='w')

path = os.environ.get("PROJECT_PATH", ".")

# Установка пути к файлу учетных данных
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"{path}/credentials/deep-ethos-436810-c7-32b7bed8002f.json"

# Инициализация клиентов Google Cloud Storage и BigQuery
storage_client = storage.Client()
bigquery_client = bigquery.Client()

# Название вашего ведра и таблицы BigQuery
bucket_name = 'basa_data_diplom'
dataset_id = 'my_dataset'
table_id = 'diplom_bd'
bucket = storage_client.bucket(bucket_name)

# Путь к локальной папке с обработанными файлами
local_folder = f'{path}/data/predictions'

def adding_to_the_db():
    logging.info('Начало выполнения функции adding_to_the_db')
    # Загрузка каждого файла из локальной папки в ведро и добавление данных в BigQuery
    for filename in os.listdir(local_folder):
        if filename.endswith('.parquet'):  # Предполагается, что файлы в формате Parquet
            local_file_path = os.path.join(local_folder, filename)
            blob = bucket.blob(filename)
            try:
                # Загрузка файла в ведро
                blob.upload_from_filename(local_file_path)

                # Загрузка данных из файла во временную таблицу в BigQuery
                temp_table_id = f"{table_id}_temp"
                table_ref = bigquery_client.dataset(dataset_id).table(temp_table_id)
                job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET)
                with open(local_file_path, "rb") as source_file:
                    load_job = bigquery_client.load_table_from_file(source_file, table_ref, job_config=job_config)
                load_job.result()  # Ожидание завершения загрузки

                # Вставка новых данных в основную таблицу
                query = f"""
                    INSERT INTO `{dataset_id}.{table_id}`
                    SELECT
                        session_id,
                        ANY_VALUE(CAST(client_id AS STRING)) AS client_id,
                        MAX(hit_date) AS hit_date,
                        MAX(hit_number) AS hit_number,
                        STRING_AGG(DISTINCT hit_page_path, ', ') AS hit_page_path,
                        STRING_AGG(DISTINCT event_category, ', ') AS event_category,
                        STRING_AGG(DISTINCT event_action, ', ') AS event_action,
                        STRING_AGG(DISTINCT event_label, ', ') AS event_label,
                        ANY_VALUE(visit_time) AS visit_time,
                        MAX(visit_number) AS visit_number,
                        STRING_AGG(DISTINCT utm_source, ', ') AS utm_source,
                        STRING_AGG(DISTINCT utm_medium, ', ') AS utm_medium,
                        STRING_AGG(DISTINCT utm_campaign, ', ') AS utm_campaign,
                        STRING_AGG(DISTINCT utm_adcontent, ', ') AS utm_adcontent,
                        STRING_AGG(DISTINCT device_category, ', ') AS device_category,
                        STRING_AGG(DISTINCT device_brand, ', ') AS device_brand,
                        STRING_AGG(DISTINCT device_browser, ', ') AS device_browser,
                        STRING_AGG(DISTINCT geo_country, ', ') AS geo_country,
                        STRING_AGG(DISTINCT geo_city, ', ') AS geo_city,
                        ANY_VALUE(predicted_class) AS predicted_class
                    FROM `{dataset_id}.{temp_table_id}`
                    WHERE session_id NOT IN (SELECT session_id FROM `{dataset_id}.{table_id}`)
                    GROUP BY session_id
                """
                query_job = bigquery_client.query(query)
                query_job.result()  # Ожидание завершения запроса

                # Удаление временной таблицы
                bigquery_client.delete_table(table_ref)

                # Удаление локального файла
                os.remove(local_file_path)

            except Exception as e:
                os.remove(local_file_path)
                logging.error(f'Ошибка при обработке файла {filename}: {e}')

    logging.info("Все файлы успешно загружены в Google Cloud Storage и BigQuery.Локальная папка стёрта ")

if __name__ == '__main__':
    adding_to_the_db()



import logging
from google.cloud import storage, bigquery
import os

from dags.session_prediction_v2_main import add_n_m_file_task

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='adding_to_the_db.log',
                    filemode='w')

path = os.environ.get("PROJECT_PATH", ".")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"{path}/credentials/deep-ethos-436810-c7-32b7bed8002f.json"


storage_client = storage.Client()
bigquery_client = bigquery.Client()

bucket_name = 'basa_data_diplom'
dataset_id = 'my_dataset'
table_id = 'diplom_bd'
bucket = storage_client.bucket(bucket_name)

local_folder = f'{path}/data/predictions'

def add_n_m_file():
    logging.info('Starting a Function add_n_m_file')
    for filename in os.listdir(local_folder):
        if filename.endswith('.parquet'):  
            local_file_path = os.path.join(local_folder, filename)
            blob = bucket.blob(filename)
            try:
                blob.upload_from_filename(local_file_path)
                temp_table_id = f"{table_id}_temp"
                table_ref = bigquery_client.dataset(dataset_id).table(temp_table_id)
                job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET)
                with open(local_file_path, "rb") as source_file:
                    load_job = bigquery_client.load_table_from_file(source_file, table_ref, job_config=job_config)
                load_job.result()  

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
                query_job.result()  

                bigquery_client.delete_table(table_ref)
                os.remove(local_file_path)

            except Exception as e:
                os.remove(local_file_path)
                logging.error(f'Error processing the file {filename}: {e}')

    logging.info("All files have been successfully uploaded to the Google Cloud Storage and BigQuery.Локальная папка стёрта ")

if __name__ == '__main__':
    add_n_m_file()

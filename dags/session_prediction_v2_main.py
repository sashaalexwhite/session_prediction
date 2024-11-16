import os
import sys
import datetime as dt
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor

path = "/opt/airflow"
os.environ["PROJECT_PATH"] = path
sys.path.insert(0, path)

default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2024, 11, 14),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
    'execution_timeout': dt.timedelta(minutes=60),
}

with DAG(
    dag_id='session_prediction_v2_main',
    default_args=default_args,
    schedule=None
) as dag:

    def pipeline_wrapper():
        from modules.pipeline import pipeline
        pipeline()

    def predict_wrapper():
        from modules.predict import predict
        predict()

    def adding_to_the_db_wrapper():
        from modules.adding_to_the_db import adding_to_the_db
        adding_to_the_db()

    def union_wrapper():
        from modules.union import union
        union()

    def pred_n_m_file_wrapper():
        from modules.pred_n_m_file import pred_n_m_file
        pred_n_m_file()

    def add_n_m_file_wrapper():
        from modules.add_n_m_file import add_n_m_file
        add_n_m_file()

    pipeline_task = PythonOperator(
        task_id='pipeline_task',
        python_callable=pipeline_wrapper,
    )

    wait_for_pipeline_task = ExternalTaskSensor(
        task_id='wait_for_pipeline_task',
        external_dag_id='session_prediction_v2_main',
        external_task_id="pipeline_task",
        mode='reschedule',
        timeout=60*60,
    )

    predict_task = PythonOperator(
        task_id='predict_task',
        python_callable=predict_wrapper,
    )

    wait_for_predict_task = ExternalTaskSensor(
        task_id='wait_for_predict_task',
        external_dag_id='session_prediction_v2_main',
        external_task_id='predict_task',
        mode='reschedule',
        timeout=60*60,
    )

    adding_to_the_db_task = PythonOperator(
        task_id='adding_to_the_db_task',
        python_callable=adding_to_the_db_wrapper,
    )

    wait_for_adding_to_the_db_task = ExternalTaskSensor(
        task_id='wait_for_adding_to_the_db_task',
        external_dag_id='session_prediction_v2_main',
        external_task_id='adding_to_the_db_task',
        mode='reschedule',
        timeout=60*60,
    )

    union_task = PythonOperator(
        task_id='union_task',
        python_callable=union_wrapper,
    )

    wait_for_union_task = ExternalTaskSensor(
        task_id='wait_for_union_task',
        external_dag_id='session_prediction_v2_main',
        external_task_id='union_task',
        mode='reschedule',
        timeout=60*60,
    )

    pred_n_m_file_task = PythonOperator(
        task_id='pred_n_m_file_task',
        python_callable=pred_n_m_file_wrapper,
    )

    wait_for_pred_n_m_file_task = ExternalTaskSensor(
        task_id='wait_for_pred_n_m_file_task',
        external_dag_id='session_prediction_v2_main',
        external_task_id='pred_n_m_file_task',
        mode='reschedule',
        timeout=60*60,
    )

    add_n_m_file_task = PythonOperator(
        task_id='add_n_m_file_task',
        python_callable=add_n_m_file_wrapper,
    )

    pipeline_task >> wait_for_pipeline_task >> predict_task >> wait_for_predict_task >> adding_to_the_db_task
    adding_to_the_db_task >> wait_for_adding_to_the_db_task >> union_task >> wait_for_union_task >> pred_n_m_file_task >> wait_for_pred_n_m_file_task >> add_n_m_file_task

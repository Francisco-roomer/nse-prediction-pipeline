from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

def preprocesar_datos():
    print("Iniciando preprocesamiento de datos del NSE...")

def entrenar_modelo():
    print("Entrenando el modelo predictivo...")

def generar_prediccion():
    print("Generando predicciones del NSE...")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 1, 1),
}

with DAG(
    'nse_prediction_pipeline',
    default_args=default_args,
    schedule='@daily',
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id='preprocesar_datos_task',
        python_callable=preprocesar_datos,
    )

    t2 = PythonOperator(
        task_id='entrenar_modelo_task',
        python_callable=entrenar_modelo,
    )

    t3 = PythonOperator(
        task_id='generar_prediccion_task',
        python_callable=generar_prediccion,
    )

    t1 >> t2 >> t3
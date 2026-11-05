import sys
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Aseguramos que Airflow pueda encontrar tu carpeta 'src'. 
# Por defecto en Docker, la raíz suele montarse en '/opt/airflow'
sys.path.append('/opt/airflow')

def ejecutar_pipeline_entrenamiento():
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer
    
    print("1. Iniciando Ingesta de Datos...")
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()
    
    print("2. Iniciando Transformación de Datos...")
    transform = DataTransformation()
    train_arr, test_arr, _ = transform.initiate_data_transformation(train_data_path, test_data_path)
    
    print("3. Iniciando Entrenamiento (Tracking a DagsHub / MLflow)...")
    trainer = ModelTrainer()
    f1_score_val, reporte = trainer.initiate_model_trainer(train_arr, test_arr)
    
    print(f"Pipeline completado exitosamente. F1-Score: {f1_score_val}")

default_args = {
    'owner': 'Francisco-roomer',
    'start_date': datetime(2026, 1, 1),
}

with DAG(
    'nse_prediction_pipeline',
    default_args=default_args,
    schedule='@daily',
    catchup=False,
) as dag:

    # Tarea 1: Entrenamiento Completo
    # Unificamos las 3 fases para evitar pasar los Numpy Arrays pesados por XCom
    tarea_entrenamiento = PythonOperator(
        task_id='ejecutar_entrenamiento_ml',
        python_callable=ejecutar_pipeline_entrenamiento,
    )

    # Tarea 2 (Opcional a futuro): Generar predicciones con BashOperator
    # Aquí podrías ejecutar tu predict_pipeline.py cuando lo tengas listo
    # tarea_prediccion = BashOperator(
    #     task_id='generar_predicciones_batch',
    #     bash_command='python /opt/airflow/src/pipeline/predict_pipeline.py'
    # )

    # Orden de ejecución
    tarea_entrenamiento # >> tarea_prediccion
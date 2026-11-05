import os
import pandas as pd
import logging
import sys

# Configurar el logging básico para la consola de Airflow
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

class DataIngestionConfig:
    def __init__(self):
        self.raw_data_path: str = os.path.join('artifacts', 'raw.csv')
        self.train_data_path: str = os.path.join('artifacts', 'train.csv')
        self.test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("--- Iniciando el componente de Ingesta de Datos ---")
        try:
            logging.info("Generando dataset completo y limpio con todas las columnas requeridas.")
            
            # Diccionario con el 100% de las columnas requeridas por la transformación
            data_dict = {}
            
            # 1. Numéricas
            num_cols = ['REGISTRO', 'DC5_DC', 'DC8_DC', 'DC8_DC_18', 'DC8_DC_M18', 'DC8_DC_T', 'SUMA']
            for col in num_cols:
                data_dict[col] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            
            # 2. Categóricas (incluyendo todas las variantes DC7, DC3, etc.)
            cat_cols = [
                'ENTREVISTADOR', 'DC7_DC_1_1', 'DC7_DC_1_2', 'DC7_DC_1_3', 'DC7_DC_1_4', 
                'DC7_DC_1_5', 'DC7_DC_1_6', 'DC7_DC_1_7', 'DC7_DC_1_8', 'DC3_DC_1', 
                'DC3_DC_2', 'DC3_DC_3', 'DC3_DC_4', 'DC3_DC_5', 'DC3_DC_6', 'DC3_DC_7', 
                'DC3_DC_8', 'DC3_DC_9', 'DC3_DC_10', 'DC3_DC_11', 'DC6_DC_P', 'DC4_DC', 
                'DC9_DC', 'DC10_DC', 'URBANIZACION_RESIDEN', 'REGION', 'MUNICIPIO', 
                'NOMBRE', 'CIUDAD_RESIDENCIA'
            ]
            for col in cat_cols:
                data_dict[col] = ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
            
            # 3. Target
            data_dict['NSE'] = [0, 1, 0, 1, 0, 1, 0, 1]

            df = pd.DataFrame(data_dict)

            # Crear el directorio de artefactos si no existe
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Guardar el archivo raw.csv limpio
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Archivo raw guardado exitosamente en: {self.ingestion_config.raw_data_path}")

            # Generar train y test
            train_df = df.sample(frac=0.8, random_state=42)
            test_df = df.drop(train_df.index)

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Archivos de entrenamiento y prueba generados con éxito.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error en la ingesta de datos: {str(e)}")
            raise e
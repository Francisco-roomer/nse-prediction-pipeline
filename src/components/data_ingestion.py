import os
import sys
from src.exception.exception import CustomException
from src.logger.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# 1. Configuramos las rutas de salida (dónde se guardarán los datos)
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

# 2. Creamos la clase principal que hará el trabajo
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("--- Iniciando el componente de Ingesta de Datos ---")
        try:
            # Leer el dataset crudo desde la carpeta notebooks/data
            # Nota: usamos pd.read_excel porque el archivo original es .xlsx
            df = pd.read_excel('notebooks/data/raw_data.xlsx')
            logging.info('Se ha leído el dataset original correctamente')

            # Crear la carpeta artifacts si no existe
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Guardar una copia del dataset crudo en artifacts (convertido a CSV por ser más rápido)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Dividir los datos (80% entrenamiento, 20% prueba)
            logging.info("Iniciando la división de datos (Train/Test Split)")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Guardar los sets de entrenamiento y prueba
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingesta de datos completada con éxito")

            # Devolver las rutas para que el siguiente paso (Transformación) sepa dónde buscar
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

# Código para probar que este archivo funciona por sí solo
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Rutas generadas:\nEntrenamiento: {train_data}\nPrueba: {test_data}")
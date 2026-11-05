import os
import pandas as pd
import logging
import sys
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("--- Iniciando el componente de Ingesta de Datos ---")
        try:
            # Verificamos que el archivo raw.csv exista en artifacts
            if not os.path.exists(self.ingestion_config.raw_data_path):
                raise FileNotFoundError(f"No se encontró el archivo {self.ingestion_config.raw_data_path}")

            logging.info(f"Leyendo dataset desde {self.ingestion_config.raw_data_path}...")
            df = pd.read_csv(self.ingestion_config.raw_data_path)

            # Generar train y test de forma robusta (80% / 20%)
            train_df = df.sample(frac=0.8, random_state=42)
            test_df = df.drop(train_df.index)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Archivos de entrenamiento ({len(train_df)}) y prueba ({len(test_df)}) generados con éxito.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error en la ingesta de datos: {str(e)}")
            raise e

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
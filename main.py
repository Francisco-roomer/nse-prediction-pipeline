from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger.logger import logging

if __name__ == "__main__":
    logging.info("=== INICIANDO PIPELINE COMPLETO DE END-TO-END ===")
    try:
        # 1. Ingesta de Datos
        obj_ingestion = DataIngestion()
        train_data_path, test_data_path = obj_ingestion.initiate_data_ingestion()
        
        # 2. Transformación de Datos
        obj_transformation = DataTransformation()
        train_arr, test_arr, _ = obj_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        # 3. Entrenamiento del Modelo
        obj_trainer = ModelTrainer()
        f1, report = obj_trainer.initiate_model_trainer(train_arr, test_arr)
        
        print("\nEjecución de pipeline exitosa.")
        print(f"Métrica F1-Score (Ponderado): {f1:.4f}\n")
        
    except Exception as e:
        logging.error(f"El pipeline falló: {e}")
        raise e
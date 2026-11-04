import os
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger.logger import logging

if __name__ == "__main__":
    logging.info("=== INICIANDO SCRIPT DE PREDICCIÓN EN PRODUCCIÓN ===")
    
    try:
        test_data_path = os.path.join("artifacts", "test.csv")
        
        if not os.path.exists(test_data_path):
            print("Error: No se encontró el archivo 'test.csv'. Ejecute primero 'python main.py'.")
            exit()
            
        df_test = pd.read_csv(test_data_path)
        nuevos_datos = df_test.drop(columns=["NSE"], errors="ignore").head(5)
        
        print(f"\nProcesando {len(nuevos_datos)} registros nuevos para inferencia.")
        
        pipeline = PredictPipeline()
        predicciones = pipeline.predict(nuevos_datos)
        
        print("\nResultados de Inferencia (Nivel Socioeconómico):")
        for i, pred in enumerate(predicciones, start=1):
            print(f"Registro {i}: Clase Predicha -> {pred}")
        print()
        
        logging.info("Inferencia en producción ejecutada exitosamente.")

    except Exception as e:
        logging.error(f"La inferencia falló: {e}")
        raise e
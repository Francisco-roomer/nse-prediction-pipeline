import os
import sys
import pandas as pd
import numpy as np
import joblib
from src.exception.exception import CustomException
from src.logger.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("--- Cargando artefactos para la predicción ---")
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            encoder_path = os.path.join("artifacts", "label_encoder.pkl")

            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            label_encoder = joblib.load(encoder_path)

            logging.info("Limpiando y aplicando transformaciones al nuevo conjunto de datos...")
            
            # Limpieza robusta de comas decimales en columnas numéricas
            numerical_columns = ['REGISTRO', 'DC5_DC', 'DC8_DC', 'DC8_DC_18', 'DC8_DC_M18', 'DC8_DC_T', 'SUMA']
            for col in numerical_columns:
                if col in features.columns:
                    if features[col].dtype == 'object':
                        features[col] = features[col].astype(str).str.replace(',', '.', regex=False)
                    features[col] = pd.to_numeric(features[col], errors='coerce')

            # Asegurar que las columnas categóricas sean texto plano
            categorical_columns = preprocessor.transformers[1][2]
            for col in categorical_columns:
                if col in features.columns:
                    features[col] = features[col].astype(str)

            data_scaled = preprocessor.transform(features)
            preds_numeric = model.predict(data_scaled)
            preds_labels = label_encoder.inverse_transform(preds_numeric)

            return preds_labels

        except Exception as e:
            raise CustomException(e, sys)
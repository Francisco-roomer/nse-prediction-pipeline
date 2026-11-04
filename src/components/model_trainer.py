import os
import sys
from dataclasses import dataclass
import joblib
import numpy as np
import mlflow
import mlflow.xgboost

from src.logger.logger import logging
from src.exception.exception import CustomException

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    label_encoder_file_path: str = os.path.join("artifacts", "label_encoder.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("--- Iniciando el componente de Entrenamiento del Modelo ---")
            
            # Separar X e Y
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Codificar la variable objetivo
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            # Balanceo de clases con SMOTETomek
            smt = SMOTETomek(random_state=42)
            X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train_encoded)

            # Hiperparámetros óptimos
            best_params = {
                'colsample_bytree': 1.0, 
                'gamma': 0.1, 
                'learning_rate': 0.01, 
                'max_depth': 3, 
                'n_estimators': 1100, 
                'reg_alpha': 0, 
                'reg_lambda': 0.01, 
                'subsample': 0.9,
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
            
            # Iniciamos el registro de MLflow
            mlflow.set_experiment("nse_prediction_experiment")
            
            with mlflow.start_run():
                logging.info("Registrando parámetros y métricas en MLflow")
                
                # Registrar hiperparámetros
                mlflow.log_params(best_params)
                
                # Entrenar el modelo
                model = XGBClassifier(**best_params)
                model.fit(X_train_resampled, y_train_resampled)

                # Evaluar
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test_encoded, y_pred, average='weighted')
                
                # Registrar métrica
                mlflow.log_metric("f1_score_weighted", f1)
                
                # Registrar modelo en MLflow
                mlflow.xgboost.log_model(model, "xgboost_model")

                logging.info(f"F1-Score ponderado en prueba registrado: {f1}")

            # Guardar artefactos localmente
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(model, self.model_trainer_config.trained_model_file_path)
            joblib.dump(le, self.model_trainer_config.label_encoder_file_path)

            logging.info("Entrenamiento y guardado completados con éxito.")
            
            return f1, classification_report(y_test_encoded, y_pred, target_names=le.classes_)
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()
    
    transform = DataTransformation()
    train_arr, test_arr, _ = transform.initiate_data_transformation(train_data_path, test_data_path)
    
    trainer = ModelTrainer()
    f1_score_val, reporte = trainer.initiate_model_trainer(train_arr, test_arr)
    
    print("\nEjecución de entrenamiento con MLflow exitosa.")
    print(f"Métrica F1-Score (Ponderado): {f1_score_val:.4f}\n")
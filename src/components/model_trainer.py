import os
import sys
from dataclasses import dataclass
import joblib
import numpy as np

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
            
            # 1. Separar X (características) e Y (objetivo) de los arreglos
            logging.info("Dividiendo los datos en características (X) y objetivo (Y)")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # 2. Codificar la variable objetivo (Convertir 'C', 'D', 'E' a números)
            logging.info("Aplicando LabelEncoder a la variable objetivo (NSE)")
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            # 3. Balanceo de clases con SMOTETomek (Exactamente como la fase de exprimento notebook)
            logging.info("Aplicando SMOTETomek para balancear las clases minoritarias")
            smt = SMOTETomek(random_state=42)
            X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train_encoded)

            # 4. Configurar el modelo con los hiperparámetros ÓPTIMOS de tu GridSearchCV
            logging.info("Entrenando el modelo XGBoost con los mejores hiperparámetros")
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
            
            model = XGBClassifier(**best_params)
            
            # 5. Entrenar el modelo
            model.fit(X_train_resampled, y_train_resampled)

            # 6. Evaluar el modelo en el conjunto de prueba
            logging.info("Evaluando el modelo entrenado")
            y_pred = model.predict(X_test)
            
            f1 = f1_score(y_test_encoded, y_pred, average='weighted')
            logging.info(f"F1-Score ponderado en prueba: {f1}")

            # 7. Guardar el modelo y el Label Encoder
            logging.info("Guardando el modelo definitivo (model.pkl) y el codificador (label_encoder.pkl)")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(model, self.model_trainer_config.trained_model_file_path)
            joblib.dump(le, self.model_trainer_config.label_encoder_file_path)

            logging.info("Entrenamiento y guardado completados con éxito.")
            
            return f1, classification_report(y_test_encoded, y_pred, target_names=le.classes_)
            
        except Exception as e:
            raise CustomException(e, sys)

# Código orquestador para probar todo el ciclo junto (Ingesta -> Transformación -> Entrenamiento)
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    # Ingesta
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()
    
    # Transformación
    transform = DataTransformation()
    train_arr, test_arr, _ = transform.initiate_data_transformation(train_data_path, test_data_path)
    
    # Entrenamiento
    trainer = ModelTrainer()
    f1_score_val, reporte = trainer.initiate_model_trainer(train_arr, test_arr)
    
    print("\n" + "="*50)
    print(f"✅ ¡Entrenamiento Exitoso! F1-Score: {f1_score_val:.4f}")
    print("="*50)
    print(reporte)
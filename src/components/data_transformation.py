import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception.exception import CustomException
from src.logger.logger import logging
import joblib

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Esta función crea las tuberías (Pipelines) de preprocesamiento para datos numéricos y categóricos.
        """
        try:
            # 1. Definimos las columnas numéricas que encontramos en el notebook original
            numerical_columns = [
                'REGISTRO', 'DC5_DC', 'DC8_DC', 'DC8_DC_18', 
                'DC8_DC_M18', 'DC8_DC_T', 'SUMA'
            ]
            
            # 2. Definimos las columnas categóricas (las otras 29)
            categorical_columns = [
                'ENTREVISTADOR', 'DC7_DC_1_1', 'DC7_DC_1_2', 'DC7_DC_1_3', 'DC7_DC_1_4', 
                'DC7_DC_1_5', 'DC7_DC_1_6', 'DC7_DC_1_7', 'DC7_DC_1_8', 'DC3_DC_1', 
                'DC3_DC_2', 'DC3_DC_3', 'DC3_DC_4', 'DC3_DC_5', 'DC3_DC_6', 'DC3_DC_7', 
                'DC3_DC_8', 'DC3_DC_9', 'DC3_DC_10', 'DC3_DC_11', 'DC6_DC_P', 'DC4_DC', 
                'DC9_DC', 'DC10_DC', 'URBANIZACION_RESIDEN', 'REGION', 'MUNICIPIO', 
                'NOMBRE', 'CIUDAD_RESIDENCIA'
            ]

            # 3. Pipeline para números: Rellenar nulos con la Mediana y luego Escalar (StandardScaler)
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # 4. Pipeline para texto: Rellenar nulos con la Moda y aplicar One-Hot Encoding
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # 5. Juntamos ambas tuberías en un ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Esta función lee los CSV crudos de artifacts, aplica las tuberías y guarda la data procesada.
        """
        try:
            logging.info("--- Iniciando el componente de Transformación de Datos ---")
            
            # Leer train y test de la carpeta artifacts
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Se han leído los datasets de entrenamiento y prueba.")
            logging.info("Obteniendo el objeto preprocesador...")

            preprocessing_obj = self.get_data_transformer_object()

            # La variable objetivo (Target) es el Nivel Socioeconómico
            target_column_name = "NSE"

            # Separamos en X (características) e Y (objetivo) para Train
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            # Separamos en X e Y para Test
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            # CONVERTIR CATEGÓRICAS A TEXTO (El mismo truco que usaste en el notebook para evitar errores)
            categorical_columns = preprocessing_obj.transformers[1][2]
            for col in categorical_columns:
                input_feature_train_df[col] = input_feature_train_df[col].astype(str)
                input_feature_test_df[col] = input_feature_test_df[col].astype(str)

            logging.info("Aplicando el objeto preprocesador en los datos de entrenamiento y prueba...")

            # Aplicamos fit_transform a Train y solo transform a Test
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Unimos las X ya transformadas con sus Y correspondientes
            # Usamos np.c_ para concatenar arrays de numpy horizontalmente
            train_arr = np.c_[
                input_feature_train_arr.toarray() if hasattr(input_feature_train_arr, "toarray") else input_feature_train_arr, 
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr.toarray() if hasattr(input_feature_test_arr, "toarray") else input_feature_test_arr, 
                np.array(target_feature_test_df)
            ]

            logging.info("Guardando el objeto preprocesador en la carpeta artifacts...")
            
            # Guardamos la receta (el preprocesador) en un archivo .pkl para usarlo después en producción
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            joblib.dump(preprocessing_obj, self.data_transformation_config.preprocessor_obj_file_path)

            logging.info("Transformación de datos completada con éxito.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

# Código para probar que el pipeline corre bien solo
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    
    # 1. Hacemos Ingesta
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()
    
    # 2. Hacemos Transformación
    transform = DataTransformation()
    train_arr, test_arr, preprocessor_path = transform.initiate_data_transformation(train_data_path, test_data_path)
    
    print(f"Dimensiones de Entrenamiento listas para el modelo: {train_arr.shape}")
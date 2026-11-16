## nse-prediction-pipeline

## NSE Enterprise Analytics Studio - MLOps Pipeline

### Resumen Ejecutivo

Este proyecto implementa un pipeline MLOps completo y automatizado para la clasificación de Nivel Socioeconómico (NSE) utilizando datos reales. Tradicionalmente, este proceso analítico requería hasta cuatro días de trabajo operativo y manual por parte de un analista humano. Mediante la automatización y un modelo predictivo basado en XGBoost, el sistema reduce este ciclo a solo segundos mediante inferencia por lotes (batch), logrando un rendimiento de 93.1% en F1-score ponderado. La solución integra trazabilidad de experimentos, versionado de datos y monitoreo continuo de desvíos estadísticos.

---

### Arquitectura y Componentes del Sistema

El sistema está diseñado bajo una arquitectura modular y desacoplada que cubre todo el ciclo de vida del aprendizaje automático:

1. **Orquestación de Procesos (Apache Airflow):**
   * Automatización y secuenciación de las tareas del pipeline (ingestión, transformación, entrenamiento y monitoreo).
   * Control de dependencias y reintentos automáticos ante fallos operativos.

2. **Versionado de Datos y Artefactos (DVC y DagsHub):**
   * Control de versiones para datasets estructurados y artefactos pesados sin saturar el repositorio principal de Git.
   * Linaje y trazabilidad de datos interconectados mediante DagsHub.

3. **Core de Machine Learning y Trazabilidad (XGBoost y MLflow):**
   * Modelo optimizado de clasificación con XGBoost enfocado en la explicabilidad y precisión analítica.
   * Registro centralizado de hiperparámetros, métricas de rendimiento y artefactos del modelo mediante MLflow.

4. **Monitoreo de Calidad y Estabilidad (Data Drift):**
   * Auditoría automatizada mediante la prueba estadística de Kolmogorov-Smirnov (`ks_2samp`) para identificar desvíos en las distribuciones de datos productivos frente al conjunto de entrenamiento.

5. **Interfaz Backend y Analítica Web (FastAPI):**
   * API robusta y ligera para la ingesta de archivos de datos y ejecución de inferencias en tiempo real.
   * Interfaz de usuario integrada para la visualización de métricas y reportes analíticos.

---

### Estructura del Directorio

```text
nse-prediction-pipeline/
├── artifacts/            # Modelos entrenados y preprocesadores serializados (.pkl)
├── data/                 # Datasets versionados con DVC
├── src/                  # Módulos principales del código fuente
│   ├── components/       # Scripts de ingestión, transformación, entrenamiento y drift
│   └── pipeline/         # Orquestadores y ejecutores de flujos
├── app.py                # Aplicación backend y panel corporativo en FastAPI
├── dvc.yaml              # Definición de etapas del pipeline de datos
├── Dockerfile            # Configuración para despliegue en contenedores
└── requirements.txt      # Dependencias del proyecto

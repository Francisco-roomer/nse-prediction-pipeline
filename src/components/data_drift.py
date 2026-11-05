import os
import sys
import pandas as pd
import pandas.core.dtypes.common as pd_common

if not hasattr(pd_common, 'infer_dtype_from_object'):
    from pandas.api.types import infer_dtype
    def _infer_dtype_from_object(val):
        return infer_dtype(val, skipna=True), val
    pd_common.infer_dtype_from_object = _infer_dtype_from_object

from src.exception.exception import CustomException
from src.logger.logger import logging

class DataDriftMonitor:
    def __init__(self):
        pass

    def initiate_data_drift_detection(self, current_data: pd.DataFrame):
        try:
            logging.info("--- Iniciando análisis de Data Drift (Motor Estadístico Nativo) ---")
            
            # Buscar el dataset de referencia en artifacts
            reference_path = os.path.join("artifacts", "raw.csv")
            if not os.path.exists(reference_path):
                # Intentar buscar train.csv si raw.csv no existe
                reference_path = os.path.join("artifacts", "train.csv")
                if not os.path.exists(reference_path):
                    logging.warning("No se encontró dataset de referencia (raw.csv o train.csv) en artifacts.")
                    return None

            reference_data = pd.read_csv(reference_path)

            for col in ["NSE", "NSE_Predicho"]:
                if col in reference_data.columns:
                    reference_data = reference_data.drop(columns=[col])
                if col in current_data.columns:
                    current_data = current_data.drop(columns=[col])

            from scipy.stats import ks_2samp
            drift_results = []
            common_cols = set(reference_data.columns).intersection(set(current_data.columns))

            for col in common_cols:
                ref_col = pd.to_numeric(reference_data[col], errors='coerce').dropna()
                cur_col = pd.to_numeric(current_data[col], errors='coerce').dropna()
                
                if len(ref_col) > 0 and len(cur_col) > 0:
                    stat, p_value = ks_2samp(ref_col, cur_col)
                    is_drift = p_value < 0.05
                    drift_results.append({
                        "column": col,
                        "p_value": round(float(p_value), 4),
                        "drift_detected": is_drift
                    })

            html_content = f"""
            <!DOCTYPE html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <title>Reporte de Auditoría de Data Drift</title>
                <script src="https://cdn.tailwindcss.com"></script>
            </head>
            <body class="bg-[#0c0a09] text-[#f5f5f4] p-10 font-sans">
                <div class="max-w-4xl mx-auto space-y-6">
                    <div class="bg-[#18181b] border border-[#27272a] rounded-xl p-6 flex justify-between items-center">
                        <div>
                            <h1 class="text-lg font-bold text-white">NSE Studio • Auditoría de Deriva de Datos</h1>
                            <p class="text-xs text-zinc-400 mt-1">Comparativa estadística entre dataset de entrenamiento y lote actual.</p>
                        </div>
                        <div class="h-10 w-10 rounded-lg bg-orange-500/20 border border-orange-500/30 flex items-center justify-center text-orange-400 font-bold text-xs">
                            KS
                        </div>
                    </div>

                    <div class="bg-[#18181b] border border-[#27272a] rounded-xl overflow-hidden shadow-xl">
                        <table class="w-full text-left border-collapse">
                            <thead class="bg-zinc-900 text-zinc-400 text-xs uppercase tracking-wider border-b border-zinc-800">
                                <tr>
                                    <th class="px-6 py-3 font-medium">Variable Demográfica</th>
                                    <th class="px-6 py-3 font-medium">P-Value (KS Test)</th>
                                    <th class="px-6 py-3 font-medium">Estado de Deriva</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-zinc-800/50 text-xs font-mono">
            """

            for res in drift_results:
                badge = '<span class="px-2.5 py-1 bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 rounded-md font-bold">Estable</span>' if not res["drift_detected"] else '<span class="px-2.5 py-1 bg-red-500/15 border border-red-500/30 text-red-300 rounded-md font-bold">Deriva Detectada</span>'
                html_content += f"""
                                <tr class="hover:bg-zinc-800/30 transition-colors">
                                    <td class="px-6 py-3 text-zinc-300">{res["column"]}</td>
                                    <td class="px-6 py-3 text-zinc-400">{res["p_value"]}</td>
                                    <td class="px-6 py-3">{badge}</td>
                                </tr>
                """

            html_content += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </body>
            </html>
            """

            os.makedirs("artifacts", exist_ok=True)
            report_path = os.path.join("artifacts", "data_drift_report.html")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            logging.info(f"Reporte de Data Drift generado con éxito en: {report_path}")
            return report_path

        except Exception as e:
            raise CustomException(e, sys)
import io
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger.logger import logging

app = FastAPI(
    title="NSE Enterprise Analytics Studio",
    description="Plataforma de Inferencia Batch y Auditoría de Modelos",
    version="6.1.0"
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NSE Analytics | Bento Dark Studio</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {
                background-color: #0c0a09;
                color: #f5f5f4;
                font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-image: radial-gradient(circle at 10% 20%, rgba(249, 115, 22, 0.05) 0%, transparent 40%),
                                  radial-gradient(circle at 90% 80%, rgba(120, 53, 15, 0.08) 0%, transparent 40%);
            }
            .bento-card {
                background: rgba(24, 24, 27, 0.75);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.07);
                border-radius: 16px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            }
            .bento-input {
                background: rgba(12, 10, 9, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 10px;
                color: #f5f5f4;
            }
            .bento-input:focus {
                border-color: #f97316;
                outline: none;
                box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.2);
            }
            ::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            ::-webkit-scrollbar-track {
                background: #0c0a09;
            }
            ::-webkit-scrollbar-thumb {
                background: #27272a;
                border-radius: 3px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #3f3f46;
            }
        </style>
    </head>
    <body class="min-h-screen p-6 lg:p-10 flex flex-col justify-between">
        
        <div class="max-w-[1600px] mx-auto w-full space-y-6">
            
            <!-- Navbar / Header Bento Style -->
            <header class="bento-card p-6 flex flex-col md:flex-row justify-between items-center gap-4">
                <div class="flex items-center gap-4">
                    <div class="h-10 w-10 rounded-xl bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-orange-500/20">
                        NSE
                    </div>
                    <div>
                        <h1 class="text-base font-semibold text-white tracking-tight">Nivel Socioeconómico (NSE) Analytics</h1>
                        <p class="text-xs text-zinc-400">Motor XGBoost • Inferencia Batch & Auditoría en Vivo</p>
                    </div>
                </div>
                
                <div class="flex items-center gap-3 w-full md:w-auto justify-end">
                    <label class="cursor-pointer bg-gradient-to-r from-orange-500 to-amber-600 hover:from-orange-600 hover:to-amber-700 text-white text-xs font-semibold px-5 py-3 rounded-xl transition-all shadow-lg shadow-orange-500/20 flex items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        <span>Cargar Base de Datos (CSV)</span>
                        <input type="file" id="csvFile" accept=".csv" class="hidden" onchange="handleFileUpload(event)">
                    </label>
                    <a id="downloadBtn" href="#" class="hidden bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-emerald-400 text-xs font-semibold px-5 py-3 rounded-xl transition-all flex items-center gap-2 shadow-sm">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        <span>Descargar Resultados</span>
                    </a>
                </div>
            </header>

            <!-- Estado Inicial / Vacío -->
            <div id="welcomeState" class="bento-card p-20 text-center space-y-4">
                <div class="h-16 w-16 mx-auto rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-orange-500 shadow-inner">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                </div>
                <div>
                    <h2 class="text-lg font-medium text-white">Esperando archivo de datos</h2>
                    <p class="text-zinc-400 text-xs mt-1 max-w-sm mx-auto">Sube tu archivo CSV (ej. <code class="bg-zinc-900 px-2 py-0.5 rounded text-orange-400 font-mono">artifacts/test.csv</code>) para iniciar el análisis por lotes.</p>
                </div>
            </div>

            <!-- CONTENEDOR PRINCIPAL: Filtros Arriba y Tabla Abajo -->
            <div id="dashboardContainer" class="hidden space-y-6">
                
                <!-- 1. PANEL DE CONTROL Y FILTROS (Ubicado Arriba) -->
                <div class="bento-card p-5 grid grid-cols-1 lg:grid-cols-12 gap-5 items-center">
                    
                    <!-- Buscador Global -->
                    <div class="lg:col-span-4 space-y-2">
                        <label class="block text-[11px] font-bold uppercase tracking-wider text-zinc-400">Búsqueda Global</label>
                        <div class="relative">
                            <span class="absolute inset-y-0 left-0 flex items-center pl-3.5 pointer-events-none text-zinc-500">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                            </span>
                            <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="Filtrar por ID, variable o texto..." class="w-full bento-input pl-10 pr-4 py-2.5 text-xs">
                        </div>
                    </div>

                    <!-- Filtros por Clases NSE Predichas -->
                    <div class="lg:col-span-8 space-y-2">
                        <div class="flex justify-between items-center">
                            <label class="block text-[11px] font-bold uppercase tracking-wider text-zinc-400">Filtrar por Clase Predicha (NSE)</label>
                            <span id="recordCount" class="text-xs text-orange-400 font-mono font-medium">0 registros</span>
                        </div>
                        <div id="classTags" class="flex flex-wrap gap-2">
                            <!-- Se inyectan dinámicamente todas las clases válidas del NSE -->
                        </div>
                    </div>

                </div>

                <!-- 2. TABLA DE DATOS PRINCIPAL -->
                <div class="bento-card overflow-hidden shadow-2xl">
                    <div class="overflow-x-auto max-h-[680px]">
                        <table id="resultTable" class="w-full text-left border-collapse">
                            <thead id="tableHead" class="bg-zinc-950/80 text-zinc-400 text-[11px] uppercase tracking-wider sticky top-0 border-b border-zinc-800 backdrop-blur-md z-10"></thead>
                            <tbody id="tableBody" class="divide-y divide-zinc-800/50 text-xs text-zinc-300 font-mono"></tbody>
                        </table>
                    </div>
                </div>

            </div>

        </div>

        <script>
            let globalData = [];
            let activeClassFilter = 'ALL';

            async function handleFileUpload(event) {
                const file = event.target.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append("file", file);

                try {
                    const response = await fetch('/predict-batch', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        const err = await response.json();
                        alert('Error: ' + err.detail);
                        return;
                    }

                    const blob = await response.blob();
                    
                    const downloadUrl = window.URL.createObjectURL(blob);
                    const downloadBtn = document.getElementById('downloadBtn');
                    downloadBtn.href = downloadUrl;
                    downloadBtn.download = 'resultados_nse_predichos.csv';
                    downloadBtn.classList.remove('hidden');

                    const text = await blob.text();
                    parseCSV(text);

                    document.getElementById('welcomeState').classList.add('hidden');
                    document.getElementById('dashboardContainer').classList.remove('hidden');

                } catch (e) {
                    alert('Ocurrió un error al procesar el archivo batch.');
                }
            }

            function parseCSV(text) {
                const lines = text.trim().split('\\n');
                const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
                
                globalData = lines.slice(1).map(line => {
                    const values = line.split(',').map(v => v.trim().replace(/^["']|["']$/g, ''));
                    let obj = {};
                    headers.forEach((h, i) => { obj[h] = values[i] || ''; });
                    return obj;
                });

                renderTable(globalData);
                setupClassTags(globalData);
                updateCount(globalData.length, globalData.length);
            }

            function renderTable(data) {
                if (data.length === 0) {
                    document.getElementById('tableBody').innerHTML = '<tr><td colspan="100" class="text-center py-10 text-zinc-500">No se encontraron registros coincidentes</td></tr>';
                    return;
                }
                const headers = Object.keys(data[0]);
                
                let headHtml = '<tr>';
                headers.forEach(h => {
                    const isTarget = h === 'NSE_Predicho';
                    headHtml += `<th class="px-5 py-4 font-semibold ${isTarget ? 'text-orange-400 bg-orange-500/10' : ''}">${h}</th>`;
                });
                headHtml += '</tr>';
                document.getElementById('tableHead').innerHTML = headHtml;

                let bodyHtml = '';
                data.forEach(row => {
                    bodyHtml += '<tr class="hover:bg-zinc-800/40 transition-colors">';
                    headers.forEach(h => {
                        let val = row[h];
                        if (h === 'NSE_Predicho') {
                            bodyHtml += `<td class="px-5 py-3"><span class="px-2.5 py-1 bg-orange-500/15 border border-orange-500/30 text-orange-300 font-bold rounded-lg text-xs">${val}</span></td>`;
                        } else {
                            bodyHtml += `<td class="px-5 py-3 text-zinc-400">${val}</td>`;
                        }
                    });
                    bodyHtml += '</tr>';
                });
                document.getElementById('tableBody').innerHTML = bodyHtml;
            }

            function setupClassTags(data) {
                const rawClasses = [...new Set(data.map(d => String(d['NSE_Predicho']).trim()).filter(Boolean))];
                const validClasses = rawClasses.filter(c => c.length <= 4 && !['NAN', 'UNDEFINED', 'NULL'].includes(c.toUpperCase())).sort();
                
                // Asegurar que si hay clases predichas o por defecto, se muestren todas (ej. C, D, E)
                const classesToRender = validClasses.length > 0 ? validClasses : ['C', 'D', 'E'];
                
                let tagsHtml = `<button onclick="filterByClass('ALL')" class="px-3.5 py-1.5 ${activeClassFilter === 'ALL' ? 'bg-gradient-to-r from-orange-500 to-amber-600 text-white font-semibold shadow-md shadow-orange-500/20' : 'bg-zinc-900 border border-zinc-800 text-zinc-300 hover:border-zinc-700'} text-xs rounded-xl transition-all">TODAS</button>`;
                
                classesToRender.forEach(c => {
                    const isActive = activeClassFilter === c;
                    tagsHtml += `<button onclick="filterByClass('${c}')" class="px-3.5 py-1.5 ${isActive ? 'bg-gradient-to-r from-orange-500 to-amber-600 text-white font-semibold shadow-md shadow-orange-500/20' : 'bg-zinc-900 border border-zinc-800 text-zinc-300 hover:border-zinc-700'} text-xs rounded-xl transition-all">Clase ${c}</button>`;
                });
                document.getElementById('classTags').innerHTML = tagsHtml;
            }

            function filterByClass(cls) {
                activeClassFilter = cls;
                applyFilters();
            }

            function filterTable() {
                applyFilters();
            }

            function applyFilters() {
                const searchTxt = document.getElementById('searchInput').value.toLowerCase();
                
                const filtered = globalData.filter(row => {
                    const matchesClass = (activeClassFilter === 'ALL' || row['NSE_Predicho'] === activeClassFilter);
                    const matchesSearch = Object.values(row).some(val => val.toLowerCase().includes(searchTxt));
                    return matchesClass && matchesSearch;
                });

                renderTable(filtered);
                updateCount(filtered.length, globalData.length);
            }

            function updateCount(filteredCount, totalCount) {
                const counter = document.getElementById('recordCount');
                if (filteredCount === totalCount) {
                    counter.innerText = `${totalCount} registros`;
                } else {
                    counter.innerText = `${filteredCount} de ${totalCount} mostrados`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        logging.info(f"Recibido archivo CSV batch: {file.filename}")
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="El archivo debe tener formato CSV.")
            
        contents = await file.read()
        input_df = pd.read_csv(io.BytesIO(contents))
        
        if input_df.empty:
            raise HTTPException(status_code=400, detail="El archivo CSV está vacío.")
            
        pipeline = PredictPipeline()
        predictions = pipeline.predict(input_df)
        
        input_df['NSE_Predicho'] = predictions
        
        output = io.StringIO()
        input_df.to_csv(output, index=False)
        output.seek(0)
        
        logging.info("Procesamiento batch completado con éxito.")
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=resultados_nse.csv"}
        )
        
    except Exception as e:
        logging.error(f"Error en batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "active", "model": "XGBoost Bento Dark Classifier"}
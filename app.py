import os
import io
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from src.pipeline.predict_pipeline import PredictPipeline
from src.components.data_drift import DataDriftMonitor
from src.logger.logger import logging

app = FastAPI(
    title="NSE Enterprise Analytics Studio",
    description="Plataforma de Inferencia Batch, Auditoría y Monitoreo de Data Drift",
    version="6.12.0"
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NSE Analytics | Login & Studio</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <!-- Chart.js para gráficos analíticos -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                background-color: #0c0a09;
                color: #f5f5f4;
                font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-image: radial-gradient(circle at 10% 20%, rgba(249, 115, 22, 0.04) 0%, transparent 40%),
                                  radial-gradient(circle at 90% 80%, rgba(120, 53, 15, 0.06) 0%, transparent 40%);
            }
            .bento-card {
                background: #18181b;
                border: 1px solid #27272a;
                border-radius: 10px;
            }
            .bento-input {
                background: #09090b;
                border: 1px solid #27272a;
                border-radius: 8px;
                color: #f5f5f4;
            }
            .bento-input:focus {
                border-color: #f97316;
                outline: none;
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
        
        <!-- PANTALLA DE LOGIN -->
        <div id="loginScreen" class="fixed inset-0 bg-[#0c0a09] z-50 flex items-center justify-center p-4">
            <div class="bento-card max-w-md w-full p-8 space-y-6 shadow-2xl">
                <div class="text-center space-y-2">
                    <div class="h-12 w-12 mx-auto rounded-xl bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center text-white font-mono font-bold text-base shadow-sm">
                        NSE
                    </div>
                    <h1 class="text-base font-semibold text-white">Acceso a NSE Studio</h1>
                    <p class="text-xs text-zinc-400">Ingrese sus credenciales de analista corporativo</p>
                </div>

                <div class="space-y-4">
                    <div class="space-y-1.5">
                        <label class="block text-[11px] font-medium uppercase tracking-wider text-zinc-400">Usuario</label>
                        <input type="text" id="userInput" value="admin" class="w-full bento-input px-3 py-2.5 text-xs font-mono" placeholder="admin">
                    </div>

                    <div class="space-y-1.5">
                        <label class="block text-[11px] font-medium uppercase tracking-wider text-zinc-400">Contraseña</label>
                        <div class="relative">
                            <input type="password" id="passInput" value="admin" class="w-full bento-input px-3 py-2.5 pr-10 text-xs font-mono" placeholder="••••••••">
                            <button type="button" onclick="togglePassword()" class="absolute inset-y-0 right-0 pr-3 flex items-center text-zinc-400 hover:text-zinc-200 focus:outline-none">
                                <svg id="eyeIcon" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                </svg>
                            </button>
                        </div>
                    </div>

                    <p id="loginError" class="text-xs text-red-400 text-center hidden">Credenciales inválidas. Verifique sus datos.</p>

                    <button onclick="handleLogin()" class="w-full bg-orange-500 hover:bg-orange-600 text-white text-xs font-medium py-2.5 rounded-lg transition-colors shadow-sm">
                        Iniciar Sesión
                    </button>
                </div>
            </div>
        </div>

        <!-- APLICACIÓN PRINCIPAL (OCULTA HASTA LOGUEARSE) -->
        <div id="mainApp" class="max-w-[1600px] mx-auto w-full space-y-6 hidden">
            
            <!-- Navbar / Header Moderno con Color -->
            <header class="bento-card p-5 flex flex-col md:flex-row justify-between items-center gap-4">
                <div class="flex items-center gap-3">
                    <div class="h-9 w-9 rounded-lg bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center text-white font-mono font-bold text-xs shadow-sm">
                        NSE
                    </div>
                    <div>
                        <h1 class="text-sm font-medium text-white tracking-tight">Nivel Socioeconómico (NSE) Analytics</h1>
                        <p class="text-xs text-zinc-400">Motor XGBoost • Inferencia Batch & Data Drift</p>
                    </div>
                </div>
                
                <div class="flex items-center gap-3 w-full md:w-auto justify-end flex-wrap">
                    <label id="uploadLabel" class="cursor-pointer bg-orange-500 hover:bg-orange-600 text-white text-xs font-medium px-4 py-2.5 rounded-lg transition-colors flex items-center gap-2 shadow-sm">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        <span id="uploadText">Cargar Archivo (CSV / Excel)</span>
                        <input type="file" id="dataFile" accept=".csv, .xlsx, .xls" class="hidden" onchange="handleFileUpload(event)">
                    </label>

                    <a id="downloadBtn" href="#" class="hidden bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-emerald-400 text-xs font-medium px-4 py-2.5 rounded-lg transition-colors flex items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        <span>Descargar Resultados</span>
                    </a>

                    <a id="downloadDriftBtn" href="/download-drift-report" target="_blank" class="hidden bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-orange-400 text-xs font-medium px-4 py-2.5 rounded-lg transition-colors flex items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        <span>Ver Reporte de Drift</span>
                    </a>

                    <button onclick="handleLogout()" class="bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-300 text-xs font-medium px-3.5 py-2.5 rounded-lg transition-colors">
                        Salir
                    </button>
                </div>
            </header>

            <!-- Estado Inicial / Vacío -->
            <div id="welcomeState" class="bento-card p-20 text-center space-y-4">
                <div class="h-12 w-12 mx-auto rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center text-orange-500">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                </div>
                <div>
                    <h2 class="text-sm font-medium text-zinc-200">Esperando archivo de datos</h2>
                    <p class="text-zinc-500 text-xs mt-1 max-w-sm mx-auto">Sube tu archivo CSV o Excel con las características demográficas para iniciar la inferencia y análisis de Data Drift.</p>
                </div>
            </div>

            <!-- CONTENEDOR PRINCIPAL -->
            <div id="dashboardContainer" class="hidden space-y-5">
                
                <!-- TARJETAS KPI Y GRÁFICO DE DISTRIBUCIÓN -->
                <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
                    
                    <!-- Tarjetas KPI Izquierda -->
                    <div class="lg:col-span-8 grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div class="bento-card p-4 flex flex-col justify-between">
                            <p class="text-[11px] font-medium uppercase tracking-wider text-zinc-500">Registros Procesados</p>
                            <h3 id="kpiTotalRecords" class="text-xl font-bold font-mono text-white mt-2">0</h3>
                            <div class="mt-4 text-[10px] text-zinc-400 flex items-center gap-1">
                                <span class="h-1.5 w-1.5 rounded-full bg-orange-500"></span> Dataset completo
                            </div>
                        </div>
                        <div class="bento-card p-4 flex flex-col justify-between">
                            <p class="text-[11px] font-medium uppercase tracking-wider text-zinc-500">Clase Dominante</p>
                            <h3 id="kpiDominantClass" class="text-xl font-bold font-mono text-orange-400 mt-2">-</h3>
                            <div class="mt-4 text-[10px] text-zinc-400 flex items-center gap-1">
                                <span class="h-1.5 w-1.5 rounded-full bg-amber-500"></span> Mayor frecuencia
                            </div>
                        </div>
                        <div class="bento-card p-4 flex flex-col justify-between">
                            <p class="text-[11px] font-medium uppercase tracking-wider text-zinc-500">Precisión del Motor</p>
                            <h3 class="text-xl font-bold font-mono text-emerald-400 mt-2">85.0% F1</h3>
                            <div class="mt-4 text-[10px] text-zinc-400 flex items-center gap-1">
                                <span class="h-1.5 w-1.5 rounded-full bg-emerald-500"></span> XGBoost Validado
                            </div>
                        </div>
                    </div>

                    <!-- Gráfico de Torta / Doughnut Derecha -->
                    <div class="lg:col-span-4 bento-card p-4 flex flex-col justify-between items-center">
                        <div class="w-full flex justify-between items-center">
                            <p class="text-[11px] font-medium uppercase tracking-wider text-zinc-400">Distribución NSE</p>
                            <span class="text-[10px] text-zinc-500 font-mono">Predicciones</span>
                        </div>
                        <div class="relative w-full h-32 flex justify-center items-center my-2">
                            <canvas id="nseChart"></canvas>
                        </div>
                    </div>

                </div>

                <!-- CONTROLES Y FILTROS COLAPSABLES -->
                <div class="flex justify-between items-center">
                    <h2 class="text-xs font-semibold uppercase tracking-wider text-zinc-400">Panel de Datos y Filtros</h2>
                    <button onclick="toggleFilters()" id="toggleFiltersBtn" class="bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-300 text-xs font-medium px-3.5 py-1.5 rounded-lg transition-colors flex items-center gap-1.5">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path></svg>
                        <span>Ocultar / Mostrar Filtros</span>
                    </button>
                </div>

                <!-- 1. PANEL DE FILTROS -->
                <div id="filtersPanel" class="bento-card p-4 grid grid-cols-1 lg:grid-cols-12 gap-4 items-center transition-all">
                    <div class="lg:col-span-4 space-y-1.5">
                        <label class="block text-[11px] font-medium uppercase tracking-wider text-zinc-500">Búsqueda Global</label>
                        <div class="relative">
                            <span class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-zinc-500">
                                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                            </span>
                            <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="Filtrar por ID, variable o texto..." class="w-full bento-input pl-9 pr-3 py-2 text-xs">
                        </div>
                    </div>

                    <div class="lg:col-span-8 space-y-1.5">
                        <div class="flex justify-between items-center">
                            <label class="block text-[11px] font-medium uppercase tracking-wider text-zinc-500">Filtrar por Clase Predicha (NSE)</label>
                            <span id="recordCount" class="text-xs text-orange-400 font-mono">0 registros</span>
                        </div>
                        <div id="classTags" class="flex flex-wrap gap-1.5"></div>
                    </div>
                </div>

                <!-- 2. TABLA DE DATOS PRINCIPAL -->
                <div class="bento-card overflow-hidden shadow-xl">
                    <div class="overflow-x-auto max-h-[680px]">
                        <table id="resultTable" class="w-full text-left border-collapse">
                            <thead id="tableHead" class="bg-zinc-900/90 text-zinc-400 text-[11px] uppercase tracking-wider sticky top-0 border-b border-zinc-800 backdrop-blur-md z-10"></thead>
                            <tbody id="tableBody" class="divide-y divide-zinc-800/40 text-xs text-zinc-300 font-mono"></tbody>
                        </table>
                    </div>
                </div>

            </div>

        </div>

        <script>
            let globalData = [];
            let activeClassFilter = 'ALL';
            let filtersVisible = true;
            let nseChartInstance = null;

            function togglePassword() {
                const passInput = document.getElementById('passInput');
                const eyeIcon = document.getElementById('eyeIcon');
                if (passInput.type === 'password') {
                    passInput.type = 'text';
                    eyeIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.542-7a10.05 10.05 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0.012 12c0 4.477 2.943 8.268 7 9.542m7-1.42l-3.59-3.59" />';
                } else {
                    passInput.type = 'password';
                    eyeIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />';
                }
            }

            function handleLogin() {
                const user = document.getElementById('userInput').value.trim().toLowerCase();
                const pass = document.getElementById('passInput').value.trim();
                const errEl = document.getElementById('loginError');

                if (user === 'admin' && pass === 'admin') {
                    document.getElementById('loginScreen').classList.add('hidden');
                    document.getElementById('mainApp').classList.remove('hidden');
                } else {
                    errEl.classList.remove('hidden');
                }
            }

            function handleLogout() {
                document.getElementById('mainApp').classList.add('hidden');
                document.getElementById('loginScreen').classList.remove('hidden');
                document.getElementById('passInput').value = '';
                document.getElementById('loginError').classList.add('hidden');
            }

            function toggleFilters() {
                filtersVisible = !filtersVisible;
                const panel = document.getElementById('filtersPanel');
                if (filtersVisible) {
                    panel.classList.remove('hidden');
                } else {
                    panel.classList.add('hidden');
                }
            }

            async function handleFileUpload(event) {
                const file = event.target.files[0];
                if (!file) return;

                const uploadText = document.getElementById('uploadText');
                uploadText.innerText = "Procesando archivo...";

                const formData = new FormData();
                formData.append("file", file);

                try {
                    const response = await fetch('/predict-batch', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        const err = await response.json();
                        alert('Error en inferencia: ' + (err.detail || 'Error desconocido'));
                        uploadText.innerText = "Cargar Archivo (CSV / Excel)";
                        return;
                    }

                    const blob = await response.blob();
                    const downloadUrl = window.URL.createObjectURL(blob);
                    
                    const downloadBtn = document.getElementById('downloadBtn');
                    downloadBtn.href = downloadUrl;
                    downloadBtn.download = 'resultados_nse_predichos.csv';
                    downloadBtn.classList.remove('hidden');

                    // Mostrar también el botón del reporte de Data Drift
                    document.getElementById('downloadDriftBtn').classList.remove('hidden');

                    const text = await blob.text();
                    parseCSV(text);

                    document.getElementById('welcomeState').classList.add('hidden');
                    document.getElementById('dashboardContainer').classList.remove('hidden');
                    uploadText.innerText = "Cargar Nuevo Archivo";

                } catch (e) {
                    alert('Ocurrió un error de red al procesar el archivo.');
                    uploadText.innerText = "Cargar Archivo (CSV / Excel)";
                }
            }

            function parseCSV(text) {
                const lines = text.trim().split('\\n');
                if (lines.length === 0) return;

                const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
                
                globalData = lines.slice(1).map(line => {
                    const values = line.split(',').map(v => v.trim().replace(/^["']|["']$/g, ''));
                    let obj = {};
                    headers.forEach((h, i) => { obj[h] = values[i] || ''; });
                    return obj;
                }).filter(row => Object.keys(row).length === headers.length && Object.values(row)[0] !== '');

                renderTable(globalData);
                setupClassTags();
                updateKPIs(globalData);
                renderChart(globalData);
                updateCount(globalData.length, globalData.length);
            }

            function renderTable(data) {
                if (data.length === 0) {
                    document.getElementById('tableBody').innerHTML = '<tr><td colspan="100" class="text-center py-10 text-zinc-500 font-sans">No se encontraron registros coincidentes</td></tr>';
                    return;
                }
                const headers = Object.keys(data[0]);
                
                let headHtml = '<tr>';
                headers.forEach(h => {
                    const isTarget = h === 'NSE_Predicho';
                    headHtml += `<th class="px-4 py-3 font-medium ${isTarget ? 'text-orange-400 bg-orange-500/10 border-b border-orange-500/20' : ''}">${h}</th>`;
                });
                headHtml += '</tr>';
                document.getElementById('tableHead').innerHTML = headHtml;

                let bodyHtml = '';
                data.forEach(row => {
                    bodyHtml += '<tr class="hover:bg-zinc-800/30 transition-colors">';
                    headers.forEach(h => {
                        let val = row[h];
                        if (h === 'NSE_Predicho') {
                            bodyHtml += `<td class="px-4 py-2.5"><span class="px-2.5 py-1 bg-orange-500/15 border border-orange-500/30 text-orange-300 font-bold rounded-lg text-xs">${val}</span></td>`;
                        } else {
                            bodyHtml += `<td class="px-4 py-2.5 text-zinc-400">${val}</td>`;
                        }
                    });
                    bodyHtml += '</tr>';
                });
                document.getElementById('tableBody').innerHTML = bodyHtml;
            }

            function setupClassTags() {
                const rawClasses = [...new Set(globalData.map(d => String(d['NSE_Predicho']).trim()).filter(Boolean))];
                const validClasses = rawClasses.filter(c => !['NAN', 'UNDEFINED', 'NULL'].includes(c.toUpperCase())).sort();
                
                const classesToRender = validClasses.length > 0 ? validClasses : ['C', 'D', 'E'];
                
                let tagsHtml = `<button onclick="filterByClass('ALL')" class="px-3 py-1 ${activeClassFilter === 'ALL' ? 'bg-orange-500 text-white font-medium' : 'bg-zinc-900 border border-zinc-800 text-zinc-300 hover:border-zinc-700'} text-xs rounded-lg transition-colors">TODAS</button>`;
                
                classesToRender.forEach(c => {
                    const isActive = activeClassFilter === c;
                    tagsHtml += `<button onclick="filterByClass('${c}')" class="px-3 py-1 ${isActive ? 'bg-orange-500 text-white font-medium' : 'bg-zinc-900 border border-zinc-800 text-zinc-300 hover:border-zinc-700'} text-xs rounded-lg transition-colors">Clase ${c}</button>`;
                });
                document.getElementById('classTags').innerHTML = tagsHtml;
            }

            function updateKPIs(data) {
                document.getElementById('kpiTotalRecords').innerText = data.length;
                
                const counts = {};
                data.forEach(d => {
                    const cls = d['NSE_Predicho'];
                    if (cls) counts[cls] = (counts[cls] || 0) + 1;
                });
                
                let dominant = '-';
                let maxCount = -1;
                for (const [cls, count] of Object.entries(counts)) {
                    if (count > maxCount) {
                        maxCount = count;
                        dominant = `Clase ${cls} (${count})`;
                    }
                }
                document.getElementById('kpiDominantClass').innerText = dominant;
            }

            function renderChart(data) {
                const counts = {};
                data.forEach(d => {
                    const cls = d['NSE_Predicho'];
                    if (cls) counts[cls] = (counts[cls] || 0) + 1;
                });

                const labels = Object.keys(counts).sort();
                const values = labels.map(l => counts[l]);

                const ctx = document.getElementById('nseChart').getContext('2d');
                
                if (nseChartInstance) {
                    nseChartInstance.destroy();
                }

                nseChartInstance = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: labels.map(l => 'Clase ' + l),
                        datasets: [{
                            data: values,
                            backgroundColor: ['#f97316', '#fbbf24', '#34d399', '#60a5fa', '#a78bfa'],
                            borderWidth: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'right',
                                labels: {
                                    boxWidth: 10,
                                    color: '#a1a1aa',
                                    font: { size: 10 }
                                }
                            }
                        },
                        cutout: '65%'
                    }
                });
            }

            function filterByClass(cls) {
                activeClassFilter = cls;
                setupClassTags();
                applyFilters();
            }

            function filterTable() {
                applyFilters();
            }

            function applyFilters() {
                const searchTxt = document.getElementById('searchInput').value.toLowerCase();
                
                const filtered = globalData.filter(row => {
                    const matchesClass = (activeClassFilter === 'ALL' || String(row['NSE_Predicho']).trim() === activeClassFilter);
                    const matchesSearch = Object.values(row).some(val => String(val).toLowerCase().includes(searchTxt));
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
        logging.info(f"Recibido archivo batch para inferencia: {file.filename}")
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            try:
                input_df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
            except UnicodeDecodeError:
                input_df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
        elif file.filename.endswith(('.xlsx', '.xls')):
            excel_file = io.BytesIO(contents)
            xls = pd.ExcelFile(excel_file)
            sheet_name = 'Base prueba IA nse' if 'Base prueba IA nse' in xls.sheet_names else xls.sheet_names[0]
            input_df = pd.read_excel(excel_file, sheet_name=sheet_name)
        else:
            raise HTTPException(status_code=400, detail="El archivo debe tener formato CSV o Excel (.xlsx/.xls válido).")
        
        if input_df.empty:
            raise HTTPException(status_code=400, detail="El archivo cargado está vacío.")
            
        # 1. Ejecutar análisis de Data Drift usando el método correcto
        try:
            drift_monitor = DataDriftMonitor()
            drift_monitor.initiate_data_drift_detection(input_df)
            logging.info("Análisis de Data Drift ejecutado correctamente.")
        except Exception as drift_err:
            logging.warning(f"No se pudo completar el análisis de drift: {drift_err}")

        # 2. Ejecutar pipeline de predicción con XGBoost
        pipeline = PredictPipeline()
        predictions = pipeline.predict(input_df)
        
        input_df.insert(0, 'NSE_Predicho', predictions)
        
        output = io.StringIO()
        input_df.to_csv(output, index=False)
        output.seek(0)
        
        logging.info("Procesamiento batch e inferencia completados con éxito.")
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=resultados_nse_predichos.csv"}
        )
        
    except Exception as e:
        logging.error(f"Error crítico en inferencia batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al procesar el modelo: {str(e)}")

@app.get("/download-drift-report")
def download_drift_report():
    report_path = os.path.join("artifacts", "data_drift_report.html")
    if os.path.exists(report_path):
        return FileResponse(report_path, media_type="text/html", filename="reporte_data_drift.html")
    raise HTTPException(status_code=404, detail="El reporte de Data Drift aún no ha sido generado. Sube un archivo primero.")

@app.get("/health")
def health_check():
    return {"status": "active", "model": "XGBoost Studio with Data Drift & Login", "target_audience": "Enterprise Analysts"}
// AttriBoost AI Application Logic

// Application State
let state = {
    currentModel: 'Linear',
    summary: {},
    attribution: {},
    reallocation: {},
    geminiApiKey: localStorage.getItem('gemini_api_key') || '',
    lookbackDays: 14,
    halfLife: 7,
    scaleUsers: 20000,
    scaleTps: 200000,
    attributionChart: null,
    benchmarkChart: null
};

// UI Elements
const els = {
    // Navigation
    navItems: document.querySelectorAll('.nav-item'),
    tabPanes: document.querySelectorAll('.tab-pane'),

    // KPIs (Overview Tab)
    kpiSpend: document.querySelector('#kpi-spend .kpi-value'),
    kpiConversions: document.querySelector('#kpi-conversions .kpi-value'),
    kpiRevenue: document.querySelector('#kpi-revenue .kpi-value'),
    kpiRoas: document.querySelector('#kpi-roas .kpi-value'),
    kpiScale: document.querySelector('#kpi-scale .kpi-value'),
    
    // Model Swapper
    modelTabs: document.querySelectorAll('.model-tab'),

    // Budget Optimization (Budget Tab)
    budgetSlider: document.querySelector('#total-budget-slider'),
    budgetValueText: document.querySelector('#budget-value'),
    reallocationTableBody: document.querySelector('#reallocation-table tbody'),
    
    // Gemini Chat Copilot (Chat Tab)
    chatMessages: document.querySelector('#chat-messages'),
    chatInput: document.querySelector('#chat-input'),
    chatSendBtn: document.querySelector('#chat-send-btn'),
    suggestionChips: document.querySelectorAll('.suggestion-chips .chip'),
    
    // Configuration Modal
    settingsToggle: document.querySelector('#settings-toggle'),
    settingsModal: document.querySelector('#settings-modal'),
    settingsClose: document.querySelector('#settings-close'),
    apiKeyInput: document.querySelector('#api-key-input'),
    scaleBtns: document.querySelectorAll('.scale-btn'),
    regenerateDataBtn: document.querySelector('#regenerate-data-btn'),
    
    // Sliders
    lookbackSlider: document.querySelector('#lookback-slider'),
    lookbackSliderValue: document.querySelector('#lookback-slider-value'),
    halfLifeSlider: document.querySelector('#half-life-slider'),
    halfLifeSliderValue: document.querySelector('#half-life-slider-value'),
    
    // Data Ingestion (Upload Tab)
    touchpointsFileInput: document.querySelector('#touchpoints-file-input'),
    conversionsFileInput: document.querySelector('#conversions-file-input'),
    uploadSubmitBtn: document.querySelector('#upload-submit-btn'),
    uploadStatus: document.querySelector('#upload-status'),

    // NVIDIA Benchmarks (Benchmark Tab)
    runBenchmarkBtn: document.querySelector('#run-benchmark-btn'),
    maxSpeedupText: document.querySelector('#max-speedup'),
    cpuTimeLabel: document.querySelector('#cpu-time-label'),
    gpuTimeLabel: document.querySelector('#gpu-time-label'),
    gpuStatusBadge: document.querySelector('#gpu-status'),
    gpuStatusText: document.querySelector('#gpu-text')
};

// Initialization
document.addEventListener('DOMContentLoaded', async () => {
    // Load Saved API Key
    if (state.geminiApiKey) {
        els.apiKeyInput.value = state.geminiApiKey;
    }
    
    // Check Health to see if GPU is active
    await checkApiHealth();
    
    // Initialize Default Charts FIRST
    initCharts();
    
    // Fetch Summary & Attribution Data SECOND
    await refreshDashboard();
    
    // Register UI Listeners
    registerEventListeners();
});

// Check Server Health (GPU Active check)
async function checkApiHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        if (data.gpu_available) {
            els.gpuStatusBadge.classList.remove('amber');
            els.gpuStatusBadge.classList.add('green');
            els.gpuStatusText.textContent = 'NVIDIA RAPIDS (Active GPU)';
        } else {
            els.gpuStatusBadge.classList.remove('green');
            els.gpuStatusBadge.classList.add('amber');
            els.gpuStatusText.textContent = 'CPU Fallback (Simulated GPU)';
        }
    } catch (e) {
        console.error("Health check failed:", e);
    }
}

// Fetch general stats and data summary
async function refreshDashboard() {
    try {
        // Fetch Summary Stats
        const summaryRes = await fetch('/api/summary');
        state.summary = await summaryRes.json();
        updateKpis(state.summary);
        
        // Fetch Attribution Results
        const attributionRes = await fetch(`/api/attribution?lookback_days=${state.lookbackDays}&half_life=${state.halfLife}`);
        state.attribution = await attributionRes.json();
        
        // Trigger Budget Reallocation Calculation
        await updateReallocation();
        
        // Redraw Attribution Chart
        renderAttributionChart();
    } catch (e) {
        console.error("Error refreshing dashboard data:", e);
    }
}

// Update KPI UI Elements
function updateKpis(summary) {
    els.kpiSpend.textContent = `$${summary.total_spend.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    els.kpiConversions.textContent = summary.total_conversions.toLocaleString();
    els.kpiRevenue.textContent = `$${summary.total_revenue.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    els.kpiRoas.textContent = `${summary.overall_roas.toFixed(2)}x`;
    els.kpiScale.textContent = summary.total_touchpoints.toLocaleString() + " rows";
}

// Update Budget Optimization Table
async function updateReallocation() {
    const budget = parseFloat(els.budgetSlider.value);
    els.budgetValueText.textContent = `$${budget.toLocaleString()}`;
    
    try {
        const res = await fetch('/api/reallocate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: state.currentModel,
                total_budget: budget,
                lookback_days: state.lookbackDays,
                half_life: state.halfLife
            })
        });
        
        state.reallocation = await res.json();
        
        // Populate Table
        els.reallocationTableBody.innerHTML = '';
        state.reallocation.recommendations.forEach(rec => {
            const tr = document.createElement('tr');
            
            const isOrganic = rec.channel === 'Organic Search';
            const changeClass = rec.percentage_change > 0 ? 'green' : (rec.percentage_change < 0 ? 'red' : '');
            const changeSymbol = rec.percentage_change > 0 ? '+' : '';
            const changeText = isOrganic ? 'N/A (Free)' : `${changeSymbol}${rec.percentage_change}%`;
            
            tr.innerHTML = `
                <td><strong>${rec.channel}</strong></td>
                <td><span class="badge ${rec.roas > 2 ? 'green' : 'amber'}">${rec.roas.toFixed(2)}x</span></td>
                <td>$${rec.current_spend.toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                <td>$${rec.recommended_spend.toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                <td><span class="delta-val ${changeClass}">${changeText}</span></td>
            `;
            els.reallocationTableBody.appendChild(tr);
        });
    } catch (e) {
        console.error("Error calculating budget reallocation:", e);
    }
}

// Initialize Empty Charts
function initCharts() {
    // 1. Attribution Chart
    const ctxAttr = document.getElementById('attribution-chart').getContext('2d');
    state.attributionChart = new Chart(ctxAttr, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Ad Spend ($)',
                    data: [],
                    backgroundColor: '#ef4444',
                    borderRadius: 6
                },
                {
                    label: 'Attributed Revenue ($)',
                    data: [],
                    backgroundColor: '#10b981',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9', font: { family: 'Inter' } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                }
            }
        }
    });

    // 2. Performance Benchmark Chart
    const ctxBench = document.getElementById('benchmark-chart').getContext('2d');
    state.benchmarkChart = new Chart(ctxBench, {
        type: 'bar',
        data: {
            labels: ['10% Scale', '50% Scale', '100% Scale'],
            datasets: [
                {
                    label: 'CPU (Pandas)',
                    data: [0, 0, 0],
                    backgroundColor: '#f59e0b',
                    borderRadius: 6
                },
                {
                    label: 'NVIDIA RAPIDS (cuDF)',
                    data: [0, 0, 0],
                    backgroundColor: '#76b900', // NVIDIA Green
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9', font: { family: 'Inter' } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                },
                y: {
                    title: { display: true, text: 'Time (milliseconds)', color: '#f1f5f9' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                }
            }
        }
    });
}

// Render data onto Attribution Chart
function renderAttributionChart() {
    const modelData = state.attribution[state.currentModel];
    if (!modelData) return;
    
    const labels = modelData.map(item => item.channel);
    const spend = modelData.map(item => item.spend);
    const revenue = modelData.map(item => item.attributed_revenue);
    
    state.attributionChart.data.labels = labels;
    state.attributionChart.data.datasets[0].data = spend;
    state.attributionChart.data.datasets[1].data = revenue;
    state.attributionChart.update();
    
    // Sync accessible table fallback
    const accBody = document.querySelector('#accessibility-table-body');
    if (accBody) {
        accBody.innerHTML = '';
        modelData.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${item.channel}</td>
                <td>$${item.spend.toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
                <td>$${item.attributed_revenue.toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
                <td>${item.roas.toFixed(2)}x</td>
            `;
            accBody.appendChild(tr);
        });
    }
}

// Register Listeners
function registerEventListeners() {
    // Navigation Tabs Switching
    els.navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active nav
            els.navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            // Switch Tab panes
            const targetTab = item.getAttribute('data-tab');
            els.tabPanes.forEach(pane => {
                pane.classList.remove('active');
                if (pane.id === `tab-${targetTab}`) {
                    pane.classList.add('active');
                }
            });
        });
    });

    // Model Selector Tabs
    els.modelTabs.forEach(tab => {
        tab.addEventListener('click', async (e) => {
            els.modelTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            state.currentModel = tab.getAttribute('data-model');
            renderAttributionChart();
            await updateReallocation();
        });
    });
    
    // Budget range slider
    els.budgetSlider.addEventListener('input', async () => {
        await updateReallocation();
    });
    
    // Lookback slider
    els.lookbackSlider.addEventListener('input', () => {
        state.lookbackDays = parseFloat(els.lookbackSlider.value);
        els.lookbackSliderValue.textContent = `${state.lookbackDays} days`;
    });
    els.lookbackSlider.addEventListener('change', async () => {
        await refreshDashboard();
    });
    
    // Half-Life slider
    els.halfLifeSlider.addEventListener('input', () => {
        state.halfLife = parseFloat(els.halfLifeSlider.value);
        els.halfLifeSliderValue.textContent = `${state.halfLife} days`;
    });
    els.halfLifeSlider.addEventListener('change', async () => {
        await refreshDashboard();
    });

    // Modal controls
    els.settingsToggle.addEventListener('click', () => {
        els.settingsModal.classList.toggle('active');
    });
    els.settingsClose.addEventListener('click', () => {
        els.settingsModal.classList.remove('active');
    });
    
    // API key inputs
    els.apiKeyInput.addEventListener('change', () => {
        state.geminiApiKey = els.apiKeyInput.value.trim();
        localStorage.setItem('gemini_api_key', state.geminiApiKey);
    });
    
    // Data scale selection
    els.scaleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            els.scaleBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.scaleUsers = parseInt(btn.getAttribute('data-users'));
            state.scaleTps = parseInt(btn.getAttribute('data-tps'));
        });
    });
    
    // Regenerate mock data triggers
    els.regenerateDataBtn.addEventListener('click', async () => {
        els.regenerateDataBtn.disabled = true;
        els.regenerateDataBtn.textContent = 'Generating...';
        
        try {
            const res = await fetch('/api/regenerate-data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    num_users: state.scaleUsers,
                    total_touchpoints: state.scaleTps
                })
            });
            const data = await res.json();
            if (data.status === 'success') {
                appendChatMessage('ai', `✅ Successfully generated new scale test dataset: **${state.scaleTps.toLocaleString()}** touchpoints. Cache updated.`);
                await refreshDashboard();
            }
        } catch (e) {
            console.error(e);
            appendChatMessage('ai', `⚠️ Error generating dataset: ${e.message}`);
        } finally {
            els.regenerateDataBtn.disabled = false;
            els.regenerateDataBtn.textContent = 'Regenerate Mock Dataset';
            els.settingsModal.classList.remove('active');
        }
    });

    // Custom Data Uploader API wiring
    els.uploadSubmitBtn.addEventListener('click', async () => {
        const tpFile = els.touchpointsFileInput.files[0];
        const convFile = els.conversionsFileInput.files[0];

        if (!tpFile || !convFile) {
            showUploadStatus('error', '⚠️ Please select both touchpoints.csv and conversions.csv files.');
            return;
        }

        els.uploadSubmitBtn.disabled = true;
        els.uploadSubmitBtn.textContent = 'Uploading & Parsing...';
        showUploadStatus('success', 'Processing files via server...');

        const formData = new FormData();
        formData.append('touchpoints_file', tpFile);
        formData.append('conversions_file', convFile);

        try {
            const response = await fetch('/api/upload-data', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok && result.status === 'success') {
                showUploadStatus('success', `✅ ${result.message}`);
                // Refresh dashboard with the uploaded data
                await refreshDashboard();
                
                // Clear inputs
                els.touchpointsFileInput.value = '';
                els.conversionsFileInput.value = '';
            } else {
                showUploadStatus('error', `❌ Upload failed: ${result.detail || 'Malformed CSV format.'}`);
            }
        } catch (e) {
            console.error(e);
            showUploadStatus('error', `❌ Network Error: ${e.message}`);
        } finally {
            els.uploadSubmitBtn.disabled = false;
            els.uploadSubmitBtn.textContent = '📤 Process Custom Datasets';
        }
    });
    
    // Run live benchmarks
    els.runBenchmarkBtn.addEventListener('click', async () => {
        els.runBenchmarkBtn.disabled = true;
        els.runBenchmarkBtn.textContent = 'Running Scale Tests...';
        
        try {
            const res = await fetch('/api/benchmark');
            const data = await res.json();
            
            // Plot timing charts
            const cpus = data.map(item => item.cpu_time_ms);
            const gpus = data.map(item => item.gpu_time_ms);
            
            state.benchmarkChart.data.datasets[0].data = cpus;
            state.benchmarkChart.data.datasets[1].data = gpus;
            state.benchmarkChart.update();
            
            // Update labels
            const lastItem = data[data.length - 1];
            els.maxSpeedupText.textContent = `${lastItem.speedup.toFixed(1)}x`;
            els.cpuTimeLabel.textContent = `${lastItem.cpu_time_ms.toFixed(1)} ms`;
            els.gpuTimeLabel.textContent = `${lastItem.gpu_time_ms.toFixed(1)} ms`;
            
            const isSimulated = lastItem.gpu_mode.includes('Simulated');
            const modeLabel = isSimulated ? 'Projected GPU (Simulated)' : 'NVIDIA GPU (RAPIDS)';
            const timingNote = isSimulated ? '\n*(Note: Running in CPU fallback mode; GPU performance projected based on cuDF benchmark speedup ratios).*' : '';
            
            appendChatMessage('ai', `⚡ **Benchmark Completed!** (${lastItem.gpu_mode})\n\nProcessed **${lastItem.rows_processed.toLocaleString()}** join operations.\n- **CPU Pandas:** ${lastItem.cpu_time_ms.toFixed(1)} ms\n- **${modeLabel}:** ${lastItem.gpu_time_ms.toFixed(1)} ms\n- **GPU Speedup Factor:** **${lastItem.speedup.toFixed(1)}x faster**!${timingNote}\n\nThis speedup allows marketers to perform real-time path calculations and budget shifts without waiting for long database pipeline processing.`);
        } catch (e) {
            console.error(e);
            appendChatMessage('ai', `⚠️ Failed running benchmark: ${e.message}`);
        } finally {
            els.runBenchmarkBtn.disabled = false;
            els.runBenchmarkBtn.textContent = '⚡ Run Performance Scale Test';
        }
    });
    
    // Send Chat Copilot Messages
    els.chatSendBtn.addEventListener('click', () => sendCopilotMessage());
    els.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendCopilotMessage();
    });
    
    // Suggestion chips handler
    els.suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            els.chatInput.value = chip.getAttribute('data-prompt');
            sendCopilotMessage();
        });
    });
}

// Upload status indicator helper
function showUploadStatus(type, message) {
    els.uploadStatus.className = `upload-status ${type}`;
    els.uploadStatus.textContent = message;
}

// Append Chat Message
function appendChatMessage(sender, text) {
    const msg = document.createElement('div');
    msg.classList.add('message', sender);
    
    // Minimal markdown conversions
    let formattedText = text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/⚠️/g, '⚠️')
        .replace(/⚡/g, '⚡')
        .replace(/✅/g, '✅');
        
    msg.innerHTML = formattedText;
    els.chatMessages.appendChild(msg);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

// Send user question to Gemini API
async function sendCopilotMessage() {
    const text = els.chatInput.value.trim();
    if (!text) return;
    
    // Append user message
    appendChatMessage('user', text);
    els.chatInput.value = '';
    
    // Loading indicator
    const loader = document.createElement('div');
    loader.classList.add('message', 'ai');
    loader.innerHTML = 'Thinking... <span class="status-dot amber pulse" style="display:inline-block; margin-left:5px;"></span>';
    els.chatMessages.appendChild(loader);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Gemini-Key': state.geminiApiKey
            },
            body: JSON.stringify({
                message: text,
                model: state.currentModel,
                lookback_days: state.lookbackDays,
                half_life: state.halfLife
            })
        });
        
        const data = await response.json();
        
        // Remove loader and append reply
        els.chatMessages.removeChild(loader);
        appendChatMessage('ai', data.reply);
    } catch (e) {
        console.error(e);
        els.chatMessages.removeChild(loader);
        appendChatMessage('ai', `⚠️ Network error connecting to backend: ${e.message}`);
    }
}

// ── Apply saved theme immediately ──
const THEME_KEY = 'mcp_theme';
(function() {
    const saved = localStorage.getItem(THEME_KEY) || 'default';
    document.documentElement.setAttribute('data-theme', saved);
})();

// ── Chart colours derived from theme ──
function getThemeColors() {
    const style = getComputedStyle(document.documentElement);
    return {
        blue:   style.getPropertyValue('--chart-blue').trim()   || '#3498db',
        purple: style.getPropertyValue('--chart-purple').trim() || '#9b59b6',
    };
}

let ws = null, metricsData = null, updateInterval = null;
let agentChart = null, llmChart = null;

function formatTimestamp(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('en-US',{ hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false });
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},${alpha})`;
}

function initCharts() {
    const colors = getThemeColors();
    const baseOptions = {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: (ctx) => ctx.parsed.y.toFixed(2)+'s' } }
        },
        scales: {
            y: { beginAtZero: true, ticks: { callback: (v) => v.toFixed(2)+'s' } },
            x: { display: true, ticks: { maxRotation:45, minRotation:45, autoSkip:true, maxTicksLimit:8, font:{size:10} } }
        }
    };

    agentChart = new Chart(document.getElementById('agentChart').getContext('2d'), {
        type: 'line', options: baseOptions,
        data: { labels:[], datasets:[{ label:'Agent Response Time (s)', data:[],
            borderColor: colors.blue,
            backgroundColor: hexToRgba(colors.blue.padEnd(7,'0'), 0.1),
            tension:0.4, fill:true }] }
    });

    llmChart = new Chart(document.getElementById('llmChart').getContext('2d'), {
        type: 'line', options: baseOptions,
        data: { labels:[], datasets:[{ label:'LLM Response Time (s)', data:[],
            borderColor: colors.purple,
            backgroundColor: hexToRgba(colors.purple.padEnd(7,'0'), 0.1),
            tension:0.4, fill:true }] }
    });
}

function updateMetrics(data) {
    metricsData = data;
    document.getElementById('toolCallsValue').textContent = data.tools.total_calls.toLocaleString();
    document.getElementById('llmCallsValue').textContent  = data.llm.calls.toLocaleString();
    document.getElementById('agentRunsValue').textContent = data.agent.runs.toLocaleString();
    document.getElementById('errorRateValue').textContent = data.agent.error_rate.toFixed(2)+'%';
    document.getElementById('totalErrorsValue').textContent = data.overall_errors.toLocaleString();
    updateAgentChart(data.agent.times);
    updateLLMChart(data.llm.times);
    updateToolsDisplay(data.tools.per_tool);
}

function updateAgentChart(timesData) {
    if (!agentChart||!timesData) return;
    const timestamps=timesData.timestamps||[], durations=timesData.durations||[];
    if (!timestamps.length) return;
    agentChart.data.labels = timestamps.map(formatTimestamp);
    agentChart.data.datasets[0].data = durations;
    agentChart.update('none');
}

function updateLLMChart(timesData) {
    if (!llmChart||!timesData) return;
    const timestamps=timesData.timestamps||[], durations=timesData.durations||[];
    if (!timestamps.length) return;
    llmChart.data.labels = timestamps.map(formatTimestamp);
    llmChart.data.datasets[0].data = durations;
    llmChart.update('none');
}

function updateToolsDisplay(toolsData) {
    const container = document.getElementById('toolsContainer');
    if (!toolsData||!Object.keys(toolsData).length) { container.innerHTML='<div class="no-data">No tool data available yet</div>'; return; }
    container.innerHTML = '';
    Object.entries(toolsData).sort((a,b)=>b[1].calls-a[1].calls).forEach(([toolName,stats]) => {
        const card = document.createElement('div'); card.className='tool-card';
        const errorBadge = stats.errors>0 ? `<span class="error-badge">${stats.errors} errors</span>` : '';
        card.innerHTML = `
            <div class="tool-name">${toolName} ${errorBadge}</div>
            <div class="tool-stats">
                <div class="tool-stat"><div class="tool-stat-label">Calls</div><div class="tool-stat-value">${stats.calls.toLocaleString()}</div></div>
                <div class="tool-stat"><div class="tool-stat-label">Avg Time</div><div class="tool-stat-value">${stats.avg_time.toFixed(2)}s</div></div>
                <div class="tool-stat"><div class="tool-stat-label">Errors</div><div class="tool-stat-value ${stats.errors>0?'metric-error':''}">${stats.errors}</div></div>
            </div>`;
        container.appendChild(card);
    });
}

function updateStatus(connected) {
    const indicator = document.getElementById('statusIndicator');
    const text      = document.getElementById('statusText');
    if (connected) { indicator.classList.add('connected'); text.textContent='Live metrics stream active'; }
    else           { indicator.classList.remove('connected'); text.textContent='Disconnected - Attempting to reconnect...'; }
}

function requestMetrics() {
    if (ws&&ws.readyState===WebSocket.OPEN) ws.send(JSON.stringify({type:'metrics_request'}));
}

function connectWebSocket() {
    const wsUrl = `ws://${window.location.hostname||'localhost'}:8765`;
    try { ws = new WebSocket(wsUrl); } catch(e) { updateStatus(false); setTimeout(connectWebSocket,3000); return; }
    ws.onopen = () => {
        updateStatus(true); requestMetrics();
        if (updateInterval) clearInterval(updateInterval);
        updateInterval = setInterval(requestMetrics, 2000);
    };
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type==='metrics_response') updateMetrics(data.metrics);
    };
    ws.onerror = () => updateStatus(false);
    ws.onclose = () => { updateStatus(false); if (updateInterval){clearInterval(updateInterval);updateInterval=null;} setTimeout(connectWebSocket,3000); };
}

initCharts();
connectWebSocket();
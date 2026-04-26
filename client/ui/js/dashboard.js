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
        red:    style.getPropertyValue('--chart-red').trim()    || '#e74c3c',
        green:  style.getPropertyValue('--chart-green').trim()  || '#2ecc71',
        orange: style.getPropertyValue('--chart-orange').trim() || '#e67e22',
    };
}

let ws = null, metricsData = null, updateInterval = null;
let agentChart = null, llmChart = null;

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatTimestamp(ts) {
    return new Date(ts * 1000).toLocaleTimeString('en-US', {
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
    });
}

function hexToRgba(hex, alpha) {
    hex = hex.padEnd(7, '0');
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

function fmtSec(v) {
    if (v === 0 || v == null) return '—';
    return v.toFixed(2) + 's';
}

// ── Charts ───────────────────────────────────────────────────────────────────

function initCharts() {
    const colors = getThemeColors();
    const baseOptions = {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: (ctx) => ctx.parsed.y.toFixed(2) + 's' } }
        },
        scales: {
            y: { beginAtZero: true, ticks: { callback: (v) => v.toFixed(2) + 's' } },
            x: { display: true, ticks: { maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } } }
        }
    };

    agentChart = new Chart(document.getElementById('agentChart').getContext('2d'), {
        type: 'line', options: baseOptions,
        data: { labels: [], datasets: [{ label: 'Agent Response Time (s)', data: [],
            borderColor: colors.blue,
            backgroundColor: hexToRgba(colors.blue, 0.1),
            tension: 0.4, fill: true }] }
    });

    llmChart = new Chart(document.getElementById('llmChart').getContext('2d'), {
        type: 'line', options: baseOptions,
        data: { labels: [], datasets: [{ label: 'LLM Response Time (s)', data: [],
            borderColor: colors.purple,
            backgroundColor: hexToRgba(colors.purple, 0.1),
            tension: 0.4, fill: true }] }
    });
}

// ── Inline histogram (CSS bars, no canvas) ───────────────────────────────────

function renderHistogram(containerId, histogram, buckets) {
    const container = document.getElementById(containerId);
    if (!container || !histogram) return;

    const bucketLabels = buckets || Object.keys(histogram);
    const values = bucketLabels.map(b => histogram[b] || 0);
    const maxVal = Math.max(...values, 1);

    // Colour gradient: green → yellow → red by bucket position
    const BAR_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'];

    container.innerHTML = `
        <div class="histogram-title">Latency Distribution (last ${(metricsData && metricsData.window_size) || 1000} samples)</div>
        <div class="histogram-bars">
            ${bucketLabels.map((label, i) => {
                const val = values[i];
                const pct = (val / maxVal * 100).toFixed(1);
                const color = BAR_COLORS[Math.min(i, BAR_COLORS.length - 1)];
                return `
                    <div class="histogram-bar-group">
                        <div class="histogram-bar-wrap">
                            <div class="histogram-bar" style="height:${pct}%;background:${color}" title="${val} samples"></div>
                        </div>
                        <div class="histogram-bar-label">${label}</div>
                        <div class="histogram-bar-count">${val}</div>
                    </div>`;
            }).join('')}
        </div>`;
}

// ── Percentile stat cards ─────────────────────────────────────────────────────

function updatePercentileCards(prefix, data) {
    const p50El  = document.getElementById(`${prefix}P50Value`);
    const p95El  = document.getElementById(`${prefix}P95Value`);
    const p99El  = document.getElementById(`${prefix}P99Value`);
    if (p50El) p50El.textContent = fmtSec(data.p50);
    if (p95El) p95El.textContent = fmtSec(data.p95);
    if (p99El) p99El.textContent = fmtSec(data.p99);
}

// ── Failure kinds breakdown ───────────────────────────────────────────────────

function updateFailureKinds(failureKinds) {
    const container = document.getElementById('failureKindsContainer');
    if (!container) return;

    const entries = Object.entries(failureKinds || {}).filter(([, v]) => v > 0);
    if (!entries.length) {
        container.innerHTML = '<div class="no-data">No failures recorded</div>';
        return;
    }

    const total = entries.reduce((s, [, v]) => s + v, 0);
    const KIND_COLORS = {
        retryable:      '#f1c40f',
        user_error:     '#e67e22',
        upstream_error: '#e74c3c',
        internal_error: '#8e44ad',
    };
    const KIND_LABELS = {
        retryable:      '🔄 Retryable',
        user_error:     '🙋 User Error',
        upstream_error: '🌐 Upstream',
        internal_error: '🐛 Internal',
    };

    container.innerHTML = entries
        .sort((a, b) => b[1] - a[1])
        .map(([kind, count]) => {
            const pct = (count / total * 100).toFixed(1);
            const color = KIND_COLORS[kind] || '#555';
            const label = KIND_LABELS[kind] || kind;
            return `
                <div class="failure-kind-card">
                    <div class="failure-kind-label">${label}</div>
                    <div class="failure-kind-value" style="color:${color}">${count}</div>
                    <div class="failure-kind-bar-wrap">
                        <div class="failure-kind-bar" style="width:${pct}%;background:${color}"></div>
                    </div>
                    <div class="failure-kind-pct">${pct}%</div>
                </div>`;
        }).join('');
}

// ── Main update ───────────────────────────────────────────────────────────────

function updateMetrics(data) {
    metricsData = data;

    // Count cards
    document.getElementById('toolCallsValue').textContent  = data.tools.total_calls.toLocaleString();
    document.getElementById('llmCallsValue').textContent   = data.llm.calls.toLocaleString();
    document.getElementById('agentRunsValue').textContent  = data.agent.runs.toLocaleString();
    document.getElementById('errorRateValue').textContent  = data.agent.error_rate.toFixed(2) + '%';
    document.getElementById('totalErrorsValue').textContent = data.overall_errors.toLocaleString();

    // Percentile cards
    updatePercentileCards('agent', data.agent);
    updatePercentileCards('llm',   data.llm);

    // Sparkline charts
    updateLineChart(agentChart, data.agent.times);
    updateLineChart(llmChart,   data.llm.times);

    // Histograms
    const buckets = data.buckets || null;
    renderHistogram('agentHistogram', data.agent.histogram, buckets);
    renderHistogram('llmHistogram',   data.llm.histogram,   buckets);

    // Failure kinds
    updateFailureKinds(data.failure_kinds);

    // Tool cards
    updateToolsDisplay(data.tools.per_tool, buckets);
}

function updateLineChart(chart, timesData) {
    if (!chart || !timesData) return;
    const timestamps = timesData.timestamps || [];
    const durations  = timesData.durations  || [];
    if (!timestamps.length) return;
    chart.data.labels            = timestamps.map(formatTimestamp);
    chart.data.datasets[0].data  = durations;
    chart.update('none');
}

function updateToolsDisplay(toolsData, buckets) {
    const container = document.getElementById('toolsContainer');
    if (!toolsData || !Object.keys(toolsData).length) {
        container.innerHTML = '<div class="no-data">No tool data available yet</div>';
        return;
    }
    container.innerHTML = '';

    Object.entries(toolsData)
        .sort((a, b) => b[1].calls - a[1].calls)
        .forEach(([toolName, stats]) => {
            const card = document.createElement('div');
            card.className = 'tool-card';

            const errorBadge = stats.errors > 0
                ? `<span class="error-badge">${stats.errors} errors</span>`
                : '';

            // Inline micro-histogram for the tool
            const hist = stats.histogram;
            const histHtml = hist ? buildMiniHistogram(hist, buckets) : '';

            card.innerHTML = `
                <div class="tool-name">${toolName} ${errorBadge}</div>
                <div class="tool-stats">
                    <div class="tool-stat">
                        <div class="tool-stat-label">Calls</div>
                        <div class="tool-stat-value">${stats.calls.toLocaleString()}</div>
                    </div>
                    <div class="tool-stat">
                        <div class="tool-stat-label">Avg</div>
                        <div class="tool-stat-value">${fmtSec(stats.avg_time)}</div>
                    </div>
                    <div class="tool-stat">
                        <div class="tool-stat-label">p50</div>
                        <div class="tool-stat-value">${fmtSec(stats.p50)}</div>
                    </div>
                    <div class="tool-stat">
                        <div class="tool-stat-label">p95</div>
                        <div class="tool-stat-value">${fmtSec(stats.p95)}</div>
                    </div>
                    <div class="tool-stat">
                        <div class="tool-stat-label">p99</div>
                        <div class="tool-stat-value">${fmtSec(stats.p99)}</div>
                    </div>
                    <div class="tool-stat">
                        <div class="tool-stat-label">Errors</div>
                        <div class="tool-stat-value ${stats.errors > 0 ? 'metric-error' : ''}">${stats.errors}</div>
                    </div>
                </div>
                ${histHtml}`;

            container.appendChild(card);
        });
}

function buildMiniHistogram(histogram, buckets) {
    const labels = buckets || Object.keys(histogram);
    const values = labels.map(b => histogram[b] || 0);
    const maxVal = Math.max(...values, 1);
    const BAR_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'];

    const bars = labels.map((label, i) => {
        const pct  = (values[i] / maxVal * 100).toFixed(1);
        const color = BAR_COLORS[Math.min(i, BAR_COLORS.length - 1)];
        return `<div class="mini-bar-group" title="${label}: ${values[i]}">
            <div class="mini-bar-wrap">
                <div class="mini-bar" style="height:${pct}%;background:${color}"></div>
            </div>
        </div>`;
    }).join('');

    return `<div class="mini-histogram">${bars}</div>`;
}

// ── Status ────────────────────────────────────────────────────────────────────

function updateStatus(connected) {
    const indicator = document.getElementById('statusIndicator');
    const text      = document.getElementById('statusText');
    if (connected) {
        indicator.classList.add('connected');
        text.textContent = 'Live metrics stream active';
    } else {
        indicator.classList.remove('connected');
        text.textContent = 'Disconnected — attempting to reconnect...';
    }
}

// ── Reset button ──────────────────────────────────────────────────────────────

function setupResetButton() {
    const btn = document.getElementById('resetMetricsBtn');
    if (!btn) return;
    btn.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'reset_metrics' }));
            btn.textContent = '✓ Reset';
            btn.disabled = true;
            setTimeout(() => { btn.textContent = '↺ Reset'; btn.disabled = false; }, 1500);
        }
    });
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

function requestMetrics() {
    if (ws && ws.readyState === WebSocket.OPEN)
        ws.send(JSON.stringify({ type: 'metrics_request' }));
}

function connectWebSocket() {
    const wsUrl = `ws://${window.location.hostname || 'localhost'}:8765`;
    try { ws = new WebSocket(wsUrl); }
    catch(e) { updateStatus(false); setTimeout(connectWebSocket, 3000); return; }

    ws.onopen = () => {
        updateStatus(true);
        requestMetrics();
        if (updateInterval) clearInterval(updateInterval);
        updateInterval = setInterval(requestMetrics, 2000);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'metrics_response') {
            updateMetrics(data.metrics);
        } else if (data.type === 'metrics_reset') {
            // Server confirmed reset — request fresh data immediately
            requestMetrics();
        }
    };

    ws.onerror = () => updateStatus(false);
    ws.onclose = () => {
        updateStatus(false);
        if (updateInterval) { clearInterval(updateInterval); updateInterval = null; }
        setTimeout(connectWebSocket, 3000);
    };
}

initCharts();
setupResetButton();
connectWebSocket();
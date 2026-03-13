// ============================================================
// THEME SYSTEM
// ============================================================
const THEME_KEY = 'mcp_theme';

// Reads --theme-list CSS variable from :root in shared.css.
// Works everywhere including file:// with no I/O or CORS issues.
function discoverThemes() {
    const list = getComputedStyle(document.documentElement)
        .getPropertyValue('--theme-list')
        .trim()
        .replace(/"/g, '');
    return list ? list.split(',').map(s => s.trim()).filter(Boolean) : ['default'];
}

function getThemeSwatches(themeId) {
    const prev = document.documentElement.getAttribute('data-theme');
    document.documentElement.setAttribute('data-theme', themeId);
    const style = getComputedStyle(document.documentElement);
    const s1 = style.getPropertyValue('--swatch1').trim();
    const s2 = style.getPropertyValue('--swatch2').trim();
    const s3 = style.getPropertyValue('--swatch3').trim();
    document.documentElement.setAttribute('data-theme', prev || 'default');
    return [s1, s2, s3];
}

function themeIdToLabel(id) {
    const overrides = { optimus: 'Optimus Prime', tokyo: 'Tokyo Night' };
    return overrides[id] ?? id.charAt(0).toUpperCase() + id.slice(1);
}

function buildThemeDropdown() {
    const themes = discoverThemes();
    const dropdown = document.getElementById('themeDropdown');
    dropdown.innerHTML = themes.map(id => {
        const [s1, s2, s3] = getThemeSwatches(id);
        return `
            <div class="theme-option" data-theme="${id}" onclick="applyTheme('${id}')">
                <div class="theme-option-swatches">
                    <div class="theme-opt-swatch" style="background:${s1}"></div>
                    <div class="theme-opt-swatch" style="background:${s2}"></div>
                    <div class="theme-opt-swatch" style="background:${s3}"></div>
                </div>
                ${themeIdToLabel(id)}
                <span class="check">&#x2713;</span>
            </div>`;
    }).join('');
}

function applyTheme(themeName) {
    document.documentElement.setAttribute('data-theme', themeName);
    localStorage.setItem(THEME_KEY, themeName);
    document.querySelectorAll('.theme-option').forEach(el => {
        el.classList.toggle('active', el.dataset.theme === themeName);
    });
    const btn = document.getElementById('themeBtn');
    if (btn) btn.title = themeIdToLabel(themeName);
    closeThemeDropdown();
}

function toggleThemeDropdown(e) {
    e.stopPropagation();
    document.getElementById('themeDropdown').classList.toggle('open');
}

function closeThemeDropdown() {
    document.getElementById('themeDropdown').classList.remove('open');
}

document.addEventListener('click', (e) => {
    if (!e.target.closest('#themeWrapper')) closeThemeDropdown();
});

// Build dropdown then restore saved theme
(function() {
    const saved = localStorage.getItem(THEME_KEY) || 'default';
    document.documentElement.setAttribute('data-theme', saved);
    buildThemeDropdown();
    applyTheme(saved);
})();

// ============================================================
// MODAL SYSTEM
// ============================================================
(function() {
    // Inject modal HTML into body on first use
    function ensureModalDOM() {
        if (document.getElementById('mcpModal')) return;
        const el = document.createElement('div');
        el.innerHTML = `
            <div id="mcpModalBackdrop" class="mcp-modal-backdrop"></div>
            <div id="mcpModal" class="mcp-modal" role="dialog" aria-modal="true" aria-labelledby="mcpModalTitle">
                <div class="mcp-modal-dialog">
                    <div class="mcp-modal-header">
                        <h5 class="mcp-modal-title" id="mcpModalTitle"></h5>
                    </div>
                    <div class="mcp-modal-body" id="mcpModalBody"></div>
                    <div class="mcp-modal-footer" id="mcpModalFooter"></div>
                </div>
            </div>`;
        document.body.appendChild(el.children[0]); // backdrop
        document.body.appendChild(el.children[0]); // modal
    }

    function openModal() {
        ensureModalDOM();
        document.getElementById('mcpModalBackdrop').classList.add('show');
        document.getElementById('mcpModal').classList.add('show');
        document.addEventListener('keydown', onModalKeydown);
    }

    function closeModal() {
        const backdrop = document.getElementById('mcpModalBackdrop');
        const modal    = document.getElementById('mcpModal');
        if (backdrop) backdrop.classList.remove('show');
        if (modal)    modal.classList.remove('show');
        document.removeEventListener('keydown', onModalKeydown);
    }

    function onModalKeydown(e) {
        if (e.key === 'Escape') closeModal();
    }

    /**
     * showConfirm({ title, message, confirmText, cancelText, danger }) → Promise<boolean>
     */
    window.showConfirm = function({ title = 'Confirm', message = '', confirmText = 'OK', cancelText = 'Cancel', danger = false } = {}) {
        return new Promise(resolve => {
            ensureModalDOM();
            document.getElementById('mcpModalTitle').textContent = title;
            document.getElementById('mcpModalBody').textContent  = message;

            const footer = document.getElementById('mcpModalFooter');
            footer.innerHTML = '';

            const cancelBtn = document.createElement('button');
            cancelBtn.textContent = cancelText;
            cancelBtn.className   = 'mcp-modal-btn mcp-modal-btn-secondary';
            cancelBtn.onclick = () => { closeModal(); resolve(false); };

            const confirmBtn = document.createElement('button');
            confirmBtn.textContent = confirmText;
            confirmBtn.className   = 'mcp-modal-btn' + (danger ? ' mcp-modal-btn-danger' : ' mcp-modal-btn-primary');
            confirmBtn.onclick = () => { closeModal(); resolve(true); };

            footer.appendChild(cancelBtn);
            footer.appendChild(confirmBtn);

            // Backdrop click cancels
            document.getElementById('mcpModalBackdrop').onclick = () => { closeModal(); resolve(false); };

            openModal();
            confirmBtn.focus();
        });
    };

    /**
     * showAlert({ title, message, buttonText }) → Promise<void>
     */
    window.showAlert = function({ title = 'Notice', message = '', buttonText = 'OK' } = {}) {
        return new Promise(resolve => {
            ensureModalDOM();
            document.getElementById('mcpModalTitle').textContent = title;
            document.getElementById('mcpModalBody').textContent  = message;

            const footer = document.getElementById('mcpModalFooter');
            footer.innerHTML = '';

            const okBtn = document.createElement('button');
            okBtn.textContent = buttonText;
            okBtn.className   = 'mcp-modal-btn mcp-modal-btn-primary';
            okBtn.onclick = () => { closeModal(); resolve(); };

            footer.appendChild(okBtn);
            document.getElementById('mcpModalBackdrop').onclick = () => { closeModal(); resolve(); };

            openModal();
            okBtn.focus();
        });
    };
})();

// CORE APP STATE
// ============================================================
const status = document.getElementById("status");
status.textContent = "Connecting…";

const MULTI_AGENT_KEY   = "mcp_multi_agent_enabled";
const CURRENT_SESSION_KEY = "mcp_current_session";
const chat    = document.getElementById("chat");
const input   = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
sendBtn.disabled = true;

const modelSelect = document.getElementById("modelSelect");
const modeStatus  = document.getElementById("modeStatus");

let isProcessing           = false;
let thinkingIndicator      = null;
let multiAgentEnabled      = loadMultiAgentSetting();
let lastResponseWasMultiAgent = false;
let lastSentMessage        = null;
let currentSessionId       = null;
let allSessions            = [];
let sessionsSidebarOpen    = false;
let openMenuSessionId      = null;
let editingSessionId       = null;

let logPanelOpen    = false;
let logAutoscroll   = true;
let logFilters      = new Set(['USER','ASSISTANT','DEBUG','INFO','WARNING','ERROR']);
let logWs           = null;
let systemMonitorOpen   = false;
let systemStatsEnabled  = true;

// ============================================================
// SESSION MANAGEMENT
// ============================================================
function toggleSessionsSidebar() {
    sessionsSidebarOpen = !sessionsSidebarOpen;
    const sidebar = document.getElementById('sessionsSidebar');
    const btn     = document.getElementById('hamburgerBtn');
    if (sessionsSidebarOpen) {
        sidebar.classList.add('open');
        btn.classList.add('open');
        loadSessions();
    } else {
        sidebar.classList.remove('open');
        btn.classList.remove('open');
        openMenuSessionId = null;
        editingSessionId  = null;
    }
}

document.addEventListener('click', (e) => {
    if (e.target.closest('.session-menu-btn') || e.target.closest('.session-submenu')) return;
    if (openMenuSessionId !== null) {
        openMenuSessionId = null;
        if (allSessions && allSessions.length > 0) renderSessions(allSessions);
    }
});

function startNewSession() {
    currentSessionId = null;
    localStorage.setItem(CURRENT_SESSION_KEY, '');
    chat.innerHTML = "";
    ws.send(JSON.stringify({ type: "new_session" }));
}

function loadSessions() {
    ws.send(JSON.stringify({ type: "list_sessions" }));
}

function formatSessionDate(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    const now   = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const sessionDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
    const diffDays = Math.floor((today - sessionDate) / 86400000);
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    return date.toLocaleDateString('en-US', { weekday:'long', month:'short', day:'numeric', year:'numeric' });
}

function renderSessions(sessions) {
    const list = document.getElementById('sessionsList');
    list.innerHTML = '';
    sessions.forEach(session => {
        const item = document.createElement('div');
        item.className = 'session-item';
        if (session.id === currentSessionId) item.classList.add('active');

        if (editingSessionId === session.id) {
            item.classList.add('editing');
            const inp = document.createElement('input');
            inp.type = 'text'; inp.className = 'session-edit-input';
            inp.value = session.name || 'Untitled Session';
            inp.onclick = (e) => e.stopPropagation();

            const actions  = document.createElement('div'); actions.className = 'session-edit-actions';
            const saveBtn2 = document.createElement('button'); saveBtn2.textContent = '✓'; saveBtn2.className = 'session-edit-btn';
            saveBtn2.onclick = (e) => { e.stopPropagation(); saveSessionRename(session.id, inp.value); };
            const cancelBtn = document.createElement('button'); cancelBtn.textContent = '✕'; cancelBtn.className = 'session-edit-cancel';
            cancelBtn.onclick = (e) => { e.stopPropagation(); cancelSessionEdit(); };

            actions.appendChild(saveBtn2); actions.appendChild(cancelBtn);
            item.appendChild(inp); item.appendChild(actions);
            setTimeout(() => inp.focus(), 10);
        } else {
            const textContainer = document.createElement('div');
            textContainer.style.cssText = 'flex:1;min-width:0;display:flex;flex-direction:column;';
            const text = document.createElement('span'); text.className = 'session-item-text';
            text.textContent = session.name || 'Untitled Session';
            const dateLabel = document.createElement('div'); dateLabel.className = 'session-item-date';
            dateLabel.textContent = formatSessionDate(session.created_at);
            textContainer.appendChild(text); textContainer.appendChild(dateLabel);

            const menuBtn2 = document.createElement('button'); menuBtn2.className = 'session-menu-btn';
            menuBtn2.textContent = '⋮';
            menuBtn2.onclick = (e) => { e.stopPropagation(); toggleSessionMenu(session.id); };
            if (openMenuSessionId === session.id) menuBtn2.classList.add('active');

            const submenu    = document.createElement('div'); submenu.className = 'session-submenu';
            if (openMenuSessionId === session.id) submenu.classList.add('open');

            const editItem   = document.createElement('div'); editItem.className = 'session-submenu-item';
            editItem.innerHTML = '✏️ Edit';
            editItem.onclick = (e) => { e.stopPropagation(); startSessionEdit(session.id); };

            const deleteItem = document.createElement('div'); deleteItem.className = 'session-submenu-item delete';
            deleteItem.innerHTML = '🗑️ Delete';
            deleteItem.onclick = (e) => { e.stopPropagation(); deleteSession(session.id); };

            submenu.appendChild(editItem); submenu.appendChild(deleteItem);
            item.appendChild(textContainer); item.appendChild(menuBtn2); item.appendChild(submenu);
            item.onclick = () => selectSession(session.id);
        }
        list.appendChild(item);
    });
}

function filterSessions() {
    const searchTerm = document.getElementById('sessionSearch').value;
    if (!searchTerm) { renderSessions(allSessions); return; }
    const regex = new RegExp(searchTerm, 'i');
    renderSessions(allSessions.filter(s => regex.test(s.name)));
}

function toggleSessionMenu(sessionId) {
    openMenuSessionId = (openMenuSessionId === sessionId) ? null : sessionId;
    renderSessions(allSessions);
}
function startSessionEdit(sessionId)  { openMenuSessionId = null; editingSessionId = sessionId; renderSessions(allSessions); }
function cancelSessionEdit()           { editingSessionId = null; renderSessions(allSessions); }

function saveSessionRename(sessionId, newName) {
    if (!newName || !newName.trim()) { showAlert({ title: 'Validation Error', message: 'Session name cannot be empty.' }); return; }
    ws.send(JSON.stringify({ type:'rename_session', session_id:sessionId, name:newName.trim() }));
    editingSessionId = null;
}

function deleteSession(sessionId) {
    showConfirm({
        title:       'Delete Conversation',
        message:     'Are you sure you want to delete this conversation? This cannot be undone.',
        confirmText: 'Delete',
        cancelText:  'Cancel',
        danger:      true
    }).then(confirmed => {
        if (!confirmed) return;
        const wasCurrent = (sessionId === currentSessionId);
        ws.send(JSON.stringify({ type:'delete_session', session_id:sessionId }));
        openMenuSessionId = null;
        if (wasCurrent) {
            currentSessionId = null;
            localStorage.setItem(CURRENT_SESSION_KEY, '');
            chat.innerHTML = '';
            ws.send(JSON.stringify({ type: "new_session" }));
        }
    });
}

function selectSession(sessionId) {
    ws.send(JSON.stringify({ type:"load_session", session_id:sessionId }));
    currentSessionId = sessionId;
    localStorage.setItem(CURRENT_SESSION_KEY, sessionId);
}

// ============================================================
// MULTI-AGENT
// ============================================================
function loadMultiAgentSetting() {
    const saved = localStorage.getItem(MULTI_AGENT_KEY);
    return saved === null ? false : saved === "true";
}
function saveMultiAgentSetting(enabled) { localStorage.setItem(MULTI_AGENT_KEY, enabled.toString()); }

function updateStatusWithMode() {
    status.innerHTML = modelSelect.value ? "Model: " + modelSelect.value : "Connected";
}

function stopProcessing() {}

function resetControlButtons() {
    isProcessing = false;
    sendBtn.style.display  = 'flex';
    sendBtn.disabled       = false;
    sendBtn.style.opacity  = '1';
    sendBtn.style.cursor   = 'pointer';
    updateStatusWithMode();
}

// ============================================================
// LOG PANEL
// ============================================================
function toggleLogPanel() {
    logPanelOpen = !logPanelOpen;
    const panel  = document.getElementById('logPanel');
    const button = document.getElementById('logToggle');
    if (logPanelOpen) { panel.classList.add('open'); button.textContent = '✖ Logs'; }
    else              { panel.classList.remove('open'); button.textContent = '📋 Logs'; }
}

function connectLogWebSocket() {
    const hostname = window.location.hostname || 'localhost';
    try { logWs = new WebSocket(`ws://${hostname}:8766`); } catch (e) { updateLogStatus(false); return; }
    logWs.onopen    = () => updateLogStatus(true);
    logWs.onmessage = (event) => { const d = JSON.parse(event.data); if (d.type==='log') addLogEntry(d); };
    logWs.onerror   = () => updateLogStatus(false);
    logWs.onclose   = () => { updateLogStatus(false); if (logPanelOpen) setTimeout(connectLogWebSocket, 3000); };
}

function updateLogStatus(connected) {
    const indicator = document.getElementById('logStatusIndicator');
    const text      = document.getElementById('logStatusText');
    if (connected) { indicator.classList.add('connected'); text.textContent='Live'; text.style.color='#4caf50'; }
    else           { indicator.classList.remove('connected'); text.textContent='Disconnected'; text.style.color='#f44336'; }
}

function addLogEntry(entry) {
    const container = document.getElementById('logContainer');
    const logEntry  = document.createElement('div');
    logEntry.className      = `log-entry ${entry.level}`;
    logEntry.dataset.level  = entry.level;
    if (!logFilters.has(entry.level)) logEntry.style.display = 'none';

    const timestamp = new Date(entry.timestamp).toLocaleTimeString('en-US',{ hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit' });
    let displayMessage = entry.message;
    if ((entry.level==='USER'||entry.level==='ASSISTANT') && displayMessage.length > 300)
        displayMessage = displayMessage.slice(0, 300) + '…';

    logEntry.innerHTML = `
        <span class="log-timestamp">${timestamp}</span>
        <span class="log-level ${entry.level}">${entry.level}</span>
        <span class="log-message">${escapeHtml(displayMessage)}</span>`;
    container.appendChild(logEntry);
    while (container.children.length > 500) container.removeChild(container.firstChild);
    if (logAutoscroll) container.scrollTop = container.scrollHeight;
}

function addLocalLogEntry(level, message) { addLogEntry({ level, message, timestamp: Date.now() }); }
function clearLogs()      { document.getElementById('logContainer').innerHTML = ''; }
function toggleAutoscroll() {
    logAutoscroll = !logAutoscroll;
    document.getElementById('autoscrollText').textContent = logAutoscroll ? 'Auto-scroll: On' : 'Auto-scroll: Off';
}
function toggleLogFilter(level) {
    const btn = document.querySelector(`.filter-btn[data-level="${level}"]`);
    if (logFilters.has(level)) { logFilters.delete(level); btn.classList.remove('active'); }
    else                       { logFilters.add(level);    btn.classList.add('active'); }
    document.querySelectorAll('.log-entry').forEach(e => {
        e.style.display = logFilters.has(e.dataset.level) ? 'flex' : 'none';
    });
}
function escapeHtml(text) { const d = document.createElement('div'); d.textContent = text; return d.innerHTML; }

// ============================================================
// SYSTEM MONITOR
// ============================================================
function toggleSystemMonitor() {
    systemMonitorOpen = !systemMonitorOpen;
    const panel  = document.getElementById('systemMonitor');
    const button = document.getElementById('monitorToggle');
    if (systemMonitorOpen) { panel.classList.add('open'); button.textContent = '✖ System'; }
    else                   { panel.classList.remove('open'); button.textContent = '📊 System'; }
}

function updateProgressBar(elementId, percent) {
    const bar = document.getElementById(elementId);
    bar.style.width = percent + '%';
    bar.classList.remove('high','critical');
    if (percent > 80) bar.classList.add('critical');
    else if (percent > 60) bar.classList.add('high');
}

function updateSystemStats(stats) {
    if (!systemStatsEnabled) return;
    if (stats.cpu) {
        document.getElementById('cpuPercent').textContent = stats.cpu.usage_percent;
        document.getElementById('cpuFreq').textContent    = stats.cpu.frequency_ghz.toFixed(2);
        updateProgressBar('cpuBar', stats.cpu.usage_percent);
    }
    if (stats.gpu) {
        document.getElementById('gpuStats').style.display = 'block';
        document.getElementById('gpuPercent').textContent  = stats.gpu.usage_percent;
        document.getElementById('gpuTemp').textContent     = stats.gpu.temperature_c;
        document.getElementById('gpuMemory').textContent   = Math.round(stats.gpu.memory_used_mb)+'/'+Math.round(stats.gpu.memory_total_mb);
        updateProgressBar('gpuBar', stats.gpu.usage_percent);
    }
    if (stats.memory) {
        document.getElementById('memPercent').textContent = stats.memory.percent;
        document.getElementById('memUsed').textContent    = stats.memory.used_gb.toFixed(1);
        document.getElementById('memTotal').textContent   = stats.memory.total_gb.toFixed(1);
        updateProgressBar('memBar', stats.memory.percent);
    }
}

// ============================================================
// THINKING INDICATOR
// ============================================================
function showThinking() {
    if (thinkingIndicator && document.contains(thinkingIndicator)) return;
    thinkingIndicator = null;
    thinkingIndicator = document.createElement("div");
    thinkingIndicator.className = "thinking";
    thinkingIndicator.innerHTML = '<div class="thinking-dots"><span>&bull;</span><span>&bull;</span><span>&bull;</span></div>';
    sendBtn.style.opacity = '0.5'; sendBtn.style.cursor = 'not-allowed';
    chat.appendChild(thinkingIndicator); chat.scrollTop = chat.scrollHeight;
}
function hideThinking() {
    if (thinkingIndicator) {
        sendBtn.style.opacity = '1'; sendBtn.style.cursor = 'auto';
        thinkingIndicator.remove(); thinkingIndicator = null;
    }
}

// ============================================================
// WEBSOCKET
// ============================================================
const hostname = window.location.hostname || 'localhost';
const ws = new WebSocket(`ws://${hostname}:8765`);
ws.onerror = () => { status.textContent = "WebSocket error"; hideThinking(); };
ws.onclose = () => { status.textContent = "Disconnected"; sendBtn.disabled = true; hideThinking(); };

// Only restore session on the FIRST connection, not on every reconnect.
// Reconnects (network blips, tab focus) were resetting backend conversation
// state mid-session, causing instructions and context to be lost.
let _wsHasConnected = false;

ws.onopen = () => {
    sendBtn.disabled = false;
    updateStatusWithMode();
    ws.send(JSON.stringify({ type:"list_models" }));
    ws.send(JSON.stringify({ type:"subscribe_system_stats" }));
    connectLogWebSocket();
    if (!_wsHasConnected) {
        _wsHasConnected = true;
        const savedSessionId = localStorage.getItem(CURRENT_SESSION_KEY);
        if (savedSessionId && savedSessionId !== '') {
            const sessionId = parseInt(savedSessionId);
            if (!isNaN(sessionId)) {
                ws.send(JSON.stringify({ type:"list_sessions" }));
                setTimeout(() => selectSession(sessionId), 100);
            }
        }
    }
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type==='system_stats')                              { updateSystemStats(data); return; }
    if (data.type==='subscribed'&&data.subscription==='system_stats') return;
    if (data.type==='sessions_list')                             { allSessions = data.sessions; renderSessions(allSessions); return; }

    if (data.type==='session_loaded') {
        chat.innerHTML = "";
        currentSessionId = data.session_id;
        localStorage.setItem(CURRENT_SESSION_KEY, currentSessionId);
        data.messages.forEach(msg => addMessage(msg.text, msg.role, false, false, msg.model, msg.timestamp, msg.image||null, msg.image_url||null));
        renderSessions(allSessions);
        // If the last message is from the user, a response is still in-flight.
        // Show the thinking indicator so the user knows to wait.
        const msgs = data.messages;
        if (msgs.length > 0 && msgs[msgs.length - 1].role === 'user') {
            showThinking(); isProcessing = true;
            sendBtn.disabled = true; status.textContent = 'Processing…';
        }
        return;
    }
    if (data.type==='session_created') {
        currentSessionId = data.session_id;
        localStorage.setItem(CURRENT_SESSION_KEY, currentSessionId);
        // Add the new session to the local list so it appears immediately
        const newSession = { id: data.session_id, name: data.name || 'Untitled Session', created_at: new Date().toISOString() };
        allSessions.unshift(newSession);
        if (sessionsSidebarOpen) renderSessions(allSessions);
        return;
    }
    if (data.type==='session_name_updated') { const s=allSessions.find(s=>s.id===data.session_id); if(s){s.name=data.name;if(sessionsSidebarOpen)renderSessions(allSessions);} return; }
    if (data.type==='session_renamed')      { const s=allSessions.find(s=>s.id===data.session_id); if(s){s.name=data.name;renderSessions(allSessions);} return; }
    if (data.type==='session_deleted')      { allSessions=allSessions.filter(s=>s.id!==data.session_id); renderSessions(allSessions); return; }

    if (data.type==="user_message") {
        if (data.text===lastSentMessage) {
            lastSentMessage = null;
            if (!data.text.startsWith(":")) { showThinking(); isProcessing=true; sendBtn.disabled=true; status.textContent="Processing…"; }
            return;
        }
        addMessage(data.text,"user",false); showThinking(); isProcessing=true; sendBtn.disabled=true; status.textContent="Processing…"; return;
    }

    if (data.type==="assistant_message") {
        if (data.text.includes("🛑")||data.text.toLowerCase().includes("interrupted")||data.text.toLowerCase().includes("stopped")) {
            hideThinking(); addMessage(data.text,"assistant",false,false,data.model); isProcessing=false; resetControlButtons(); return;
        }
        hideThinking(); lastResponseWasMultiAgent = data.multi_agent===true;
        addMessage(data.text,"assistant",false,lastResponseWasMultiAgent,data.model,new Date().toISOString(),data.image||null,data.image_url||null);
        const modelLabel = data.model ? `[${data.model}] ` : '';
        addLocalLogEntry('ASSISTANT', modelLabel+data.text);
        isProcessing=false; sendBtn.style.display='flex'; sendBtn.disabled=false; sendBtn.style.opacity='1'; sendBtn.style.cursor="pointer";
        updateStatusWithMode(); return;
    }

    if (data.type==="complete") {
        hideThinking(); isProcessing=false; sendBtn.disabled=false; sendBtn.style.opacity='1'; sendBtn.style.cursor="pointer";
        if (data.stopped) status.textContent="Stopped"; else updateStatusWithMode(); return;
    }

    if (data.type==="model_switched") { updateStatusWithMode(); sendBtn.disabled=false; isProcessing=false; hideThinking(); return; }

    if (data.type==="models_list") {
        const select = document.getElementById("modelSelect");
        select.innerHTML = "";
        const ollamaModels = data.all_models.filter(m=>m.backend!=="gguf").sort((a,b)=>a.name.localeCompare(b.name));
        const ggufModels   = data.all_models.filter(m=>m.backend==="gguf").sort((a,b)=>a.name.localeCompare(b.name));
        const allLabels    = [...ollamaModels.map(m=>m.name), ...ggufModels.map(m=>`${m.name} [GGUF ${(m.size_mb/1024).toFixed(1)} GB]`)];
        const longest      = allLabels.reduce((a,b)=>b.length>a?b.length:a, 0);
        const separatorText= "─".repeat(Math.max(10, Math.floor(longest*0.7)));
        ollamaModels.forEach(m => { const o=document.createElement("option"); o.value=m.name; o.textContent=m.name; select.appendChild(o); });
        if (ollamaModels.length>0&&ggufModels.length>0) { const sep=document.createElement("option"); sep.disabled=true; sep.value=""; sep.textContent=separatorText; sep.style.color="#888"; sep.style.fontStyle="italic"; select.appendChild(sep); }
        ggufModels.forEach(m => { const o=document.createElement("option"); o.value=m.name; o.textContent=`${m.name} [GGUF ${(m.size_mb/1024).toFixed(1)} GB]`; select.appendChild(o); });
        if (data.last_used&&data.all_models.some(m=>m.name===data.last_used)) { select.value=data.last_used; if(!isProcessing) updateStatusWithMode(); }
    }
};

modelSelect.addEventListener("change", (e) => {
    status.textContent = "Switching model…"; sendBtn.disabled=true; isProcessing=true; showThinking();
    ws.send(JSON.stringify({ type:"switch_model", model:e.target.value }));
});

// ============================================================
// MESSAGE FORMATTING
// ============================================================
function formatMessage(text) {
    const blocks=[],inline=[],links=[],mathInline=[],mathBlock=[];
    text=text.replace(/\$\$([\s\S]*?)\$\$/g,(m,math)=>{const i=mathBlock.length;mathBlock.push(math);return`@@MATHBLOCK_${i}@@`;});
    text=text.replace(/\\\[([\s\S]*?)\\\]/g,(m,math)=>{const i=mathBlock.length;mathBlock.push(math);return`@@MATHBLOCK_${i}@@`;});
    text=text.replace(/\\\(([\s\S]*?)\\\)/g,(m,math)=>{const i=mathInline.length;mathInline.push(math);return`@@MATHINLINE_${i}@@`;});
    text=text.replace(/([ \t]*)```(\w+)?\s*\n?([\s\S]*?)```/g,(m,indent,lang,code)=>{
        const indentLen=indent.length,lines=code.split('\n');
        const dedented=lines.map(l=>l.trim().length===0?'':l.startsWith(indent)?l.slice(indentLen):l.trimStart()).join('\n').trim();
        const i=blocks.length;blocks.push({lang:lang||"code",code:dedented});return`@@CODEBLOCK_${i}@@`;
    });
    text=text.replace(/'''([\s\S]*?)'''/g,(m,code)=>{const i=blocks.length;blocks.push({lang:"code",code});return`@@CODEBLOCK_${i}@@`;});
    text=text.replace(/`([^`]+)`/g,(m,code)=>{const i=inline.length;inline.push(code);return`@@INLINE_${i}@@`;});
    const images=[];
    text=text.replace(/!\[([^\]]*)\]\((https?:\/\/[^\)]+)\)/g,(m,alt,url)=>{const i=images.length;images.push({alt,url});return`@@IMAGE_${i}@@`;});
    text=text.replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g,(m,label,url)=>{const i=links.length;links.push({label,url});return`@@LINK_${i}@@`;});
    text=text.replace(/(?<!\]\()(?<!")https?:\/\/[^\s<>"]+/g,(url)=>{const trailing=url.match(/[.,;:!?'")\]]+$/)?.[0]||'';const cleanUrl=url.slice(0,url.length-trailing.length);const i=links.length;links.push({label:cleanUrl,url:cleanUrl});return`@@LINK_${i}@@${trailing}`;});
    text=text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    text=text.replace(/^######\s+(.+)$/gm,"<h6>$1</h6>").replace(/^#####\s+(.+)$/gm,"<h5>$1</h5>").replace(/^####\s+(.+)$/gm,"<h4>$1</h4>").replace(/^###\s+(.+)$/gm,"<h3>$1</h3>").replace(/^##\s+(.+)$/gm,"<h2>$1</h2>").replace(/^#\s+(.+)$/gm,"<h1>$1</h1>");
    text=text.replace(/\*\*\*(.+?)\*\*\*/g,"<strong><em>$1</em></strong>").replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>").replace(/\*(.+?)\*/g,"<em>$1</em>").replace(/~~(.+?)~~/g,"<del>$1</del>");
    text=text.replace(/^>\s+(.+)$/gm,"<blockquote>$1</blockquote>").replace(/^([-*_]){3,}$/gm,"<hr>").replace(/^- /gm,"&bull; ").replace(/^\* /gm,"&bull; ");
    text=text.replace(/^(\|.+\|)\s*\n(\|[-:\s|]+\|)\s*\n((?:\|.*\|\s*\n?)*)/gm,(match,header,divider,rows)=>{
        const makeRow=row=>"<tr>"+row.trim().slice(1,-1).split("|").map(cell=>`<td>${cell.trim()}</td>`).join("")+"</tr>";
        return`<table><thead>${makeRow(header).replace(/<td>/g,"<th>").replace(/<\/td>/g,"</th>")}</thead><tbody>${rows.trim().split("\n").filter(r=>r.trim().startsWith("|")).map(makeRow).join("")}</tbody></table>`;
    });
    text=text.replace(/\n\s*\n/g,"<br><br>").replace(/\n/g,"<br>").replace(/(<\/h[1-6]>)(<br>){2}/g,"$1<br>");
    const processMath=(math)=>{
        math=math.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
        math=math.replace(/\\text\{([^}]+)\}/g,'$1');
        const symbols={'\\pm':'±','\\times':'×','\\div':'÷','\\cdot':'·','\\neq':'≠','\\leq':'≤','\\geq':'≥','\\approx':'≈','\\infty':'∞','\\pi':'π','\\rightarrow':'→','\\forall':'∀','\\exists':'∃','\\degree':'°','\\phi':'Φ'};
        Object.entries(symbols).forEach(([t,s])=>{math=math.split(t).join(s);});
        math=math.replace(/\\sqrt\[([^\]]+)\]\{([^}]+)\}/g,'<sup>$1</sup>√($2)').replace(/\\sqrt\{([^}]+)\}/g,'√($1)').replace(/\^\{([^}]+)\}/g,'<sup>$1</sup>').replace(/\^([a-zA-Z0-9+-]+)/g,'<sup>$1</sup>').replace(/_\{([^}]+)\}/g,'<sub>$1</sub>').replace(/_([a-zA-Z0-9+-]+)/g,'<sub>$1</sub>').replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g,'($1)/($2)');
        return`<strong>${math}</strong>`;
    };
    text=text.replace(/@@INLINE_(\d+)@@/g,(m,i)=>`<code class="code-inline">${inline[i].replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}</code>`);
    text=text.replace(/@@CODEBLOCK_(\d+)@@/g,(m,i)=>{
        const block=blocks[i],escaped=block.code.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"),id=`cb_${Date.now()}_${i}`;
        return`<div style="position:relative;margin:8px 0;"><button class="copy-btn" data-target="${id}" style="position:absolute;top:6px;right:6px;background:rgba(255,255,255,0.1);border:none;border-radius:4px;padding:4px 6px;cursor:pointer;color:#aaa;display:flex;align-items:center;z-index:1;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button><pre><code id="${id}" class="language-${block.lang}">${escaped.trim()}</code></pre></div>`;
    });
    text=text.replace(/@@MATHINLINE_(\d+)@@/g,(m,i)=>processMath(mathInline[i]));
    text=text.replace(/@@MATHBLOCK_(\d+)@@/g,(m,i)=>processMath(mathBlock[i]));
    text=text.replace(/@@LINK_(\d+)@@/g,(m,i)=>`<a href="${links[i].url}" target="_blank">${links[i].label}</a>`);
    text=text.replace(/@@IMAGE_(\d+)@@/g,(m,i)=>`<img src="${images[i].url}" alt="${images[i].alt}" style="max-width:100%;max-height:200px;border-radius:6px;display:block;margin:4px 0;object-fit:contain;">`);
    return text;
}

document.addEventListener('click', function (e) {
    const btn = e.target.closest('.copy-btn');
    if (!btn) return;

    const code = document.getElementById(btn.dataset.target);
    if (!code) return;

    const text = code.innerText;

    function showCopied() {
        btn.style.transition = 'opacity 0.2s ease';
        btn.style.opacity = '0';

        setTimeout(() => {
            btn.innerHTML = '<span style="color:#22c55e;font-size:12px;font-weight:600;">✓ Copied</span>';
            btn.style.opacity = '1';
        }, 200);

        setTimeout(() => {
            btn.style.opacity = '0';
            setTimeout(() => {
                btn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                    </svg>`;
                btn.style.opacity = '1';
            }, 200);
        }, 2200);
    }

    // --- Primary method: modern clipboard API ---
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text)
            .then(showCopied)
            .catch(() => fallbackCopy(text, showCopied));
    } else {
        // --- Fallback immediately ---
        fallbackCopy(text, showCopied);
    }
});

function fallbackCopy(text, onSuccess) {
    const textarea = document.createElement('textarea');
    textarea.value = text;

    // Prevent scrolling to bottom on iOS
    textarea.style.position = 'fixed';
    textarea.style.top = '-9999px';

    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    try {
        const ok = document.execCommand('copy');
        if (ok) onSuccess();
    } catch (err) {
        console.warn('Copy fallback failed:', err);
    }

    document.body.removeChild(textarea);
}

// ============================================================
// ADD MESSAGE
// ============================================================
function addMessage(text, role, saveToDb=false, isMultiAgent=false, modelName=null, timestamp=null, imageB64=null, imageUrl=null) {
    text = text || "";
    if (text.startsWith("[TextContent(")) return;
    if (text.trim()===""&&!imageB64&&!imageUrl) return;
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    if (role==="assistant"&&isMultiAgent) div.className += " multi-agent";
    if (role==="user") {
        div.textContent = text;
    } else {
        const imgSrc = imageUrl || (imageB64 ? `data:image/jpeg;base64,${imageB64}` : null);
        if (imgSrc) {
            const img = document.createElement("img");
            img.src = imgSrc;
            img.style.cssText = "max-width:100%;max-height:320px;border-radius:6px;display:block;margin-bottom:8px;object-fit:contain;";
            img.alt = "Image result";
            div.appendChild(img);
        }
        if (text.trim()) {
            const textNode = document.createElement("span");
            textNode.innerHTML = formatMessage(text);
            div.appendChild(textNode);
        }
    }
    if (role === 'assistant') {
        const wrapper = document.createElement('div');
        wrapper.className = 'msg-wrapper';
        wrapper.appendChild(div);

        const meta = document.createElement('div');
        meta.className = 'msg-meta';

        const ts = document.createElement('div');
        ts.className = 'msg-timestamp';
        const d = timestamp ? new Date(timestamp.includes('T') || timestamp.endsWith('Z') ? timestamp : timestamp.replace(' ', 'T') + 'Z') : new Date();
        const { locale, timeZone } = Intl.DateTimeFormat().resolvedOptions();
        ts.textContent = d.toLocaleString(locale, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true, timeZone });
        meta.appendChild(ts);

        if (modelName) {
            const model = document.createElement('div');
            model.className = 'msg-model';
            const FALLBACK = new Set(["unknown","direct-answer","mcp error"]);
            model.textContent = FALLBACK.has(modelName.toLowerCase()) ? "MCP" : modelName;
            meta.appendChild(model);
        }

        wrapper.appendChild(meta);
        chat.appendChild(wrapper);
    } else {
        chat.appendChild(div);
    }
    chat.scrollTop = chat.scrollHeight;
    if (saveToDb) saveMessageToSession(role, text);
}

// ============================================================
// SEND MESSAGE
// ============================================================
function send() {
    sendBtn.style.opacity='0.5'; sendBtn.disabled=true; sendBtn.style.cursor="not-allowed";
    const text = input.value.trim();
    if (isProcessing&&text!==":stop") { sendBtn.style.opacity='1'; sendBtn.disabled=false; sendBtn.style.cursor="pointer"; return; }
    if (!text) { sendBtn.style.opacity='1'; sendBtn.disabled=false; sendBtn.style.cursor="pointer"; return; }
    if (ws.readyState!==WebSocket.OPEN) { status.textContent="Cannot send: WebSocket not connected"; sendBtn.style.opacity='1'; sendBtn.disabled=false; sendBtn.style.cursor="pointer"; return; }

    addMessage(text,"user",false);
    addLocalLogEntry('USER', text);
    input.value=""; lastSentMessage=text;

    if (text===":stop") {
        ws.send(JSON.stringify({type:"user",text:":stop"})); isProcessing=false; status.textContent="Stop signal sent...";
        sendBtn.style.opacity='1'; sendBtn.disabled=false; sendBtn.style.cursor="pointer"; return;
    }

    const quickCommands=[':commands',':tools',':model',':models',':stats',':multi',':a2a',':health',':metrics'];
    const isQuickCommand=quickCommands.some(cmd=>text.startsWith(cmd));
    isProcessing=true; sendBtn.disabled=true; status.textContent="Processing…"; lastResponseWasMultiAgent=false;
    if (!isQuickCommand) showThinking();
    ws.send(JSON.stringify({type:"user",text,session_id:currentSessionId}));
}

sendBtn.addEventListener('click', send);
input.addEventListener("keydown", (e) => { if (e.key==="Enter"&&!e.shiftKey) { e.preventDefault(); send(); } });
// ============================================================
// CHAT CONTAINER RESIZE OBSERVER
// Adds .narrow class when chatContainer width <= 768px so
// CSS can scale message widths regardless of which panels are open
// ============================================================
(function() {
    const chatContainer = document.getElementById('chatContainer');
    if (!chatContainer || !window.ResizeObserver) return;
    const observer = new ResizeObserver(entries => {
        for (const entry of entries) {
            const width = entry.contentRect.width;
            chatContainer.classList.toggle('narrow', width <= 768);
        }
    });
    observer.observe(chatContainer);
})();
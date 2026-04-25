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
    const overrides = { optimus: 'Optimus Prime', tokyo: 'Tokyo Night', te: 'te', mono: 'Mono' };
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

    // Update border-radius on already-rendered images and copy buttons
    const isMatrix = themeName === 'matrix';
    document.querySelectorAll('#chat img').forEach(img => {
        img.style.borderRadius = isMatrix ? '0' : '6px';
    });
    document.querySelectorAll('#chat .copy-btn').forEach(btn => {
        btn.style.borderRadius = isMatrix ? '0' : '4px';
    });

    if (isMatrix) {
        const root = document.documentElement;
        root.classList.remove('matrix-booting');
        void root.offsetWidth;
        root.classList.add('matrix-booting');
        setTimeout(() => root.classList.remove('matrix-booting'), 1700);
    }
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
let selectionMode          = false;
let selectedSessionIds     = new Set();

let logPanelOpen    = false;
let logAutoscroll   = true;
let logFilters      = new Set(['USER','ASSISTANT','DEBUG','INFO','WARNING','ERROR']);
let logWs           = null;
let systemMonitorOpen   = false;
let systemStatsEnabled  = true;

// ============================================================
// PROMPT NAVIGATOR STATE
// ============================================================
let messageIndex        = [];   // { text, domRef } — user messages only
let promptNavCollapsed  = false;

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
        openMenuSessionId  = null;
        editingSessionId   = null;
        selectionMode      = false;
        selectedSessionIds = new Set();
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
    messageIndex = [];
    renderNavigator();
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

    // Sync header button label
    const selectBtn = document.getElementById('selectModeBtn');
    if (selectBtn) selectBtn.textContent = selectionMode ? 'Done' : 'Select';

    // Sync toolbar
    const toolbar = document.getElementById('selectionToolbar');
    if (toolbar) toolbar.style.display = selectionMode ? 'flex' : 'none';

    if (selectionMode) {
        const allCb  = document.getElementById('selectAllCb');
        const delBtn = document.getElementById('bulkDeleteBtn');
        const n = selectedSessionIds.size;
        if (allCb) {
            allCb.checked       = n > 0 && n === sessions.length;
            allCb.indeterminate = n > 0 && n < sessions.length;
        }
        if (delBtn) {
            delBtn.disabled     = n === 0;
            delBtn.textContent  = n > 0 ? `Delete (${n})` : 'Delete';
        }
    }

    sessions.forEach(session => {
        const item = document.createElement('div');
        item.className = 'session-item';
        item.title = session.name || 'Untitled Session';
        if (session.id === currentSessionId) item.classList.add('active');
        if (session.pinned) item.classList.add('pinned');

        if (selectionMode) {
            const checked = selectedSessionIds.has(session.id);
            if (checked) item.classList.add('session-checked');

            const cb = document.createElement('input');
            cb.type = 'checkbox'; cb.className = 'session-cb';
            cb.checked = checked;
            cb.onclick = e => e.stopPropagation();
            cb.onchange = () => {
                if (cb.checked) selectedSessionIds.add(session.id);
                else            selectedSessionIds.delete(session.id);
                item.classList.toggle('session-checked', cb.checked);
                renderSessions(allSessions);
            };

            const textWrap = document.createElement('div');
            textWrap.style.cssText = 'flex:1;min-width:0;display:flex;flex-direction:column;';
            const text = document.createElement('span'); text.className = 'session-item-text';
            text.textContent = session.name || 'Untitled Session';
            const dateLabel = document.createElement('div'); dateLabel.className = 'session-item-date';
            dateLabel.textContent = formatSessionDate(session.created_at);
            textWrap.appendChild(text); textWrap.appendChild(dateLabel);

            item.appendChild(cb); item.appendChild(textWrap);
            item.onclick = () => { cb.checked = !cb.checked; cb.dispatchEvent(new Event('change')); };

        } else if (editingSessionId === session.id) {
            item.classList.add('editing');
            const inp = document.createElement('input');
            inp.type = 'text'; inp.className = 'session-edit-input';
            inp.value = session.name || 'Untitled Session';
            inp.onclick = (e) => e.stopPropagation();
            inp.addEventListener('mouseenter', () => {
                let tip = document.getElementById('session-edit-tip');
                if (!tip) { tip = document.createElement('div'); tip.id = 'session-edit-tip'; tip.className = 'session-edit-tooltip'; document.body.appendChild(tip); }
                tip.textContent = inp.value;
                const r = inp.getBoundingClientRect();
                tip.style.left = r.left + 'px';
                tip.style.top  = (r.bottom + 4) + 'px';
                tip.style.display = 'block';
            });
            inp.addEventListener('mouseleave', () => { const tip = document.getElementById('session-edit-tip'); if (tip) tip.style.display = 'none'; });

            const actions  = document.createElement('div'); actions.className = 'session-edit-actions';
            const saveBtn2 = document.createElement('button'); saveBtn2.textContent = '✓'; saveBtn2.className = 'session-edit-btn';
            saveBtn2.onclick = (e) => { e.stopPropagation(); saveSessionRename(session.id, inp.value); };
            const cancelBtn = document.createElement('button'); cancelBtn.textContent = '✕'; cancelBtn.className = 'session-edit-cancel';
            cancelBtn.onclick = (e) => { e.stopPropagation(); cancelSessionEdit(); };

            actions.appendChild(saveBtn2); actions.appendChild(cancelBtn);
            item.appendChild(inp); item.appendChild(actions);
            setTimeout(() => { inp.focus(); inp.setSelectionRange(0, 0); }, 10);
        } else {
            const textContainer = document.createElement('div');
            textContainer.style.cssText = 'flex:1;min-width:0;display:flex;flex-direction:column;';
            const text = document.createElement('span'); text.className = 'session-item-text';
            if (session.pinned) {
                const pin = document.createElement('span');
                pin.className = 'session-pin-indicator';
                pin.textContent = '📌 ';
                text.appendChild(pin);
            }
            text.appendChild(document.createTextNode(session.name || 'Untitled Session'));
            const dateLabel = document.createElement('div'); dateLabel.className = 'session-item-date';
            dateLabel.textContent = formatSessionDate(session.created_at);
            textContainer.appendChild(text); textContainer.appendChild(dateLabel);

            const menuBtn2 = document.createElement('button'); menuBtn2.className = 'session-menu-btn';
            menuBtn2.textContent = '⋮';
            menuBtn2.onclick = (e) => { e.stopPropagation(); toggleSessionMenu(session.id); };
            if (openMenuSessionId === session.id) menuBtn2.classList.add('active');

            const submenu    = document.createElement('div'); submenu.className = 'session-submenu';
            if (openMenuSessionId === session.id) submenu.classList.add('open');

            const pinItem    = document.createElement('div'); pinItem.className = 'session-submenu-item pin';
            pinItem.innerHTML = session.pinned ? '📌 Unpin' : '📌 Pin';
            pinItem.onclick = (e) => { e.stopPropagation(); pinSession(session.id, !session.pinned); };

            const editItem   = document.createElement('div'); editItem.className = 'session-submenu-item';
            editItem.innerHTML = '✏️ Edit';
            editItem.onclick = (e) => { e.stopPropagation(); startSessionEdit(session.id); };

            const deleteItem = document.createElement('div'); deleteItem.className = 'session-submenu-item delete';
            deleteItem.innerHTML = '🗑️ Delete';
            deleteItem.onclick = (e) => { e.stopPropagation(); deleteSession(session.id); };

            submenu.appendChild(pinItem); submenu.appendChild(editItem); submenu.appendChild(deleteItem);
            item.appendChild(textContainer); item.appendChild(menuBtn2); item.appendChild(submenu);
            item.onclick = () => selectSession(session.id);
        }
        list.appendChild(item);
    });
}

function toggleSelectionMode() {
    selectionMode      = !selectionMode;
    selectedSessionIds = new Set();
    openMenuSessionId  = null;
    editingSessionId   = null;
    renderSessions(allSessions);
}

function toggleSelectAll() {
    const allCb = document.getElementById('selectAllCb');
    if (!allCb) return;
    if (allCb.checked) allSessions.forEach(s => selectedSessionIds.add(s.id));
    else               selectedSessionIds = new Set();
    renderSessions(allSessions);
}

function bulkDeleteSessions() {
    const count = selectedSessionIds.size;
    if (count === 0) return;
    showConfirm({
        title:       `Delete ${count} Conversation${count !== 1 ? 's' : ''}`,
        message:     `Are you sure you want to delete ${count} conversation${count !== 1 ? 's' : ''}? This cannot be undone.`,
        confirmText: 'Delete',
        cancelText:  'Cancel',
        danger:      true
    }).then(confirmed => {
        if (!confirmed) return;
        let deletedCurrent = false;
        selectedSessionIds.forEach(id => {
            ws.send(JSON.stringify({ type: 'delete_session', session_id: id }));
            if (id === currentSessionId) deletedCurrent = true;
        });
        selectedSessionIds = new Set();
        selectionMode      = false;
        if (deletedCurrent) {
            currentSessionId = null;
            localStorage.setItem(CURRENT_SESSION_KEY, '');
            chat.innerHTML = '';
            messageIndex = [];
            renderNavigator();
            ws.send(JSON.stringify({ type: 'new_session' }));
        }
    });
}

// ── Message search state ──
let _searchDebounceTimer = null;
let _searchDropdownVisible = false;

function filterSessions() {
    const searchTerm = document.getElementById('sessionSearch').value;

    // Clear any pending debounce
    if (_searchDebounceTimer) clearTimeout(_searchDebounceTimer);

    // Hide dropdown and restore session list if empty
    if (!searchTerm.trim()) {
        hideSearchDropdown();
        renderSessions(allSessions);
        return;
    }

    // Local name filter immediately
    const regex = new RegExp(searchTerm, 'i');
    renderSessions(allSessions.filter(s => regex.test(s.name)));

    // Debounce DB content search — only fire after 350ms pause
    _searchDebounceTimer = setTimeout(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'search_messages', term: searchTerm.trim() }));
        }
    }, 350);
}

function handleSearchResults(data) {
    const term = data.term || '';
    const results = data.results || [];
    const currentTerm = (document.getElementById('sessionSearch').value || '').trim();

    // Discard stale responses
    if (term !== currentTerm) return;

    if (!results.length) {
        hideSearchDropdown();
        return;
    }

    showSearchDropdown(results, term);
}

function showSearchDropdown(results, term) {
    let dropdown = document.getElementById('searchResultsDropdown');
    if (!dropdown) {
        dropdown = document.createElement('div');
        dropdown.id = 'searchResultsDropdown';
        dropdown.className = 'search-results-dropdown';
        // Insert after the search container
        const container = document.querySelector('.sessions-search-container');
        container.parentNode.insertBefore(dropdown, container.nextSibling);
    }

    // Deduplicate by message_id — same session can appear multiple times
    const seen = new Set();
    const deduped = results.filter(r => {
        if (seen.has(r.message_id)) return false;
        seen.add(r.message_id);
        return true;
    });

    dropdown.innerHTML = deduped.map(r => {
        const snippet = makeSnippet(r.content, term, 80);
        const full    = escapeHtml(r.content);
        const role    = r.role === 'user' ? 'You' : 'Assistant';
        return `
            <div class="search-result-item"
                 data-session-id="${r.session_id}"
                 data-message-id="${r.message_id}"
                 data-full="${escapeAttr(r.content)}"
                 onclick="goToSearchResult(${r.session_id}, ${r.message_id})"
                 title="${escapeAttr(r.content)}">
                <div class="search-result-session">${escapeHtml(r.session_name)}</div>
                <div class="search-result-snippet"><span class="search-result-role">${role}:</span> ${snippet}</div>
            </div>`;
    }).join('');

    // Hide session list while showing search results — dropdown overlays absolutely
    const sessionsList = document.getElementById('sessionsList');
    if (sessionsList) sessionsList.style.display = 'none';

    dropdown.style.display = 'block';
    _searchDropdownVisible = true;
}

function hideSearchDropdown() {
    const dropdown = document.getElementById('searchResultsDropdown');
    if (dropdown) dropdown.style.display = 'none';
    // Restore session list
    const sessionsList = document.getElementById('sessionsList');
    if (sessionsList) sessionsList.style.display = '';
    _searchDropdownVisible = false;
}

function makeSnippet(content, term, maxLen) {
    const idx = content.toLowerCase().indexOf(term.toLowerCase());
    let raw;
    if (idx === -1) {
        raw = content.slice(0, maxLen);
    } else {
        const start = Math.max(0, idx - 20);
        const end   = Math.min(content.length, idx + term.length + 60);
        raw = (start > 0 ? '…' : '') + content.slice(start, end) + (end < content.length ? '…' : '');
    }
    // Highlight the term
    const escaped = escapeHtml(raw);
    const re = new RegExp(`(${escapeRegex(term)})`, 'gi');
    return escaped.replace(re, '<mark>$1</mark>');
}

function escapeAttr(str) {
    return String(str).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ── Pending post-load scroll (set before selectSession, consumed in session_loaded) ──
let _pendingScrollMessageId = null;

function goToSearchResult(sessionId, messageId) {
    hideSearchDropdown();
    document.getElementById('sessionSearch').value = '';
    renderSessions(allSessions);

    _pendingScrollMessageId = messageId;
    selectSession(sessionId);
}

function scrollToMessage(messageId) {
    const tryScroll = (attempts) => {
        const el = document.querySelector(`[data-chat-message-id="${messageId}"]`);
        if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            el.classList.add('search-highlight');
            setTimeout(() => el.classList.remove('search-highlight'), 2000);
        } else if (attempts > 0) {
            setTimeout(() => tryScroll(attempts - 1), 150);
        }
    };
    setTimeout(() => tryScroll(10), 200);
}

// Dismiss dropdown when clicking outside
document.addEventListener('click', (e) => {
    if (!_searchDropdownVisible) return;
    const dropdown = document.getElementById('searchResultsDropdown');
    const container = document.querySelector('.sessions-search-container');
    if (dropdown && !dropdown.contains(e.target) && container && !container.contains(e.target)) {
        hideSearchDropdown();
    }
});

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
            messageIndex = [];
            renderNavigator();
            ws.send(JSON.stringify({ type: "new_session" }));
        }
    });
}

function pinSession(sessionId, pinned) {
    ws.send(JSON.stringify({ type: 'pin_session', session_id: sessionId, pinned: pinned }));
    openMenuSessionId = null;
}

// ============================================================
// PROMPT NAVIGATOR
// ============================================================
function renderNavigator() {
    const nav     = document.getElementById('promptNavigator');
    const list    = document.getElementById('promptNavList');
    const count   = document.getElementById('promptNavCount');
    const chevron = document.getElementById('promptNavChevron');
    if (!nav || !list) return;

    if (messageIndex.length === 0) {
        nav.style.display = 'none';
        return;
    }

    nav.style.display = 'flex';
    count.textContent = messageIndex.length;
    chevron.textContent = promptNavCollapsed ? '\u25bc' : '\u25b2';
    list.style.display  = promptNavCollapsed ? 'none' : 'block';

    list.innerHTML = '';
    messageIndex.forEach((entry, i) => {
        const item = document.createElement('div');
        item.className = 'prompt-nav-item';
        item.title = entry.text;

        const label = document.createElement('span');
        label.className = 'prompt-nav-text';
        label.textContent = entry.text;

        item.appendChild(label);
        item.onclick = () => {
            entry.domRef.scrollIntoView({ behavior: 'smooth', block: 'center' });
            entry.domRef.classList.add('prompt-nav-highlight');
            setTimeout(() => entry.domRef.classList.remove('prompt-nav-highlight'), 1200);
        };
        list.appendChild(item);
    });

    // Highlight the first item by default until scroll sync takes over
    const firstItem = list.querySelector('.prompt-nav-item');
    if (firstItem) firstItem.classList.add('nav-hover-active');
}

function togglePromptNavigator() {
    promptNavCollapsed = !promptNavCollapsed;
    renderNavigator();
}

// ============================================================
// PROMPT NAVIGATOR — scroll-based highlight sync
// Watches #chat scroll position; highlights the nav item
// corresponding to whichever user/assistant turn is most
// visible in the viewport. Stays highlighted until scroll moves.
// ============================================================
(function initNavScrollSync() {
    const chat = document.getElementById('chat');
    if (!chat) return;

    let activeNavIndex = -1;
    let observer = null;

    function setNavHighlight(index) {
        if (index === activeNavIndex) return;
        const list = document.getElementById('promptNavList');
        if (!list) return;
        list.querySelectorAll('.prompt-nav-item').forEach(el =>
            el.classList.remove('nav-hover-active')
        );
        activeNavIndex = index;
        if (index < 0) return;
        const items = list.querySelectorAll('.prompt-nav-item');
        if (items[index]) {
            items[index].classList.add('nav-hover-active');
            // Scroll the nav item into view within the list if needed
            items[index].scrollIntoView({ block: 'nearest' });
        }
    }

    function getNavIndexForElement(el) {
        // Direct user message
        const userIdx = messageIndex.findIndex(entry => entry.domRef === el);
        if (userIdx !== -1) return userIdx;

        // Assistant wrapper — find preceding user message
        const wrapper = el.closest('.msg-wrapper');
        if (wrapper) {
            let sibling = wrapper.previousElementSibling;
            while (sibling) {
                const idx = messageIndex.findIndex(entry => entry.domRef === sibling);
                if (idx !== -1) return idx;
                sibling = sibling.previousElementSibling;
            }
        }
        return -1;
    }

    function buildObserver() {
        if (observer) observer.disconnect();

        // Track which elements are visible and how much
        const visibilityMap = new Map();

        observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                visibilityMap.set(entry.target, entry.intersectionRatio);
            });

            // Find the most visible element that maps to a nav index
            let bestRatio = 0;
            let bestIndex = -1;
            visibilityMap.forEach((ratio, el) => {
                if (ratio > bestRatio) {
                    const idx = getNavIndexForElement(el);
                    if (idx !== -1) {
                        bestRatio = ratio;
                        bestIndex = idx;
                    }
                }
            });

            if (bestRatio > 0) setNavHighlight(bestIndex);
        }, {
            root: chat,
            threshold: [0, 0.1, 0.25, 0.5, 0.75, 1.0]
        });

        // Observe all .msg elements inside #chat
        chat.querySelectorAll('.msg').forEach(el => observer.observe(el));
    }

    // Rebuild observer when new messages are added
    // Patch renderNavigator to also rebuild
    const _origRender = window.renderNavigator;
    window.renderNavigator = function() {
        _origRender && _origRender();
        // Small delay so DOM is settled before observing
        setTimeout(buildObserver, 50);
    };

    // Allow external code to reset the active index so index 0 re-highlights
    window.resetNavHighlight = function() {
        activeNavIndex = -1;
        setTimeout(() => setNavHighlight(0), 150);
    };

    // Also rebuild when a session loads (chat gets repopulated)
    const chatMutationObserver = new MutationObserver(() => {
        buildObserver();
    });
    chatMutationObserver.observe(chat, { childList: true });

    buildObserver();
})();

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
const FAVORITES_SETTING_KEY = 'tool_favorites'; // declared here — used in onopen before favorites section
let ws;
let _wsHasConnected = false;
let _wsReconnectTimer = null;
let _wsReconnectDelay = 1000;

function connectMainWS() {
    if (_wsReconnectTimer) { clearTimeout(_wsReconnectTimer); _wsReconnectTimer = null; }
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;

    try { ws = new WebSocket(`ws://${hostname}:8765`); }
    catch (e) { scheduleReconnect(); return; }

    ws.onerror = () => { status.textContent = "WebSocket error"; hideThinking(); };

    ws.onclose = () => {
        status.textContent = "Disconnected";
        sendBtn.disabled = true;
        hideThinking();
        scheduleReconnect();
    };

    ws.onopen = () => {
        sendBtn.disabled = false;
        _wsReconnectDelay = 1000;
        updateStatusWithMode();
        ws.send(JSON.stringify({ type:"list_models" }));
        ws.send(JSON.stringify({ type:"list_tools" }));
        ws.send(JSON.stringify({ type:"subscribe_system_stats" }));
        connectLogWebSocket();
        ws.send(JSON.stringify({ type:"get_setting", key:FAVORITES_SETTING_KEY }));
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
        } else {
            if (currentSessionId) {
                ws.send(JSON.stringify({ type:"load_session", session_id: currentSessionId }));
            } else {
                ws.send(JSON.stringify({ type:"list_sessions" }));
            }
        }
    };
}

function scheduleReconnect() {
    if (_wsReconnectTimer) return;
    _wsReconnectTimer = setTimeout(() => {
        _wsReconnectTimer = null;
        connectMainWS();
    }, _wsReconnectDelay);
    _wsReconnectDelay = Math.min(_wsReconnectDelay * 2, 30000);
}

document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
            _wsReconnectDelay = 1000;
            connectMainWS();
        }
    }
});

connectMainWS();

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type==='system_stats')                              { updateSystemStats(data); return; }
    if (data.type==='setting_value') {
        if (data.key === FAVORITES_SETTING_KEY) {
            try { _favorites = JSON.parse(data.value) || []; } catch { _favorites = []; }
            if (_allTools.length > 0) renderToolsPanel(_allTools);
        }
        return;
    }
    if (data.type==='setting_saved')                             { return; }
    if (data.type==='subscribed'&&data.subscription==='system_stats') return;
    if (data.type==='sessions_list')                             { allSessions = data.sessions; renderSessions(allSessions); return; }
    if (data.type==='search_messages_result')                    { handleSearchResults(data); return; }

    if (data.type==='session_loaded') {
        chat.innerHTML = "";
        messageIndex = [];
        renderNavigator();
        currentSessionId = data.session_id;
        localStorage.setItem(CURRENT_SESSION_KEY, currentSessionId);
        data.messages.forEach(msg => addMessage(msg.text, msg.role, false, false, msg.model, msg.timestamp, msg.image||null, msg.image_url||null, msg.id||null));
        renderSessions(allSessions);
        // If the last message is from the user, a response is still in-flight.
        // Show the thinking indicator so the user knows to wait.
        const msgs = data.messages;
        if (msgs.length > 0 && msgs[msgs.length - 1].role === 'user') {
            showThinking(); isProcessing = true;
            sendBtn.disabled = true; status.textContent = 'Processing…';
        }
        // Reset nav highlight so first item is highlighted on session switch
        if (window.resetNavHighlight) window.resetNavHighlight();
        // Consume any pending search scroll
        if (_pendingScrollMessageId !== null) {
            const mid = _pendingScrollMessageId;
            _pendingScrollMessageId = null;
            scrollToMessage(mid);
        }
        return;
    }
    if (data.type==='session_created') {
        currentSessionId = data.session_id;
        localStorage.setItem(CURRENT_SESSION_KEY, currentSessionId);
        const newSession = { id: data.session_id, name: data.name || 'Untitled Session', created_at: new Date().toISOString() };
        const firstUnpinned = allSessions.findIndex(s => !s.pinned);
        if (firstUnpinned === -1) allSessions.push(newSession);
        else allSessions.splice(firstUnpinned, 0, newSession);
        renderSessions(allSessions);
        renderNavigator();
        return;
    }
    if (data.type==='session_name_updated') { const s=allSessions.find(s=>s.id===data.session_id); if(s){s.name=data.name;if(sessionsSidebarOpen)renderSessions(allSessions);} return; }
    if (data.type==='session_renamed')      { const s=allSessions.find(s=>s.id===data.session_id); if(s){s.name=data.name;renderSessions(allSessions);} return; }
    if (data.type==='session_deleted') {
        allSessions = allSessions.filter(s => s.id !== data.session_id);
        renderNavigator();
        renderSessions(allSessions);
        return;
    }
    if (data.type==='session_pinned') {
        const s = allSessions.find(s => s.id === data.session_id);
        if (s) {
            s.pinned = data.pinned;
            // Re-sort: pinned first, then by updated_at desc (mirrors backend ORDER BY)
            allSessions.sort((a, b) => {
                if (a.pinned !== b.pinned) return (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0);
                return new Date(b.updated_at || b.created_at) - new Date(a.updated_at || a.created_at);
            });
            renderSessions(allSessions);
        }
        return;
    }

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
        if (messageIndex.length > 0)
            messageIndex[messageIndex.length - 1].domRef.scrollIntoView({ behavior: 'smooth', block: 'center' });
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

    if (data.type==="tools_list") {
        buildToolPrompts(data.tools);
        renderToolsPanel(data.tools);
        return;
    }
    if (data.type==="metrics_reset") {
        // Dashboard handles this — index.js just ignores it cleanly
        return;
    }
};

modelSelect.addEventListener("change", (e) => {
    status.textContent = "Switching model…"; sendBtn.disabled=true; isProcessing=true; showThinking();
    ws.send(JSON.stringify({ type:"switch_model", model:e.target.value }));
});

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
function addMessage(text, role, saveToDb=false, isMultiAgent=false, modelName=null, timestamp=null, imageB64=null, imageUrl=null, messageId=null) {
    text = text || "";
    if (text.startsWith("[TextContent(")) return;
    if (text.trim()===""&&!imageB64&&!imageUrl) return;
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    if (messageId && role !== 'assistant') div.setAttribute('data-chat-message-id', messageId);
    if (role==="assistant"&&isMultiAgent) div.className += " multi-agent";
    if (role==="user") {
        div.textContent = text;
    } else {
        const imgSrc = imageUrl || (imageB64 ? `data:image/jpeg;base64,${imageB64}` : null);
        if (imgSrc) {
            const img = document.createElement("img");
            img.src = imgSrc;
            img.style.cssText = `max-width:100%;height:auto;border-radius:${document.documentElement.getAttribute('data-theme')==='matrix'?'0':'6px'};display:block;margin-bottom:8px;object-fit:contain;`;
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
        if (messageId) wrapper.setAttribute('data-chat-message-id', messageId);
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
        if (role === 'user') {
            messageIndex.push({ text: text.slice(0, 200), domRef: div });
            renderNavigator();
        }
    }
    // scroll handled by complete handler after response arrives
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

// ============================================================
// TOOLS PANEL
// ============================================================
let toolsPanelOpen = false;

function toggleToolsPanel() {
    toolsPanelOpen = !toolsPanelOpen;
    const panel  = document.getElementById('toolsPanel');
    const button = document.getElementById('toolsToggle');
    if (toolsPanelOpen) { panel.classList.add('open'); button.textContent = '✖ Tools'; }
    else                { panel.classList.remove('open'); button.textContent = '🔧 Tools'; }
}

// ── Favorites ─────────────────────────────────────────────────
let _favorites = []; // in-memory cache, loaded from DB on connect

function getFavorites() { return _favorites; }

function setFavorites(favs) {
    _favorites = favs;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type:  "set_setting",
            key:   FAVORITES_SETTING_KEY,
            value: JSON.stringify(favs)
        }));
    }
}

function isFavorite(toolName) { return _favorites.includes(toolName); }

function toggleFavorite(toolName) {
    const favs = [..._favorites];
    const idx = favs.indexOf(toolName);
    if (idx === -1) favs.push(toolName);
    else favs.splice(idx, 1);
    setFavorites(favs);
    return idx === -1;
}

function addFavorite(toolName) {
    if (!_favorites.includes(toolName)) setFavorites([..._favorites, toolName]);
}

function removeFavorite(toolName) {
    setFavorites(_favorites.filter(n => n !== toolName));
}
// ──────────────────────────────────────────────────────────────

function buildToolItem(tool, { onFavoriteToggle, draggable: isDraggable = false } = {}) {
    const item = document.createElement('div');
    const isDisabled = tool.enabled === false;
    item.className = 'tool-item' + (isDisabled ? ' tool-disabled' : '');
    if (isDraggable) {
        item.dataset.toolName = tool.name;
        item.setAttribute('draggable', 'true');
    }

    const fullDesc = tool.description || '';
    const previewDesc = fullDesc.split(/\.\s+/)[0].replace(/\n.*/s, '').trim();
    const params = tool.required_params || [];
    const optionalParams = [];
    if (tool.example) {
        const optMatches = tool.example.matchAll(/\[(\w+)=["'][^"']*["']\]/g);
        for (const m of optMatches) optionalParams.push(m[1]);
    }
    const paramsHtml = (params.length || optionalParams.length)
        ? `<div class="tool-item-params">${
            params.map(p => `<span class="tool-param">${p.name}</span>`).join('')
          }${
            optionalParams.map(p => `<span class="tool-param tool-param-optional">[${p}]</span>`).join('')
          }</div>`
        : '';

    const TAG_COLORS = {
        destructive: '#e74c3c', write: '#e67e22', external: '#8e44ad',
        ai: '#2980b9', search: '#27ae60', read: '#555', media: '#16a085',
        calendar: '#2471a3', email: '#2471a3', notes: '#7d6608',
        code: '#1a5276', system: '#6c3483', rag: '#117a65',
        vision: '#922b21',
    };
    const tags = tool.tags || [];
    const tagsHtml = tags.length
        ? `<div class="tool-item-tags">${tags.map(t => {
            const bg = TAG_COLORS[t] || '#444';
            return `<span class="tool-tag" style="background:${bg}">${t}</span>`;
        }).join('')}</div>`
        : '';

    const rateHtml = tool.rate_limit
        ? `<span class="tool-rate-limit" title="Rate limit">⏱ ${tool.rate_limit}</span>`
        : '';
    const disabledHtml = isDisabled ? `<span class="tool-disabled-badge">disabled</span>` : '';

    const fav = isFavorite(tool.name);
    const gripHtml = isDraggable ? `<span class="tool-drag-handle" title="Drag to reorder" aria-hidden="true">⠿</span>` : '';
    item.innerHTML = `
        <div class="tool-item-header">
            ${gripHtml}<div class="tool-item-name">${tool.name}${disabledHtml}</div>
            <div class="tool-item-actions">
                ${rateHtml}
                <button class="tool-fav-btn${fav ? ' active' : ''}" title="${fav ? 'Remove from favorites' : 'Add to favorites'}" aria-label="Toggle favorite">★</button>
            </div>
        </div>
        <div class="tool-item-desc">${previewDesc || 'No description available.'}</div>
        ${tagsHtml}
        ${paramsHtml}`;

    const starBtn = item.querySelector('.tool-fav-btn');
    starBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const added = toggleFavorite(tool.name);
        starBtn.classList.toggle('active', added);
        starBtn.title = added ? 'Remove from favorites' : 'Add to favorites';
        if (onFavoriteToggle) onFavoriteToggle();
    });

    item.addEventListener('click', (e) => {
        if (e.target === starBtn) return;
        const inputEl = document.getElementById('input');
        inputEl.value = getToolPrompt(tool.name, params);
        inputEl.focus();
        const firstQuote = inputEl.value.indexOf('""');
        if (firstQuote !== -1) {
            inputEl.selectionStart = inputEl.selectionEnd = firstQuote + 1;
        } else {
            inputEl.selectionStart = inputEl.selectionEnd = inputEl.value.length;
        }
        if (window.innerWidth <= 768) toggleToolsPanel();
    });

    return item;
}

let _allTools = [];

function attachFavDragHandlers(container) {
    let dragSrc = null;

    container.querySelectorAll('.tool-item[draggable="true"]').forEach(item => {
        item.addEventListener('dragstart', (e) => {
            dragSrc = item;
            item.classList.add('fav-dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', item.dataset.toolName);
        });

        item.addEventListener('dragend', () => {
            dragSrc = null;
            container.querySelectorAll('.tool-item').forEach(i => {
                i.classList.remove('fav-dragging', 'fav-drag-over');
            });
        });

        item.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            if (item !== dragSrc) {
                container.querySelectorAll('.tool-item').forEach(i => i.classList.remove('fav-drag-over'));
                item.classList.add('fav-drag-over');
            }
        });

        item.addEventListener('dragleave', () => {
            item.classList.remove('fav-drag-over');
        });

        item.addEventListener('drop', (e) => {
            e.preventDefault();
            if (!dragSrc || dragSrc === item) return;

            const srcName = dragSrc.dataset.toolName;
            const tgtName = item.dataset.toolName;

            const favs = [..._favorites];
            const srcIdx = favs.indexOf(srcName);
            const tgtIdx = favs.indexOf(tgtName);
            if (srcIdx === -1 || tgtIdx === -1) return;

            favs.splice(srcIdx, 1);
            favs.splice(tgtIdx, 0, srcName);
            setFavorites(favs);
            renderToolsPanel(_allTools);
        });
    });
}

function renderToolsPanel(tools) {
    _allTools = tools;
    const body = document.getElementById('toolsBody');
    if (!body) return;

    // Snapshot which groups are currently open before wiping
    const openGroups = new Set();
    body.querySelectorAll('.tools-category.open').forEach(cat => {
        const label = cat.querySelector('.tools-category-header span')?.textContent?.trim();
        if (label) openGroups.add(label);
    });

    body.innerHTML = '';

    // ── Favorites section ──────────────────────────────────────
    const favNames = getFavorites();
    const favTools = favNames.map(n => tools.find(t => t.name === n)).filter(Boolean);

    const favCat = document.createElement('div');
    favCat.className = 'tools-category tools-favorites-category';

    const favHeader = document.createElement('div');
    favHeader.className = 'tools-category-header tools-favorites-header';
    favHeader.innerHTML = `<span>Favorites <span class="tools-fav-count">(${favTools.length})</span></span><span class="tools-category-arrow">▶</span>`;
    favHeader.onclick = () => favCat.classList.toggle('open');

    const favItemsDiv = document.createElement('div');
    favItemsDiv.className = 'tools-category-items';

    if (favTools.length === 0) {
        favItemsDiv.innerHTML = '<div class="tools-fav-empty">Star a tool below to add it here</div>';
    } else {
        favTools.forEach(tool => {
            favItemsDiv.appendChild(buildToolItem(tool, {
                draggable: true,
                onFavoriteToggle: () => renderToolsPanel(_allTools)
            }));
        });
        attachFavDragHandlers(favItemsDiv);
    }

    favCat.appendChild(favHeader);
    favCat.appendChild(favItemsDiv);
    if (favTools.length > 0) favCat.classList.add('open');
    body.appendChild(favCat);

    // ── Server groups ──────────────────────────────────────────
    const groups = {};
    tools.forEach(t => {
        const server = t.source_server || 'unknown';
        if (!groups[server]) groups[server] = [];
        groups[server].push(t);
    });

    Object.keys(groups).sort().forEach(serverName => {
        const items = groups[serverName];
        const cat = document.createElement('div');
        cat.className = 'tools-category';

        const isExternal = items.some(t => t.external);
        const header = document.createElement('div');
        header.className = 'tools-category-header';
        const allFaved = items.every(t => isFavorite(t.name));
        const groupStar = document.createElement('button');
        groupStar.className = 'tool-fav-btn group-fav-btn' + (allFaved ? ' active' : '');
        groupStar.title = allFaved ? 'Remove all from favorites' : 'Add all to favorites';
        groupStar.textContent = '★';
        groupStar.addEventListener('click', (e) => {
            e.stopPropagation();
            const nowAllFaved = items.every(t => isFavorite(t.name));
            items.forEach(t => {
                if (nowAllFaved) removeFavorite(t.name);
                else addFavorite(t.name);
            });
            renderToolsPanel(_allTools);
        });
        header.innerHTML = `<span>${serverName}${isExternal ? ' <span style="color:#858585;font-size:0.7rem;font-weight:normal;">[external]</span>' : ''} <span style="color:#858585;font-weight:normal;">(${items.length})</span></span><span class="tools-category-arrow">▶</span>`;
        header.insertBefore(groupStar, header.firstChild);
        header.onclick = (e) => {
            if (e.target === groupStar) return;
            body.querySelectorAll('.tools-category:not(.tools-favorites-category)').forEach(c => {
                if (c !== cat) c.classList.remove('open');
            });
            cat.classList.toggle('open');
        };

        const itemsDiv = document.createElement('div');
        itemsDiv.className = 'tools-category-items';

        items.sort((a, b) => a.name.localeCompare(b.name)).forEach(tool => {
            itemsDiv.appendChild(buildToolItem(tool, {
                onFavoriteToggle: () => renderToolsPanel(_allTools)
            }));
        });

        cat.appendChild(header);
        cat.appendChild(itemsDiv);
        // Restore open state
        const catLabel = cat.querySelector('.tools-category-header span')?.textContent?.trim();
        if (catLabel && openGroups.has(catLabel)) cat.classList.add('open');
        body.appendChild(cat);
    });
}

let _toolPrompts = {};
function buildToolPrompts(tools) {
    _toolPrompts = {};
    tools.forEach(t => { if (t.example) _toolPrompts[t.name] = t.example; });
}
function getToolPrompt(toolName, params) {
    // Prefer @tool_meta example — has bracket notation for optional params
    if (_toolPrompts[toolName]) return _toolPrompts[toolName];
    // Fall back to generating from required params only
    if (params && params.length > 0) {
        const paramStr = params.map(p => `${p.name}=""`).join(' ');
        return `use ${toolName}: ${paramStr}`;
    }
    return `use ${toolName}`;
}

sendBtn.addEventListener('click', send);
input.addEventListener("keydown", (e) => { if (e.key==="Enter"&&!e.shiftKey) { e.preventDefault(); send(); } });
document.getElementById('toolsToggle').addEventListener('click', toggleToolsPanel);
const _toolsCloseBtn = document.getElementById('toolsCloseBtn');
if (_toolsCloseBtn) _toolsCloseBtn.addEventListener('click', toggleToolsPanel);
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

// ============================================================
// INACTIVITY → MATRIX RAIN EASTER EGG
// Only active on [data-theme="matrix"].
// Triggers after 5 min of inactivity. Any activity stops it.
// ============================================================
(() => {
    // ── Configurable timing ──────────────────────────────────────────────
    const INACTIVITY_MS  = 5 * 60 * 1000;
    const BURST_FIRST_MIN    = 120;
    const BURST_FIRST_MAX    = 240;
    const BURST_DURATION_MIN = 10;
    const BURST_DURATION_MAX = 18;
    const BURST_INTERVAL_MIN = 30;
    const BURST_INTERVAL_MAX = 60;

    // ── Drop speed (0–100): 0=frozen, 1=barely moving, 50=default, 100=fastest
    const DROP_SPEED     = 98;

    // ── Speed tiers: frameSkip (lower=faster) + weight (0–100, should sum to 100)
    const SPEED_TIERS = [
        { frameSkip: 2,  weight: 50 },  // fast
        { frameSkip: 4,  weight: 42 },  // medium
        { frameSkip: 10, weight: 4  },  // slow
        { frameSkip: 13, weight: 4  },  // super slow
    ];

    // ── Dual stream frequency (0–100): 0=never, 100=very frequent, default=50
    const DUAL_FREQUENCY = 50;
    const DUAL_MIN_GAP   = 10;
    // ─────────────────────────────────────────────────────────────────────

    const THEME_NAME = 'matrix';
    const CHARS      = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEF'.split('');
    const fontSize   = 14;

    let inactivityTimer = null;
    let canvas          = null;
    let ctx             = null;
    let rafId           = null;
    let frameCount      = 0;

    let cols     = [];
    let greenLUT = [];
    let booting    = true;
    let bootStream = null; // pre-built color strings — avoids template alloc per trail entry

    // Burst state
    let burstActive      = false;
    let burstFramesLeft  = 0;
    let burstTotalFrames  = 0;
    let nextBurstFrame    = 0;
    let burstEpicenter    = -1;
    let burstEpicenterRow = -1;
    let burstRadius       = 0;
    let burstAngle        = 0;
    let burstNoise        = null;
    let burstJag          = null;

    function isMatrix() {
        return document.documentElement.getAttribute('data-theme') === THEME_NAME;
    }

    function resetTimer() {
        clearTimeout(inactivityTimer);
        if (canvas) stopMatrix();
        if (!isMatrix()) return;
        inactivityTimer = setTimeout(startMatrix, INACTIVITY_MS);
    }

    // Convert DROP_SPEED (0–100) to a frame-skip multiplier
    function speedMultiplier() {
        if (DROP_SPEED <= 0)   return 999;
        if (DROP_SPEED >= 100) return 1;
        return Math.round(1 + (99 - DROP_SPEED) / 99 * 59);
    }

    // Pick a frameSkip from SPEED_TIERS using weighted random, scaled by DROP_SPEED
    function makeFrameSkip() {
        const mult = speedMultiplier();
        let total = 0;
        for (const t of SPEED_TIERS) total += t.weight;
        let r = Math.random() * total;
        for (const t of SPEED_TIERS) {
            r -= t.weight;
            if (r <= 0) return Math.max(1, t.frameSkip * mult);
        }
        return Math.max(1, SPEED_TIERS[SPEED_TIERS.length - 1].frameSkip * mult);
    }

    function makeSteps(frameSkip) {
        const minSkip = SPEED_TIERS[0].frameSkip;
        const maxSkip = SPEED_TIERS[SPEED_TIERS.length - 1].frameSkip;
        const ratio   = Math.min(1, (frameSkip - minSkip) / Math.max(1, maxSkip - minSkip));
        return Math.round(28 + (70 - 28) * ratio);
    }

    function makeStream(delayMax) {
        const speed = makeFrameSkip();
        return {
            row: 0, speed, steps: makeSteps(speed),
            delay: Math.random() * (delayMax ?? 60) | 0,
            trails: [], active: true, suppressTicks: 0,
        };
    }

    // Second stream: same speed as primary (trails exactly behind it)
    function makeSecondStream(primarySpeed, delayMax) {
        return {
            row: 0, speed: primarySpeed, steps: makeSteps(primarySpeed),
            delay: Math.random() * (delayMax ?? 30) | 0,
            trails: [], active: true, suppressTicks: 0,
        };
    }

    // Convert DUAL_FREQUENCY (0–100) to a cooldown value
    function dualCooldown() {
        if (DUAL_FREQUENCY <= 0) return 999999;
        const min = Math.round(10 + (100 - DUAL_FREQUENCY) / 100 * 190);
        const max = min + Math.round(20 + (100 - DUAL_FREQUENCY) / 100 * 180);
        return min + (Math.random() * (max - min) | 0);
    }

    function makeBurstNoise(epi, numCols) {
        const reach = 80;
        const noise = new Float32Array(numCols);
        for (let i = 0; i < numCols; i++) {
            const absDelta = i > epi ? i - epi : epi - i;
            const edgeBias = absDelta / reach;
            noise[i] = Math.max(0, Math.min(1,
                edgeBias * 0.7 + Math.random() * 0.5 - 0.1
            ));
        }
        return noise;
    }

    function makeBurstJag(numCols, w) {
        const jag = new Float32Array(numCols);
        let walk  = 0;
        for (let i = 0; i < numCols; i++) {
            walk += (Math.random() - 0.5) * w * 0.8;
            walk *= 0.85;
            jag[i] = walk;
        }
        return jag;
    }
    const FAST_TIER_SKIP = SPEED_TIERS[0].frameSkip * speedMultiplier() * 1.5;

    function startMatrix() {
        if (!isMatrix() || canvas) return;
        const mainContainer = document.getElementById('mainContainer');
        if (!mainContainer) return;

        const containerRect = mainContainer.getBoundingClientRect();

        canvas        = document.createElement('canvas');
        canvas.width  = containerRect.width;
        canvas.height = containerRect.height;

        Object.assign(canvas.style, {
            position:      'absolute',
            top:           '0',
            height:        '100%',
            pointerEvents: 'none',
            opacity:       '1',
            zIndex:        '10',
            borderRadius:  '0',
        });

        if (window.getComputedStyle(mainContainer).position === 'static') {
            mainContainer.style.position = 'relative';
        }
        mainContainer.appendChild(canvas);
        ctx = canvas.getContext('2d');

        // Build green LUT once — 256 pre-allocated color strings
        greenLUT = new Array(256);
        for (let g = 0; g < 256; g++) greenLUT[g] = `rgb(0,${g},0)`;

        chat.style.background = '#050505';
        mainContainer.style.background = '#050505';

        // Boot: single medium-speed center stream
        booting = true;
        const mult    = speedMultiplier();
        const medSkip = Math.max(1, (SPEED_TIERS[1] ? SPEED_TIERS[1].frameSkip : 6) * mult);
        bootStream = { row: 0, speed: medSkip, steps: makeSteps(medSkip), trails: [] };

        initColumns();

        nextBurstFrame = 999999; // fires after boot completes

        rafId = requestAnimationFrame(drawFrame);
        themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    }

    function initColumns() {
        const n = Math.floor(canvas.width / fontSize);
        cols = Array.from({ length: n }, () => ({
            streams:  [ makeStream(booting ? 999999 : 60) ],
            spawnCD:  dualCooldown(),
            mapA:     Object.create(null),
            mapB:     Object.create(null),
            useA:     true,
        }));
    }

    function drawFrame() {
        if (!ctx || !canvas) return;
        frameCount++;

        const maxRow  = Math.floor(canvas.height / fontSize);
        const numCols = cols.length;
        const fw      = fontSize;

        ctx.font = `${fw}px "Share Tech Mono", monospace`;
        const glowLUT = `rgba(0,255,0,0.6)`;

        // ── Boot phase: single center stream ──────────────────────────────
        if (booting && bootStream) {
            const bs      = bootStream;
            const introDropRow = maxRow >> 2;
            const centerX = Math.floor(numCols / 2) * fw;
            const CHARS   = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEF';

            if (frameCount % bs.speed === 0) {
                const char = CHARS[Math.random() * CHARS.length | 0];
                for (let t = 0; t < bs.trails.length; t++) bs.trails[t].brightness--;
                for (let t = bs.trails.length - 1; t >= 0; t--) {
                    if (bs.trails[t].brightness <= 0) { bs.trails[t] = bs.trails[bs.trails.length-1]; bs.trails.pop(); }
                }
                bs.trails.push({ row: bs.row, char, brightness: bs.steps + 6 });
                bs.row++;
            }

            for (let t = 0; t < bs.trails.length; t++) {
                const e  = bs.trails[t];
                const cy = e.row * fw;
                ctx.fillStyle = '#050505';
                ctx.fillRect(centerX, cy, fw, fw);
                const isHead = (t === bs.trails.length - 1);
                if (isHead) {
                    ctx.fillStyle = glowLUT;
                    ctx.fillText(e.char, centerX - 1, cy + fw - 2);
                    ctx.fillText(e.char, centerX + 1, cy + fw - 2);
                    ctx.fillText(e.char, centerX,     cy + fw - 3);
                    ctx.fillText(e.char, centerX,     cy + fw - 1);
                    ctx.fillStyle = '#00ff41';
                } else {
                    const cl = Math.min(1, e.brightness / bs.steps);
                    ctx.fillStyle = greenLUT[cl * cl * 255 | 0];
                }
                ctx.fillText(e.char, centerX, cy + fw - 2);
            }

            if (bs.row >= introDropRow) {
                booting = false;
                const centerCol = Math.floor(numCols / 2);
                // Hand off the boot stream to the center column so it continues naturally
                initColumns();
                const liveStream = {
                    row: bs.row, speed: bs.speed, steps: bs.steps,
                    delay: 0, trails: bs.trails.slice(), active: true, suppressTicks: 0,
                };
                cols[centerCol].streams[0] = liveStream;
                bootStream = null;
                nextBurstFrame = frameCount + Math.round(
                    (BURST_FIRST_MIN + Math.random() * (BURST_FIRST_MAX - BURST_FIRST_MIN)) * 60
                );
            }

            rafId = requestAnimationFrame(drawFrame);
            return;
        }

        const burstElapsed  = burstActive ? burstTotalFrames - burstFramesLeft : 0;
        const burstProgress = burstActive && burstTotalFrames > 0 ? burstElapsed / burstTotalFrames : 0;
        const BURST_REACH   = 140;
        const BURST_WIDTH   = 10;

        // ── Burst (lightning) ─────────────────────────────────────────────
        if (!burstActive && frameCount >= nextBurstFrame) {
            burstActive      = true;
            burstTotalFrames = Math.round((BURST_DURATION_MIN + Math.random() * (BURST_DURATION_MAX - BURST_DURATION_MIN)) * 60);
            burstFramesLeft  = burstTotalFrames;
            burstEpicenter    = Math.random() * numCols | 0;
            burstEpicenterRow = Math.floor(canvas.height / fontSize * Math.random());
            burstAngle        = (Math.random() < 0.5 ? 1 : -1) * (0.18 * (0.5 + Math.random()));
            burstRadius       = 0;
            burstNoise        = makeBurstNoise(burstEpicenter, numCols);
            burstJag          = makeBurstJag(numCols, 10);
        }
        if (burstActive) {
            if (--burstFramesLeft <= 0) {
                burstActive = false; burstTotalFrames = 0;
                burstEpicenter = -1; burstEpicenterRow = -1; burstRadius = 0;
                burstAngle = 0; burstNoise = null; burstJag = null; burstJag = null;
                nextBurstFrame = frameCount + Math.round(
                    (BURST_INTERVAL_MIN + Math.random() * (BURST_INTERVAL_MAX - BURST_INTERVAL_MIN)) * 60
                );
            }
        }

        for (let i = 0; i < numCols; i++) {
            const col = cols[i];
            const x   = i * fw;

            // ── Lightning column intensity (noise-based erosion) ────────────
            let colBIntens = 0;
            if (burstActive && burstEpicenter >= 0 && burstNoise) {
                const colDelta    = i - burstEpicenter;
                const absDelta    = colDelta < 0 ? -colDelta : colDelta;
                const noiseThresh = burstNoise[i] ?? 1;
                if (absDelta <= BURST_REACH && burstProgress < noiseThresh) {
                    const reach_t = 1 - absDelta / BURST_REACH;
                    colBIntens = reach_t * reach_t;
                }
            }

            const rb        = colBIntens * 255 | 0;
            const glowAlpha = 0.6 + colBIntens * 0.5;

            // ── Try to spawn second stream (fast columns only, controlled by DUAL_FREQUENCY) ──
            if (DUAL_FREQUENCY > 0 && col.streams.length === 1 && col.streams[0].active
                && col.streams[0].speed <= FAST_TIER_SKIP) {
                col.spawnCD--;
                if (col.spawnCD <= 0) {
                    if (col.streams[0].row > DUAL_MIN_GAP * 2) {
                        col.streams.push(makeSecondStream(col.streams[0].speed, 30));
                    }
                    col.spawnCD = dualCooldown();
                }
            }

            // ── STEP 1: Advance state ──────────────────────────────────────
            for (let s = 0; s < col.streams.length; s++) {
                const st = col.streams[s];
                if (!st.active) continue;
                if (frameCount % st.speed !== 0) continue;
                if (st.delay > 0) { st.delay--; continue; }

                let tooClose = false;
                for (let o = 0; o < col.streams.length; o++) {
                    if (o === s) continue;
                    if (col.streams[o].active && Math.abs(st.row - col.streams[o].row) < DUAL_MIN_GAP) {
                        tooClose = true; break;
                    }
                }
                if (tooClose) {
                    if (++st.suppressTicks > 120) st.active = false;
                    continue;
                }
                st.suppressTicks = 0;

                if (st.row < maxRow) {
                    const char = CHARS[Math.random() * CHARS.length | 0];
                    for (const e of st.trails) e.brightness--;
                    for (let t = st.trails.length - 1; t >= 0; t--) {
                        if (st.trails[t].brightness <= 0) st.trails.splice(t, 1);
                    }
                    st.trails.push({ row: st.row, char, brightness: st.steps + 6 });
                    st.row++;
                } else {
                    st.active = false;
                    for (let t = 0; t < st.trails.length; t++) st.trails[t].brightness--;
                    for (let t = st.trails.length - 1; t >= 0; t--) {
                        if (st.trails[t].brightness <= 0) { st.trails[t] = st.trails[st.trails.length-1]; st.trails.pop(); }
                    }
                }
            }

            // Fade inactive streams on their speed tick
            for (let s = 0; s < col.streams.length; s++) {
                const st = col.streams[s];
                if (st.active || frameCount % st.speed !== 0) continue;
                for (let t = 0; t < st.trails.length; t++) st.trails[t].brightness--;
                for (let t = st.trails.length - 1; t >= 0; t--) {
                    if (st.trails[t].brightness <= 0) { st.trails[t] = st.trails[st.trails.length-1]; st.trails.pop(); }
                }
            }

            // Remove fully faded inactive streams
            for (let s = col.streams.length - 1; s >= 0; s--) {
                if (!col.streams[s].active && col.streams[s].trails.length === 0) col.streams.splice(s, 1);
            }
            if (col.streams.length === 0) col.streams.push(makeStream(Math.random() * 60 | 0));
        }

        // ── STEP 2: Per-cell render with per-column row tracking ───────────
        // Use Uint8Array as a fast bitset to track which rows are occupied this frame.
        // Clear vacated rows (in prev but not cur), then draw occupied rows.
        // Use existing maxRow from above

        for (let i = 0; i < numCols; i++) {
            const col = cols[i];
            const x   = i * fw;

            // Uint8Arrays reused per column — allocated once on first use
            if (!col.curRows)  col.curRows  = new Uint8Array(maxRow);
            if (!col.prevRows) col.prevRows = new Uint8Array(maxRow);

            // Clear curRows for this frame
            col.curRows.fill(0);

            // Mark occupied rows
            for (let s = 0; s < col.streams.length; s++) {
                const trails = col.streams[s].trails;
                for (let t = 0; t < trails.length; t++) {
                    const r = trails[t].row;
                    if (r < maxRow) col.curRows[r] = 1;
                }
            }

            // Clear vacated rows
            ctx.fillStyle = '#050505';
            for (let r = 0; r < maxRow; r++) {
                if (col.prevRows[r] && !col.curRows[r]) ctx.fillRect(x, r * fw, fw, fw);
            }

            const colDelta = burstActive && burstEpicenter >= 0 ? i - burstEpicenter : 0;
            let colBIntens = 0;
            if (burstActive && burstEpicenter >= 0 && burstNoise) {
                const absDelta    = colDelta < 0 ? -colDelta : colDelta;
                const noiseThresh = burstNoise[i] ?? 1;
                if (absDelta <= BURST_REACH && burstProgress < noiseThresh) {
                    const reach_t = 1 - absDelta / BURST_REACH;
                    colBIntens = reach_t * reach_t;
                }
            }

            for (let s = 0; s < col.streams.length; s++) {
                const st      = col.streams[s];
                const trails  = st.trails;
                const headIdx = st.active ? trails.length - 1 : -1;

                for (let t = 0; t < trails.length; t++) {
                    const e  = trails[t];
                    const cy = e.row * fw;

                    // Per-entry lightning intensity — row falloff around bolt path
                    let bIntens = 0;
                    if (colBIntens > 0) {
                        const jagOff  = burstJag ? burstJag[i] : 0;
                        const boltRow = burstEpicenterRow + burstAngle * colDelta + jagOff;
                        const rowDist = e.row - boltRow;
                        const absDist = rowDist < 0 ? -rowDist : rowDist;
                        if (absDist < BURST_WIDTH * 4) {
                            const rowFalloff = Math.exp(-(rowDist * rowDist) / (2 * BURST_WIDTH * BURST_WIDTH));
                            bIntens = colBIntens * rowFalloff;
                        }
                    }

                    const rb        = bIntens * 255 | 0;
                    const whiten    = bIntens * bIntens;
                    const glowAlpha = 0.6 + bIntens * 0.5;

                    // Clear cell then draw
                    ctx.fillStyle = '#050505';
                    ctx.fillRect(x, cy, fw, fw);

                    if (t === headIdx) {
                        ctx.fillRect(x - 1, cy - 1, fw + 2, fw + 2);
                        if (bIntens > 0) {
                            ctx.fillStyle = `rgba(${rb},255,${rb},${glowAlpha})`;
                            ctx.fillText(e.char, x - 1, cy + fw - 2);
                            ctx.fillText(e.char, x + 1, cy + fw - 2);
                            ctx.fillText(e.char, x,     cy + fw - 3);
                            ctx.fillText(e.char, x,     cy + fw - 1);
                            const hw = 255 * whiten | 0;
                            ctx.fillStyle = `rgb(${hw},255,${hw})`;
                        } else {
                            ctx.fillStyle = glowLUT;
                            ctx.fillText(e.char, x - 1, cy + fw - 2);
                            ctx.fillText(e.char, x + 1, cy + fw - 2);
                            ctx.fillText(e.char, x,     cy + fw - 3);
                            ctx.fillText(e.char, x,     cy + fw - 1);
                            ctx.fillStyle = '#00ff41';
                        }
                        ctx.fillText(e.char, x, cy + fw - 2);
                    } else {
                        const cl1 = e.brightness / st.steps;
                        const cl  = cl1 > 1 ? 1 : cl1;
                        if (bIntens > 0) {
                            const base_g = cl * cl * 255 | 0;
                            const g      = base_g + (bIntens * 220 | 0);
                            const boost  = whiten * 255 | 0;
                            const trb    = ((bIntens * cl * 230 | 0) + boost) > 255 ? 255 : (bIntens * cl * 230 | 0) + boost;
                            const tg     = (g + boost) > 255 ? 255 : g + boost;
                            ctx.fillStyle = `rgb(${trb},${tg},${trb})`;
                        } else {
                            ctx.fillStyle = greenLUT[cl * cl * 255 | 0];
                        }
                        ctx.fillText(e.char, x, cy + fw - 2);
                    }
                }
            }

            // Swap cur/prev for next frame
            const tmp    = col.prevRows;
            col.prevRows = col.curRows;
            col.curRows  = tmp;
        }

        rafId = requestAnimationFrame(drawFrame);
    }

    function stopMatrix() {
        if (rafId)  { cancelAnimationFrame(rafId); rafId = null; }
        if (canvas) { canvas.remove(); canvas = null; ctx = null; }
        const chat = document.getElementById('chat');
        if (chat) chat.style.background = '';
        const chatContainer = document.getElementById('chatContainer');
        if (chatContainer) chatContainer.style.background = '';
        cols       = [];
        frameCount = 0;
        booting    = true;
        bootStream = null;
        burstActive = false; burstTotalFrames = 0;
        burstEpicenter = -1; burstEpicenterRow = -1; burstRadius = 0; burstAngle = 0; burstNoise = null; burstJag = null;
        themeObserver.disconnect();
    }

    const themeObserver = new MutationObserver(() => {
        if (!isMatrix()) { stopMatrix(); clearTimeout(inactivityTimer); }
        else             { resetTimer(); }
    });

    window.addEventListener('resize', () => {
        if (!canvas) return;
        const chat          = document.getElementById('chat');
        const chatContainer = document.getElementById('chatContainer');
        if (!chat || !chatContainer) return;
        const chatRect      = chat.getBoundingClientRect();
        const containerRect = chatContainer.getBoundingClientRect();
        canvas.width        = chatRect.width;
        canvas.height       = containerRect.height;
        canvas.style.top    = '0';
        canvas.style.left   = (chatRect.left - containerRect.left) + 'px';
        canvas.style.width  = chatRect.width + 'px';
        canvas.style.height = '100%';
        initColumns();
    });

    ['mousemove', 'mousedown', 'keydown', 'touchstart', 'scroll'].forEach(evt =>
        window.addEventListener(evt, resetTimer, { passive: true })
    );

    const _chatEl = document.getElementById('chat');
    if (_chatEl) _chatEl.addEventListener('scroll', resetTimer, { passive: true });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', resetTimer);
    } else {
        resetTimer();
    }
})();

// ============================================================
// MATRIX THEME – COSMETIC BLINKING CURSOR ON INPUT
// ============================================================
(() => {
    const textarea   = document.getElementById('input');
    const cursorText = document.getElementById('matrixCursorText');
    const inputRow   = document.getElementById('inputRow');
    if (!textarea || !cursorText || !inputRow) return;

    inputRow.style.position = 'relative';

    function syncCursor() {
        const pos    = textarea.selectionStart ?? textarea.value.length;
        let before   = textarea.value.substring(0, pos);
        if (before.endsWith('\n')) before += '\u200b';
        cursorText.textContent = before;
    }

    textarea.addEventListener('input',     syncCursor);
    textarea.addEventListener('keydown',   syncCursor);
    textarea.addEventListener('keyup',     syncCursor);
    textarea.addEventListener('click',     syncCursor);
    textarea.addEventListener('mouseup',   syncCursor);
    textarea.addEventListener('focus',   () => { syncCursor(); inputRow.classList.add('matrix-focused'); });
    textarea.addEventListener('blur',    () => inputRow.classList.remove('matrix-focused'));

    syncCursor();
})();
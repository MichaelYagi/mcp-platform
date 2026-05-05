/**
 * tests/js/setup.js
 *
 * Loaded by Jest before every test file (see package.json "setupFiles").
 * Provides:
 *   - All browser API mocks (localStorage, WebSocket, Canvas, etc.)
 *   - global.loadDom(htmlFile)  — loads a real HTML file into jsdom
 *   - global.extractFn(code)    — evaluates a JS snippet and returns exports
 */

'use strict';

const fs   = require('fs');
const path = require('path');

// ── Paths ─────────────────────────────────────────────────────────────────────
const ROOT   = path.resolve(__dirname, '../../');
const UI_DIR = path.join(ROOT, 'client/ui');

// ── localStorage ─────────────────────────────────────────────────────────────
const localStorageMock = (() => {
    let store = {};
    return {
        getItem:    (k)    => store[k] ?? null,
        setItem:    (k, v) => { store[k] = String(v); },
        removeItem: (k)    => { delete store[k]; },
        clear:      ()     => { store = {}; },
    };
})();
Object.defineProperty(global, 'localStorage', { value: localStorageMock });

// ── WebSocket ─────────────────────────────────────────────────────────────────
global.WebSocket = class MockWebSocket {
    constructor(url) {
        this.url        = url;
        this.readyState = 1;
        this._sent      = [];
        this.onopen = this.onmessage = this.onerror = this.onclose = null;
    }
    send(data)  { this._sent.push(data); }
    close()     { this.readyState = 3; this.onclose && this.onclose({}); }
    /** Test helper: simulate an inbound server message. */
    simulateMessage(data) {
        this.onmessage && this.onmessage({ data: JSON.stringify(data) });
    }
};
global.WebSocket.OPEN       = 1;
global.WebSocket.CONNECTING = 0;
global.WebSocket.CLOSING    = 2;
global.WebSocket.CLOSED     = 3;

// ── CSS custom property stubs ─────────────────────────────────────────────────
const CSS_VARS = {
    '--theme-list':   'default,matrix,tokyo,te,mono',
    '--swatch1':      '#333333',
    '--swatch2':      '#555555',
    '--swatch3':      '#777777',
    '--chart-blue':   '#3498db',
    '--chart-purple': '#9b59b6',
    '--chart-red':    '#e74c3c',
    '--chart-green':  '#2ecc71',
    '--chart-orange': '#e67e22',
};
global.getComputedStyle = () => ({
    getPropertyValue: (prop) => CSS_VARS[prop] ?? '',
    position: 'static',
});

// ── Canvas ────────────────────────────────────────────────────────────────────
HTMLCanvasElement.prototype.getContext = jest.fn(() => ({
    font: '', fillStyle: '',
    fillRect: jest.fn(), fillText: jest.fn(), clearRect: jest.fn(),
    beginPath: jest.fn(), arc: jest.fn(), fill: jest.fn(), stroke: jest.fn(),
    moveTo: jest.fn(), lineTo: jest.fn(), save: jest.fn(), restore: jest.fn(),
    scale: jest.fn(), translate: jest.fn(),
}));

// ── Timers / animation ────────────────────────────────────────────────────────
global.requestAnimationFrame = (cb) => setTimeout(cb, 0);
global.cancelAnimationFrame  = (id) => clearTimeout(id);

// ── Observer stubs ────────────────────────────────────────────────────────────
global.IntersectionObserver = class {
    observe() {} unobserve() {} disconnect() {}
};
global.MutationObserver = class {
    constructor(cb) { this._cb = cb; }
    observe() {} disconnect() {}
};
global.ResizeObserver = class {
    observe() {} unobserve() {} disconnect() {}
};

// ── DOM helpers ───────────────────────────────────────────────────────────────
window.HTMLElement.prototype.scrollIntoView = jest.fn();

// ── Clipboard ─────────────────────────────────────────────────────────────────
Object.defineProperty(global.navigator, 'clipboard', {
    value: { writeText: jest.fn().mockResolvedValue(undefined) },
    configurable: true,
});

// ── Intl.DateTimeFormat ───────────────────────────────────────────────────────
// Minimal stub — only resolvedOptions() is needed by addMessage.
// Avoid Object.assign on DTF prototype instances (read-only props).
const _OrigDTF = Intl.DateTimeFormat;
global.Intl = {
    ...global.Intl,
    DateTimeFormat: function(...args) {
        const fmt = new _OrigDTF(...args);
        const stub = Object.create(null);
        stub.resolvedOptions = () => ({ locale: 'en-US', timeZone: 'UTC' });
        stub.format = fmt.format.bind(fmt);
        stub.formatToParts = fmt.formatToParts ? fmt.formatToParts.bind(fmt) : () => [];
        return stub;
    },
};

// ── Chart.js stub ─────────────────────────────────────────────────────────────
global.Chart = class MockChart {
    constructor(ctx, config) {
        this.data     = config.data    || { labels: [], datasets: [] };
        this.options  = config.options || {};
        this._updates = [];
    }
    update(mode) { this._updates.push(mode); }
    destroy()    {}
};

// ── External dep stubs ────────────────────────────────────────────────────────
global.formatMessage = (text) => text;   // loaded from CDN in index.html
global.showConfirm   = jest.fn().mockResolvedValue(true);
global.showAlert     = jest.fn().mockResolvedValue(undefined);

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED TEST HELPERS  (attached to global so every test file can use them)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * loadDom(filename)
 * Reads a real HTML file from client/ui/ and injects it into jsdom.
 * Uses the production file — no separate fixture to maintain.
 *
 * @param {string} filename  e.g. 'index.html' or 'dashboard.html'
 */
global.loadDom = function loadDom(filename) {
    const html = fs.readFileSync(path.join(UI_DIR, filename), 'utf8');
    document.documentElement.innerHTML = html;
};

/**
 * extractFn(code)
 * Evaluates a JS snippet and returns whatever it returns.
 * Used to pull pure functions out of index.js / dashboard.js for unit testing
 * without having to execute the entire file (which needs a full DOM + WS).
 *
 * Usage:
 *   const myFn = extractFn(`
 *       function myFn(x) { return x * 2; }
 *       return myFn;
 *   `);
 *
 * @param {string} code  Function body ending in `return <expr>;`
 * @returns {*}  Whatever the snippet returns
 */
global.extractFn = function extractFn(code) {
    // eslint-disable-next-line no-new-func
    return new Function(
        'document', 'localStorage', 'WebSocket', 'getComputedStyle',
        'requestAnimationFrame', 'cancelAnimationFrame',
        'IntersectionObserver', 'MutationObserver', 'ResizeObserver',
        'navigator', 'Intl', 'Chart', 'formatMessage',
        'showConfirm', 'showAlert',
        'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval',
        'console',
        code
    )(
        document, localStorage, WebSocket, getComputedStyle,
        requestAnimationFrame, cancelAnimationFrame,
        IntersectionObserver, MutationObserver, ResizeObserver,
        navigator, Intl, Chart, formatMessage,
        showConfirm, showAlert,
        setTimeout, clearTimeout, setInterval, clearInterval,
        console
    );
};


/**
 * requireUI(filename)
 * Requires a UI JS file so Jest instruments it for coverage.
 * The file's top-level startup code (WS connect, initCharts, etc.) is
 * safely guarded by the browser-globals check — in jsdom they exist but
 * are mocked, so execution completes without crashing.
 *
 * Returns the module.exports block appended to the file.
 *
 * @param {string} filename  e.g. 'index.js' or 'dashboard.js'
 * @returns {object}  Exported functions
 */
global.requireUI = function requireUI(filename) {
    // Clear the entire require cache for UI files on each call so the module
    // re-evaluates against the current jsdom document (loadDom must be called first).
    const filePath = path.join(UI_DIR, 'js', filename);

    // Also clear any transitive cached deps from the same directory
    Object.keys(require.cache).forEach(key => {
        if (key.startsWith(UI_DIR)) delete require.cache[key];
    });
    delete require.cache[filePath];

    return require(filePath);
};


// ── Debug helper: log whether key elements exist after loadDom ────────────────
// Remove this once addMessage tests pass
global._debugDom = function() {
    console.log('chat:', !!document.getElementById('chat'));
    console.log('themeDropdown:', !!document.getElementById('themeDropdown'));
    console.log('sendBtn:', !!document.getElementById('sendBtn'));
    console.log('input:', !!document.getElementById('input'));
};
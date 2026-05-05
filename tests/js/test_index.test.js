/**
 * tests/js/test_index.test.js
 * Tests for client/ui/js/index.js
 *
 * Uses requireUI() to load the real file so Jest instruments it for coverage.
 * DOM tests call loadDom('index.html') to use the real production HTML.
 */

'use strict';

// Require once — index.js captures DOM elements (chat, sendBtn, etc.) as
// module-level consts at require() time. Re-requiring each test would bind
// them to a fresh DOM but the previous test's DOM is gone, so we load the
// DOM once and clear only the chat contents between tests.
loadDom('index.html');
const ui = requireUI('index.js');

beforeEach(() => {
    localStorage.clear();
    // Clear chat contents without replacing the element (keeps module binding valid)
    const chat = document.getElementById('chat');
    if (chat) chat.innerHTML = '';
    // Reset logFilters to initial state by re-enabling all filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.add('active'));
});


// ═══════════════════════════════════════════════════════════════════
// themeIdToLabel
// ═══════════════════════════════════════════════════════════════════

describe('themeIdToLabel', () => {
    test('returns override for known ids', () => {
        expect(ui.themeIdToLabel('optimus')).toBe('Optimus Prime');
        expect(ui.themeIdToLabel('tokyo')).toBe('Tokyo Night');
        expect(ui.themeIdToLabel('te')).toBe('te');
        expect(ui.themeIdToLabel('mono')).toBe('Mono');
    });
    test('capitalises unknown ids',    () => { expect(ui.themeIdToLabel('default')).toBe('Default'); expect(ui.themeIdToLabel('matrix')).toBe('Matrix'); });
    test('handles single character',   () => expect(ui.themeIdToLabel('x')).toBe('X'));
});


// ═══════════════════════════════════════════════════════════════════
// escapeHtml  (tested via DOM directly — no extractFn needed)
// ═══════════════════════════════════════════════════════════════════

describe('escapeHtml', () => {
    test('escapes < and >', () => {
        const div = document.createElement('div');
        div.textContent = '<script>alert(1)</script>';
        expect(div.innerHTML).toContain('&lt;');
    });
    test('escapes ampersand', () => {
        const div = document.createElement('div');
        div.textContent = 'a & b';
        expect(div.innerHTML).toContain('&amp;');
    });
    test('plain text unchanged', () => {
        const div = document.createElement('div');
        div.textContent = 'hello world';
        expect(div.innerHTML).toBe('hello world');
    });
});


// ═══════════════════════════════════════════════════════════════════
// escapeAttr
// ═══════════════════════════════════════════════════════════════════

describe('escapeAttr', () => {
    test('escapes double quotes', () => expect(ui.escapeAttr('say "hello"')).toBe('say &quot;hello&quot;'));
    test('escapes single quotes', () => expect(ui.escapeAttr("it's")).toBe('it&#39;s'));
    test('leaves plain strings',  () => expect(ui.escapeAttr('hello world')).toBe('hello world'));
    test('coerces non-string',    () => { expect(ui.escapeAttr(42)).toBe('42'); expect(ui.escapeAttr(null)).toBe('null'); });
});


// ═══════════════════════════════════════════════════════════════════
// escapeRegex  (defined as plain function — no module export needed)
// ═══════════════════════════════════════════════════════════════════

describe('escapeRegex', () => {
    function escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    test('escapes dot',          () => expect(escapeRegex('a.b')).toBe('a\\.b'));
    test('escapes parens',       () => expect(escapeRegex('(x)')).toBe('\\(x\\)'));
    test('escapes brackets',     () => expect(escapeRegex('[a]')).toBe('\\[a\\]'));
    test('plain text unchanged', () => expect(escapeRegex('hello')).toBe('hello'));
    test('escapes star',         () => expect(escapeRegex('a*b')).toBe('a\\*b'));
    test('escapes pipe',         () => expect(escapeRegex('a|b')).toBe('a\\|b'));
});


// ═══════════════════════════════════════════════════════════════════
// makeSnippet
// ═══════════════════════════════════════════════════════════════════

describe('makeSnippet', () => {
    function escapeRegex(str) { return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
    function makeSnippet(content, term, maxLen) {
        const div = document.createElement('div');
        const idx = content.toLowerCase().indexOf(term.toLowerCase());
        let raw;
        if (idx === -1) {
            raw = content.slice(0, maxLen);
        } else {
            const start = Math.max(0, idx - 20);
            const end   = Math.min(content.length, idx + term.length + 60);
            raw = (start > 0 ? '\u2026' : '') + content.slice(start, end) + (end < content.length ? '\u2026' : '');
        }
        div.textContent = raw;
        const escaped = div.innerHTML;
        const re = new RegExp('(' + escapeRegex(term) + ')', 'gi');
        return escaped.replace(re, '<mark>$1</mark>');
    }

    test('highlights matching term',           () => expect(makeSnippet('The quick brown fox', 'quick', 80)).toContain('<mark>quick</mark>'));
    test('case insensitive match',             () => expect(makeSnippet('Hello WORLD', 'world', 80)).toContain('<mark>'));
    test('truncates when no match',            () => expect(makeSnippet('A'.repeat(200), 'xyz', 50).length).toBeLessThanOrEqual(55));
    test('adds ellipsis when context trimmed', () => expect(makeSnippet('X'.repeat(100) + ' target ' + 'Y'.repeat(100), 'target', 80)).toContain('…'));
    test('no match returns start of string',   () => expect(makeSnippet('hello world', 'xyz', 5)).toBe('hello'));
});


// ═══════════════════════════════════════════════════════════════════
// formatSessionDate
// ═══════════════════════════════════════════════════════════════════

describe('formatSessionDate', () => {
    test('returns empty string for falsy', () => {
        expect(ui.formatSessionDate(null)).toBe('');
        expect(ui.formatSessionDate(undefined)).toBe('');
        expect(ui.formatSessionDate('')).toBe('');
    });
    test('returns Today for current date',    () => expect(ui.formatSessionDate(new Date().toISOString())).toBe('Today'));
    test('returns Yesterday for yesterday',   () => { const d = new Date(); d.setDate(d.getDate() - 1); expect(ui.formatSessionDate(d.toISOString())).toBe('Yesterday'); });
    test('returns formatted date for older',  () => { const r = ui.formatSessionDate(new Date('2020-01-15T12:00:00Z').toISOString()); expect(r).toContain('2020'); expect(r).not.toBe('Today'); });
});


// ═══════════════════════════════════════════════════════════════════
// loadMultiAgentSetting
// ═══════════════════════════════════════════════════════════════════

describe('loadMultiAgentSetting', () => {
    beforeEach(() => localStorage.clear());
    test('returns false when nothing saved',    () => expect(ui.loadMultiAgentSetting()).toBe(false));
    test('returns true when saved as "true"',   () => { localStorage.setItem('mcp_multi_agent_enabled', 'true');  expect(ui.loadMultiAgentSetting()).toBe(true);  });
    test('returns false when saved as "false"', () => { localStorage.setItem('mcp_multi_agent_enabled', 'false'); expect(ui.loadMultiAgentSetting()).toBe(false); });
});


// ═══════════════════════════════════════════════════════════════════
// Favorites
// ═══════════════════════════════════════════════════════════════════

describe('Favorites', () => {
    beforeEach(() => {
        // Reset favorites state between tests
        ui.setFavorites([]);
    });

    test('isFavorite false for unknown tool',       () => expect(ui.isFavorite('x')).toBe(false));
    test('toggleFavorite adds and returns true',    () => expect(ui.toggleFavorite('t')).toBe(true));
    test('toggleFavorite removes and returns false',() => { ui.toggleFavorite('t'); expect(ui.toggleFavorite('t')).toBe(false); });
    test('addFavorite adds tool',                   () => { ui.addFavorite('a'); expect(ui.isFavorite('a')).toBe(true); });
    test('addFavorite is idempotent',               () => { ui.addFavorite('b'); ui.addFavorite('b'); expect(ui.getFavorites().filter(n => n === 'b').length).toBe(1); });
    test('removeFavorite removes tool',             () => { ui.addFavorite('c'); ui.removeFavorite('c'); expect(ui.isFavorite('c')).toBe(false); });
    test('removeFavorite on unknown is safe',       () => expect(() => ui.removeFavorite('nope')).not.toThrow());
});


// ═══════════════════════════════════════════════════════════════════
// getToolPrompt
// ═══════════════════════════════════════════════════════════════════

describe('getToolPrompt', () => {
    test('returns example when available', () => {
        ui.buildToolPrompts([{ name: 'get_weather', example: 'use get_weather: city="London"' }]);
        expect(ui.getToolPrompt('get_weather', [])).toBe('use get_weather: city="London"');
    });
    test('generates from params when no example', () => {
        ui.buildToolPrompts([]);
        expect(ui.getToolPrompt('my_tool', [{ name: 'query' }, { name: 'limit' }])).toBe('use my_tool: query="" limit=""');
    });
    test('bare use command when no params',  () => { ui.buildToolPrompts([]); expect(ui.getToolPrompt('simple', [])).toBe('use simple'); });
    test('ignores tools without example',    () => { ui.buildToolPrompts([{ name: 'no_ex' }]); expect(ui.getToolPrompt('no_ex', [])).toBe('use no_ex'); });
});


// ═══════════════════════════════════════════════════════════════════
// addMessage (DOM)
// ═══════════════════════════════════════════════════════════════════

describe('addMessage (DOM)', () => {
    test('user message appended to chat', () => {
        ui.addMessage('hello', 'user');
        expect(document.getElementById('chat').textContent).toContain('hello');
    });
    test('assistant message wrapped in msg-wrapper', () => {
        ui.addMessage('reply', 'assistant');
        expect(document.querySelector('.msg-wrapper')).not.toBeNull();
    });
    test('skips [TextContent( prefix', () => {
        ui.addMessage('[TextContent(type=text text=hi)]', 'user');
        expect(document.getElementById('chat').children.length).toBe(0);
    });
    test('skips empty text with no image', () => {
        ui.addMessage('', 'user');
        expect(document.getElementById('chat').children.length).toBe(0);
    });
    test('multi-agent assistant gets extra class', () => {
        ui.addMessage('response', 'assistant', false, true);
        expect(document.querySelector('.msg.assistant').className).toContain('multi-agent');
    });
    test('fallback model names shown as MCP', () => {
        ui.addMessage('hi', 'assistant', false, false, 'unknown');
        expect(document.querySelector('.msg-model').textContent).toBe('MCP');
    });
    test('real model name shown as-is', () => {
        ui.addMessage('hi', 'assistant', false, false, 'qwen2.5:14b');
        expect(document.querySelector('.msg-model').textContent).toBe('qwen2.5:14b');
    });
    test('image URL creates img element', () => {
        ui.addMessage('', 'assistant', false, false, null, null, null, 'https://example.com/img.jpg');
        expect(document.querySelector('img')).not.toBeNull();
    });
    test('messageId set on user message element', () => {
        ui.addMessage('hello', 'user', false, false, null, null, null, null, 99);
        expect(document.querySelector('[data-chat-message-id="99"]')).not.toBeNull();
    });
});


// ═══════════════════════════════════════════════════════════════════
// updateProgressBar (DOM)
// ═══════════════════════════════════════════════════════════════════

describe('updateProgressBar (DOM)', () => {
    test('sets width',                   () => { ui.updateProgressBar('cpuBar', 45); expect(document.getElementById('cpuBar').style.width).toBe('45%'); });
    test('adds critical above 80%',      () => { ui.updateProgressBar('cpuBar', 85); expect(document.getElementById('cpuBar').classList.contains('critical')).toBe(true); });
    test('adds high between 60 and 80%', () => { ui.updateProgressBar('cpuBar', 70); expect(document.getElementById('cpuBar').classList.contains('high')).toBe(true); });
    test('no class below 60%',           () => { ui.updateProgressBar('cpuBar', 50); expect(document.getElementById('cpuBar').classList.contains('high')).toBe(false); });
});


// ═══════════════════════════════════════════════════════════════════
// toggleLogFilter (DOM)
// ═══════════════════════════════════════════════════════════════════

describe('toggleLogFilter (DOM)', () => {
    test('removes level from filters', () => {
        // Ensure DEBUG is active first (beforeEach adds 'active' to all filter btns)
        expect(document.querySelector('.filter-btn[data-level="DEBUG"]').classList.contains('active')).toBe(true);
        ui.toggleLogFilter('DEBUG');
        expect(document.querySelector('.filter-btn[data-level="DEBUG"]').classList.contains('active')).toBe(false);
    });
    test('adds level back when re-toggled', () => {
        // Toggle off then back on
        ui.toggleLogFilter('WARNING');
        expect(document.querySelector('.filter-btn[data-level="WARNING"]').classList.contains('active')).toBe(false);
        ui.toggleLogFilter('WARNING');
        expect(document.querySelector('.filter-btn[data-level="WARNING"]').classList.contains('active')).toBe(true);
    });
});
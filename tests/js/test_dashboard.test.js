/**
 * tests/js/test_dashboard.test.js
 * Tests for client/ui/js/dashboard.js
 *
 * Uses requireUI() to load the real file so Jest instruments it for coverage.
 * DOM tests call loadDom('dashboard.html') to use the real production HTML.
 */

'use strict';

let dash;

beforeEach(() => {
    loadDom('dashboard.html');
    localStorage.clear();
    dash = requireUI('dashboard.js');
});


describe('formatTimestamp', () => {
    test('returns HH:MM:SS format', () => expect(dash.formatTimestamp(1704067200)).toMatch(/\d{2}:\d{2}:\d{2}/));
    test('returns a string',        () => expect(typeof dash.formatTimestamp(0)).toBe('string'));
});


describe('hexToRgba', () => {
    test('#3498db',  () => expect(dash.hexToRgba('#3498db', 0.1)).toBe('rgba(52,152,219,0.1)'));
    test('#ffffff',  () => expect(dash.hexToRgba('#ffffff', 1)).toBe('rgba(255,255,255,1)'));
    test('#000000',  () => expect(dash.hexToRgba('#000000', 0.5)).toBe('rgba(0,0,0,0.5)'));
    test('#e74c3c',  () => expect(dash.hexToRgba('#e74c3c', 1)).toBe('rgba(231,76,60,1)'));
});


describe('fmtSec', () => {
    test('dash for 0',         () => expect(dash.fmtSec(0)).toBe('—'));
    test('dash for null',      () => expect(dash.fmtSec(null)).toBe('—'));
    test('dash for undefined', () => expect(dash.fmtSec(undefined)).toBe('—'));
    test('1.5 → 1.50s',        () => expect(dash.fmtSec(1.5)).toBe('1.50s'));
    test('0.123 → 0.12s',      () => expect(dash.fmtSec(0.123)).toBe('0.12s'));
    test('120.999 → 121.00s',  () => expect(dash.fmtSec(120.999)).toBe('121.00s'));
});


describe('updateStatus (DOM)', () => {
    test('connected adds class and sets text', () => {
        dash.updateStatus(true);
        expect(document.getElementById('statusIndicator').classList.contains('connected')).toBe(true);
        expect(document.getElementById('statusText').textContent).toBe('Live metrics stream active');
    });
    test('disconnected removes class and sets text', () => {
        dash.updateStatus(true);
        dash.updateStatus(false);
        expect(document.getElementById('statusIndicator').classList.contains('connected')).toBe(false);
        expect(document.getElementById('statusText').textContent).toContain('Disconnected');
    });
});


describe('updatePercentileCards (DOM)', () => {
    test('sets agent percentile values', () => {
        dash.updatePercentileCards('agent', { p50: 1.2, p95: 3.4, p99: 5.6 });
        expect(document.getElementById('agentP50Value').textContent).toBe('1.20s');
        expect(document.getElementById('agentP95Value').textContent).toBe('3.40s');
        expect(document.getElementById('agentP99Value').textContent).toBe('5.60s');
    });
    test('sets llm percentile values',  () => { dash.updatePercentileCards('llm', { p50: 0.5, p95: 1.0, p99: 2.0 }); expect(document.getElementById('llmP50Value').textContent).toBe('0.50s'); });
    test('shows dash for zero values',  () => { dash.updatePercentileCards('agent', { p50: 0, p95: 0, p99: 0 }); expect(document.getElementById('agentP50Value').textContent).toBe('—'); });
});


describe('updateFailureKinds (DOM)', () => {
    test('shows no-data when empty',         () => { dash.updateFailureKinds({}); expect(document.getElementById('failureKindsContainer').textContent).toContain('No failures'); });
    test('shows no-data when all zero',      () => { dash.updateFailureKinds({ retryable: 0 }); expect(document.getElementById('failureKindsContainer').textContent).toContain('No failures'); });
    test('renders correct number of cards',  () => { dash.updateFailureKinds({ retryable: 5, internal_error: 2 }); expect(document.querySelectorAll('.failure-kind-card').length).toBe(2); });
    test('sorts by count descending',        () => { dash.updateFailureKinds({ retryable: 1, internal_error: 10 }); expect(document.querySelectorAll('.failure-kind-label')[0].textContent).toContain('Internal'); });
    test('shows correct percentage',         () => { dash.updateFailureKinds({ retryable: 1, internal_error: 1 }); expect(document.querySelectorAll('.failure-kind-pct')[0].textContent).toBe('50.0%'); });
    test('uses correct color for retryable', () => {
        dash.updateFailureKinds({ retryable: 3 });
        const color = document.querySelector('.failure-kind-value').style.color;
        expect(color === '#f1c40f' || color === 'rgb(241, 196, 15)').toBe(true);
    });
});


describe('renderHistogram (DOM)', () => {
    test('renders correct number of bars', () => {
        dash.renderHistogram('agentHistogram', { '<1s': 10, '1-3s': 5, '3-5s': 2 });
        expect(document.querySelectorAll('#agentHistogram .histogram-bar').length).toBe(3);
    });
    test('no-op when histogram is null', () => {
        document.getElementById('agentHistogram').innerHTML = 'kept';
        dash.renderHistogram('agentHistogram', null);
        expect(document.getElementById('agentHistogram').innerHTML).toBe('kept');
    });
    test('tallest bar gets 100% height', () => {
        dash.renderHistogram('agentHistogram', { '<1s': 100, '1-3s': 50 });
        const h = document.querySelectorAll('#agentHistogram .histogram-bar')[0].style.height;
        expect(parseFloat(h)).toBe(100);
    });
    test('shows bar counts',       () => { dash.renderHistogram('agentHistogram', { '<1s': 42, '1-3s': 0 }); expect(document.querySelectorAll('#agentHistogram .histogram-bar-count')[0].textContent).toBe('42'); });
    test('uses provided bucket order', () => { dash.renderHistogram('agentHistogram', { '<1s': 5, '1-3s': 3 }, ['<1s', '1-3s']); expect(document.querySelectorAll('#agentHistogram .histogram-bar-label')[0].textContent).toBe('<1s'); });
});


describe('buildMiniHistogram', () => {
    test('returns HTML string',          () => expect(typeof dash.buildMiniHistogram({ a: 1 })).toBe('string'));
    test('wraps in mini-histogram div',  () => expect(dash.buildMiniHistogram({ a: 1 })).toContain('mini-histogram'));
    test('correct number of bar groups', () => expect((dash.buildMiniHistogram({ a: 1, b: 2, c: 3 }).match(/mini-bar-group/g) || []).length).toBe(3));
    test('tallest bar is 100%',          () => expect(dash.buildMiniHistogram({ low: 10, high: 100 })).toContain('height:100.0%'));
});


describe('updateMetrics (DOM)', () => {
    const baseData = () => ({
        tools:          { total_calls: 42, per_tool: {} },
        llm:            { calls: 10, p50: 1.2, p95: 3.4, p99: 5.6, times: { timestamps: [], durations: [] }, histogram: { '<1s': 5 } },
        agent:          { runs: 5, error_rate: 2.5, p50: 0.8, p95: 2.1, p99: 4.0, times: { timestamps: [], durations: [] }, histogram: { '<1s': 3 } },
        overall_errors: 3,
        failure_kinds:  { retryable: 2, internal_error: 1 },
        buckets:        ['<1s', '1-3s', '3-5s', '>5s'],
        window_size:    1000,
    });

    test('updates tool call count',  () => { dash.updateMetrics(baseData()); expect(document.getElementById('toolCallsValue').textContent).toBe('42'); });
    test('updates LLM call count',   () => { dash.updateMetrics(baseData()); expect(document.getElementById('llmCallsValue').textContent).toBe('10'); });
    test('updates agent run count',  () => { dash.updateMetrics(baseData()); expect(document.getElementById('agentRunsValue').textContent).toBe('5'); });
    test('formats error rate',       () => { dash.updateMetrics(baseData()); expect(document.getElementById('errorRateValue').textContent).toBe('2.50%'); });
    test('renders failure cards',    () => { dash.updateMetrics(baseData()); expect(document.querySelectorAll('.failure-kind-card').length).toBeGreaterThan(0); });
    test('renders histograms',       () => { dash.updateMetrics(baseData()); expect(document.querySelector('#agentHistogram .histogram-bars')).not.toBeNull(); });
    test('no-data when no tools',    () => { const d = baseData(); d.tools.per_tool = {}; dash.updateMetrics(d); expect(document.getElementById('toolsContainer').textContent).toContain('No tool data'); });
    test('renders tool cards',       () => {
        const d = baseData();
        d.tools.per_tool = {
            search_tool: { calls: 5, errors: 0, avg_time: 0.5, p50: 0.4, p95: 0.9, p99: 1.2, histogram: {} },
            plex_tool:   { calls: 3, errors: 1, avg_time: 1.2, p50: 1.0, p95: 2.0, p99: 3.0, histogram: {} },
        };
        dash.updateMetrics(d);
        expect(document.querySelectorAll('.tool-card').length).toBe(2);
    });
});
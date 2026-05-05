#!/bin/bash
# run_tests.sh
# Runs both Python and JS tests with full reporting.
# Usage: ./run_tests.sh [--no-coverage] [--py-only] [--js-only]

set -e

NO_COVERAGE=false
PY_ONLY=false
JS_ONLY=false

for arg in "$@"; do
    case $arg in
        --no-coverage) NO_COVERAGE=true ;;
        --py-only)     PY_ONLY=true ;;
        --js-only)     JS_ONLY=true ;;
    esac
done

# ── Resolve Python — prefer active venv, then known locations ─────────────────
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -f "/home/myagi/.virtualenvs/mcp_a2a/bin/python" ]; then
    PYTHON="/home/myagi/.virtualenvs/mcp_a2a/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    echo "❌ No Python virtualenv found. Activate one first or create .venv"
    exit 1
fi

echo "Using Python: $PYTHON"
mkdir -p tests/results tests/js-results

# ── Python tests ──────────────────────────────────────────────────────────────
if [ "$JS_ONLY" = false ]; then
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Running Python tests..."
    echo "════════════════════════════════════════════════════════"

    if [ "$NO_COVERAGE" = true ]; then
        "$PYTHON" -m pytest --no-cov
    else
        "$PYTHON" -m pytest \
            --cov=client \
            --cov-report=xml:tests/results/coverage.xml \
            --cov-report=term-missing \
            --cov-fail-under=22
    fi
fi

# ── JS tests ──────────────────────────────────────────────────────────────────
if [ "$PY_ONLY" = false ]; then
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Running JavaScript tests..."
    echo "════════════════════════════════════════════════════════"

    if [ "$NO_COVERAGE" = true ]; then
        npx jest --no-coverage
    else
        npm test
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  All tests complete"
echo ""
echo "  Python reports:  tests/results/"
echo "    test-report.html     — test results"
echo "    coverage-report.html — coverage (lines hit)"
echo ""
echo "  JS reports:      tests/js-results/"
echo "    test-report.html     — test results"
echo "    coverage/lcov-report/index.html — coverage"
echo "════════════════════════════════════════════════════════"
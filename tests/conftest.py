"""
Shared pytest fixtures and configuration for MCP A2A test suite
"""
import asyncio
import os
import sys
import subprocess
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator, Dict, Any

# Add project root to path so we can import client, servers, etc.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables before any imports
os.environ["TESTING"] = "true"
os.environ["MAX_MESSAGE_HISTORY"] = "10"  # Smaller for tests
os.environ["LLM_BACKEND"] = "ollama"
os.environ["DISABLED_TOOLS"] = ""  # Enable all tools for testing

# Import after env setup
from client.session_manager import SessionManager
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ═══════════════════════════════════════════════════════════════════
# HOOKS: Pytest Configuration
# ═══════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Configure pytest environment"""
    # Create results directory (relative to where pytest runs from)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Set test markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests (>1 second)")
    config.addinivalue_line("markers", "requires_ollama: Requires Ollama server")
    config.addinivalue_line("markers", "requires_db: Requires database")


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Auto-mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

        # Auto-mark integration tests (tests with multiple fixtures)
        if len(item.fixturenames) > 3:
            if "integration" not in item.keywords:
                item.add_marker(pytest.mark.integration)


def pytest_sessionfinish(session, exitstatus):
    """Hook that runs after all tests complete - generate HTML from XML"""
    if exitstatus in [0, 1]:  # Success or test failures (but not collection errors)
        # Results directory is wherever pytest was run from + results/
        results_dir = Path("results").absolute()
        html_generator = results_dir / "generate_html.py"

        if html_generator.exists():
            print("\n" + "=" * 70)
            print("📊 Generating HTML reports from XML...")
            print(f"📁 Results directory: {results_dir}")
            print("=" * 70)

            try:
                # Run the HTML generator
                result = subprocess.run(
                    [sys.executable, str(html_generator)],
                    cwd=str(results_dir),
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    print(result.stdout)
                    print("=" * 70)
                    print("✅ HTML reports generated successfully!")
                    print(f"📁 Location: {results_dir}")
                    print(f"   🧪 test-report.html")
                    print(f"   📊 coverage-report.html")
                    print("=" * 70)
                else:
                    print(f"⚠️  HTML generation had issues:")
                    print(result.stdout)
                    if result.stderr:
                        print(result.stderr)
            except subprocess.TimeoutExpired:
                print("⚠️  HTML generation timed out")
            except Exception as e:
                print(f"⚠️  Could not generate HTML reports: {e}")
        else:
            print(f"\n⚠️  HTML generator not found at: {html_generator}")


# ═══════════════════════════════════════════════════════════════════
# FIXTURES: Test Directories
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def test_root():
    """Root directory of the project"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(test_root):
    """Test data directory"""
    data_dir = test_root / "tests" / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def protect_last_model_file(tmp_path):
    """
    Autouse fixture — patches MODEL_STATE_FILE for every test so no test
    can accidentally write to the real last_model.txt and change the active
    model in the running client.
    """
    fake_state_file = tmp_path / "last_model.txt"
    try:
        with patch("client.models.MODEL_STATE_FILE", str(fake_state_file)):
            yield fake_state_file
    except Exception:
        # models.py may not be importable in all test contexts — fail silently
        yield fake_state_file


# ═══════════════════════════════════════════════════════════════════
# FIXTURES: Database & Session Management
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_db(temp_dir):
    """Temporary SQLite database"""
    db_path = temp_dir / "test_sessions.db"
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def session_manager(temp_db):
    """SessionManager with temporary database"""
    sm = SessionManager(db_path=str(temp_db))
    yield sm
    # Cleanup
    if temp_db.exists():
        temp_db.unlink()


@pytest.fixture
def populated_session_manager(session_manager):
    """SessionManager with pre-populated test data"""
    # Create test sessions with messages
    session_id_1 = session_manager.create_session("Test Session 1")
    session_manager.add_message(session_id_1, "user", "Hello", max_history=30, model=None)
    session_manager.add_message(session_id_1, "assistant", "Hi there!", max_history=30, model="llama3.1:8b")
    # Use a path that's long enough (10+ chars after /mnt/c/)
    session_manager.add_message(session_id_1, "user", "Analyze /mnt/c/Users/Michael/project", max_history=30, model=None)
    session_manager.add_message(session_id_1, "assistant", "Project analyzed.", max_history=30, model="llama3.1:8b")

    session_id_2 = session_manager.create_session("Test Session 2")
    session_manager.add_message(session_id_2, "user", "What's the weather?", max_history=30, model=None)
    session_manager.add_message(session_id_2, "assistant", "Sunny, 22°C", max_history=30, model="llama3.1:8b")

    yield session_manager, session_id_1, session_id_2


# ═══════════════════════════════════════════════════════════════════
# FIXTURES: Mock LLM & Tools
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_llm():
    """Mock ChatOllama LLM"""
    llm = MagicMock()
    llm.model = "llama3.1:8b"
    llm.model_name = "llama3.1:8b"

    # Mock ainvoke to return AIMessage
    async def mock_ainvoke(messages):
        # Extract last user message
        user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_msg = msg.content
                break

        response_text = f"Mock response to: {user_msg[:50]}..."
        return AIMessage(content=response_text)

    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    llm.bind_tools = MagicMock(return_value=llm)

    return llm


@pytest.fixture
def mock_tool():
    """Mock MCP tool"""
    tool = MagicMock()
    tool.name = "test_tool"
    tool.description = "A test tool"

    async def mock_ainvoke(args):
        return {"result": "success", "args": args}

    tool.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    return tool


@pytest.fixture
def mock_tools(mock_tool):
    """List of mock tools"""
    tools = []
    for i in range(3):
        tool = MagicMock()
        tool.name = f"tool_{i}"
        tool.description = f"Test tool {i}"
        tool.ainvoke = AsyncMock(return_value={"result": f"tool_{i}_result"})
        tools.append(tool)
    return tools


# ═══════════════════════════════════════════════════════════════════
# FIXTURES: Conversation State
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def empty_conversation_state():
    """Empty conversation state"""
    return {
        "messages": [],
        "loop_count": 0
    }


@pytest.fixture
def conversation_state_with_history():
    """Conversation state with message history"""
    return {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi! How can I help?"),
            HumanMessage(content="What's 2+2?"),
            AIMessage(content="2+2 equals 4.")
        ],
        "loop_count": 0
    }


@pytest.fixture
def conversation_state_with_session(conversation_state_with_history):
    """Conversation state with session ID"""
    state = conversation_state_with_history.copy()
    state["session_id"] = 1
    return state


# ═══════════════════════════════════════════════════════════════════
# FIXTURES: Mock Ollama
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_ollama_list():
    """Mock ollama list command output"""
    return """NAME                    ID              SIZE      MODIFIED
llama3.1:8b            abc123          4.7 GB    2 days ago
qwen2.5:7b             def456          4.2 GB    5 days ago
mistral-nemo:latest    ghi789          7.2 GB    1 week ago"""


@pytest.fixture
def mock_ollama_running():
    """Mock Ollama server running"""
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}

        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        yield mock_client


@pytest.fixture
def mock_ollama_not_running():
    """Mock Ollama server not running"""
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        yield mock_client


# ═══════════════════════════════════════════════════════════════════
# FIXTURES: Mock GGUF Models
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_gguf_registry():
    """Mock GGUF model registry"""
    with patch("client.models.GGUFModelRegistry") as mock_registry:
        mock_registry.list_models.return_value = [
            "tinyllama-merged",
            "llama-3.2-1b-instruct-q4"
        ]
        mock_registry.get_model_info.return_value = {
            "path": "/fake/path/model.gguf",
            "size_mb": 1200,
            "alias": "tinyllama-merged"
        }
        yield mock_registry


# ═══════════════════════════════════════════════════════════════════
# FIXTURES: Mock WebSocket
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection"""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock()
    ws.close = AsyncMock()
    return ws


# ═══════════════════════════════════════════════════════════════════
# FIXTURES: Test Code Files
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_python_file(temp_dir):
    """Sample Python file for code review tests"""
    code = '''"""Sample module for testing"""
import os
import sys

class Calculator:
    """A simple calculator class"""
    
    def add(self, a, b):
        """Add two numbers"""
        return a + b
    
    def divide(self, a, b):
        """Divide two numbers"""
        return a / b  # BUG: No zero check!

def main():
    calc = Calculator()
    print(calc.add(2, 2))

if __name__ == "__main__":
    main()
'''

    file_path = temp_dir / "calculator.py"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_python_file_with_issues(temp_dir):
    """Python file with security and quality issues"""
    code = '''import os
import eval

# Security issues
PASSWORD = "hardcoded_password"
API_KEY = "sk-1234567890"

def run_code(user_input):
    # CRITICAL: Using eval
    result = eval(user_input)
    return result

def query_db(user_id):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return query

try:
    x = 1 / 0
except:  # Bare except
    pass
'''

    file_path = temp_dir / "vulnerable.py"
    file_path.write_text(code)
    return file_path


# ═══════════════════════════════════════════════════════════════════
# HELPERS: Assertions
# ═══════════════════════════════════════════════════════════════════

def assert_valid_session(session):
    """Assert session has required fields"""
    assert "id" in session
    assert "name" in session
    assert "created_at" in session
    assert "updated_at" in session


def assert_valid_message(message):
    """Assert message has required fields"""
    assert "role" in message
    assert message["role"] in ["user", "assistant", "system"]
    assert "text" in message
    assert isinstance(message["text"], str)


def assert_conversation_state_valid(state):
    """Assert conversation state is valid"""
    assert isinstance(state, dict)
    assert "messages" in state
    assert isinstance(state["messages"], list)


# Export helpers
pytest.assert_valid_session = assert_valid_session
pytest.assert_valid_message = assert_valid_message
pytest.assert_conversation_state_valid = assert_conversation_state_valid
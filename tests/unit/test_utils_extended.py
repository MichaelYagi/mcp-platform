"""
Extended tests for client/utils.py
Covers: open_browser_file, open_browser_url, get_public_ip,
        get_venv_python Windows path, venv-wsl fallback
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════
# get_public_ip
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGetPublicIp:
    def test_returns_ip_string_on_success(self):
        from client.utils import get_public_ip
        with patch("requests.get") as mock_get:
            mock_get.return_value.text = "1.2.3.4"
            result = get_public_ip()
        assert result == "1.2.3.4"

    def test_returns_none_on_failure(self):
        from client.utils import get_public_ip
        with patch("requests.get", side_effect=Exception("no network")):
            result = get_public_ip()
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# get_venv_python
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGetVenvPython:
    def test_finds_venv_wsl_fallback(self, temp_dir):
        from client.utils import get_venv_python
        # Create .venv-wsl/bin/python
        wsl_venv = temp_dir / ".venv-wsl" / "bin"
        wsl_venv.mkdir(parents=True)
        python = wsl_venv / "python"
        python.touch()
        with patch("platform.system", return_value="Linux"):
            result = get_venv_python(temp_dir)
        assert result == str(python)

    def test_raises_when_neither_venv_exists(self, temp_dir):
        from client.utils import get_venv_python
        with patch("platform.system", return_value="Linux"):
            with pytest.raises(FileNotFoundError, match="No valid Python"):
                get_venv_python(temp_dir)

    def test_windows_checks_scripts_path(self, temp_dir):
        from client.utils import get_venv_python
        scripts = temp_dir / ".venv" / "Scripts"
        scripts.mkdir(parents=True)
        python = scripts / "python.exe"
        python.touch()
        with patch("platform.system", return_value="Windows"):
            result = get_venv_python(temp_dir)
        assert result == str(python)


# ═══════════════════════════════════════════════════════════════════
# open_browser_file
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestOpenBrowserFile:
    def test_wsl_uses_cmd_exe(self, temp_dir):
        from client.utils import open_browser_file
        test_file = temp_dir / "test.html"
        test_file.touch()

        with patch("platform.uname") as mock_uname:
            mock_uname.return_value.release = "5.15.0-microsoft-standard-WSL2"
            with patch("subprocess.run") as mock_run:
                open_browser_file(test_file)
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert "cmd.exe" in args

    def test_non_wsl_uses_webbrowser(self, temp_dir):
        from client.utils import open_browser_file
        test_file = temp_dir / "test.html"
        test_file.touch()

        with patch("platform.uname") as mock_uname:
            mock_uname.return_value.release = "5.15.0-generic"
            with patch("webbrowser.open") as mock_wb:
                open_browser_file(test_file)
                mock_wb.assert_called_once()
                assert "file://" in mock_wb.call_args[0][0]

    def test_wsl_path_converted_to_windows(self, temp_dir):
        from client.utils import open_browser_file
        test_file = Path("/mnt/c/Users/test/file.html")

        with patch("platform.uname") as mock_uname:
            mock_uname.return_value.release = "microsoft-WSL2"
            with patch("subprocess.run") as mock_run:
                open_browser_file(test_file)
                call_args = str(mock_run.call_args)
                assert "C:" in call_args


# ═══════════════════════════════════════════════════════════════════
# open_browser_url
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestOpenBrowserUrl:
    def test_wsl_uses_cmd_exe(self):
        from client.utils import open_browser_url

        with patch("platform.uname") as mock_uname:
            mock_uname.return_value.release = "microsoft-WSL2"
            with patch("subprocess.run") as mock_run:
                open_browser_url("http://localhost:9000")
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert "cmd.exe" in args
                assert "http://localhost:9000" in args

    def test_non_wsl_uses_webbrowser(self):
        from client.utils import open_browser_url

        with patch("platform.uname") as mock_uname:
            mock_uname.return_value.release = "5.15.0-generic"
            with patch("webbrowser.open") as mock_wb:
                open_browser_url("http://example.com")
                mock_wb.assert_called_once_with("http://example.com")
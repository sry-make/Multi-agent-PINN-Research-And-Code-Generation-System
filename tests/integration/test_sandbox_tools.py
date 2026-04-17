from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from sandbox.docker_runner import DockerSandbox, ExecutionResult
from tools.code_tools import execute_python, run_shell


pytestmark = pytest.mark.integration


def test_run_shell_uses_docker_sandbox(project_root) -> None:
    mock_sandbox = MagicMock()
    mock_sandbox.run_command.return_value = ExecutionResult(
        success=True,
        stdout="/workspace/project",
        stderr="",
        exit_code=0,
    )

    with patch("sandbox.docker_runner.get_sandbox", return_value=mock_sandbox):
        shell_result = run_shell.invoke({"cmd": "pwd"})

    args, kwargs = mock_sandbox.run_command.call_args
    assert "/workspace/project" in shell_result
    assert args[0] == ["pwd"]
    assert kwargs.get("timeout") == 15
    assert kwargs.get("mount_dir") == project_root


def test_run_shell_supports_and_chaining() -> None:
    mock_sandbox = MagicMock()
    mock_sandbox.run_command.side_effect = [
        ExecutionResult(success=True, stdout="/workspace/project", stderr="", exit_code=0),
        ExecutionResult(success=True, stdout="total 3\nfoo.py", stderr="", exit_code=0),
    ]

    with patch("sandbox.docker_runner.get_sandbox", return_value=mock_sandbox):
        shell_result = run_shell.invoke({"cmd": "pwd && ls -la"})

    assert "$ pwd" in shell_result
    assert "$ ls -la" in shell_result
    assert mock_sandbox.run_command.call_count == 2


def test_execute_python_returns_host_artifacts() -> None:
    mock_sandbox = MagicMock()
    mock_sandbox.run_python.return_value = ExecutionResult(
        success=True,
        stdout="",
        stderr="",
        exit_code=0,
        artifacts=["/home/kyrie/PINN_AGENT_PROJECT_V2/outputs/sandbox_runs/run_demo/loss.png"],
    )

    with patch("sandbox.docker_runner.get_sandbox", return_value=mock_sandbox):
        result = execute_python.invoke({"code": "print('ok')", "timeout": 5})

    assert "[artifacts]" in result
    assert "loss.png" in result
    assert "（无输出）" not in result


def test_docker_sandbox_cleanup_error_does_not_hide_real_result() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._run_container = MagicMock(
        return_value=ExecutionResult(
            success=True,
            stdout="train ok",
            stderr="",
            exit_code=0,
        )
    )
    sandbox._export_runtime_artifacts = MagicMock(return_value=["/tmp/demo/train_log.txt"])

    runtime_root = tempfile.mkdtemp(prefix="sandbox_test_")
    with patch("sandbox.docker_runner.tempfile.mkdtemp", return_value=runtime_root):
        with patch("sandbox.docker_runner.shutil.rmtree", side_effect=PermissionError("cleanup denied")):
            result = sandbox.run_python("print('ok')")

    assert result.success is True
    assert result.artifacts == ["/tmp/demo/train_log.txt"]

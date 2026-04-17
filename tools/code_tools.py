"""
tools/code_tools.py — 代码执行与文件操作工具

工具列表:
    execute_python(code, timeout)  → Docker 沙盒执行 Python 代码
    read_file(path)                → 读取本地文件
    write_file(path, content)      → 写入本地文件
    run_shell(cmd)                 → 白名单 Shell 命令

所有工具均有安全限制，不会逃逸出沙盒或操作敏感路径。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from config import (
    SANDBOX_ARTIFACTS_DIR,
    SANDBOX_SHELL_WHITELIST,
    SANDBOX_TIMEOUT_SEC,
    ROOT_DIR,
)


# 允许文件操作的根目录（禁止访问项目外路径）
_ALLOWED_ROOT = ROOT_DIR


def _safe_path(path: str) -> Path:
    """解析路径并确认在允许范围内，防止路径穿越攻击。
    使用 Path.is_relative_to()（Python 3.9+），避免 startswith 前缀误判：
    例如 /project 不会错误放行 /project_evil。
    """
    resolved     = (_ALLOWED_ROOT / path).resolve()
    allowed_root = _ALLOWED_ROOT.resolve()
    if not resolved.is_relative_to(allowed_root):
        raise PermissionError(
            f"路径 '{path}' 超出允许范围 '{_ALLOWED_ROOT}'，操作被拒绝。"
        )
    return resolved


def _run_shell_tokens(tokens: list[str]) -> str:
    """执行单条已分词的白名单命令。"""
    from sandbox.docker_runner import get_sandbox

    if not tokens:
        return "空命令。"

    base_cmd = tokens[0]
    if base_cmd not in SANDBOX_SHELL_WHITELIST:
        allowed = ", ".join(sorted(SANDBOX_SHELL_WHITELIST))
        return (
            f"[拒绝] 命令 '{base_cmd}' 不在白名单内。\n"
            f"允许的命令: {allowed}"
        )

    try:
        sandbox = get_sandbox()
        result  = sandbox.run_command(
            tokens,
            timeout=15,
            mount_dir=_ALLOWED_ROOT,
        )
        if result.timed_out:
            return "[超时] 命令执行超过 15s，已终止。"

        output = (
            result.stdout
            + ("\n" if result.stdout and result.stderr else "")
            + result.stderr
        ).strip()
        if output:
            return output
        if not result.success:
            return f"[执行失败] exit_code={result.exit_code}"
        return "（无输出）"
    except RuntimeError as e:
        return f"[沙盒错误] {e}"
    except Exception as e:
        return f"[执行错误] {e}"


@tool
def execute_python(code: str, timeout: int = SANDBOX_TIMEOUT_SEC) -> str:
    """
    在隔离的 Docker 沙盒中执行 Python 代码，返回运行结果。

    安全保证:
    - 网络完全隔离（无法访问外部服务）
    - CPU / 内存资源限制
    - 执行超时强制终止
    - 非 root 用户运行

    适用场景: PINN 模型训练、数值求解、绘图、数据处理等。

    Args:
        code:    要执行的 Python 源码（支持多行，自动 dedent）
        timeout: 执行超时秒数（最大 60s，超出则截断）

    Returns:
        执行结果字符串，包含 stdout / stderr / 超时信息
    """
    from sandbox.docker_runner import get_sandbox

    timeout = min(timeout, 60)  # 硬上限

    try:
        sandbox = get_sandbox()
        result  = sandbox.run_python(
            code,
            timeout=timeout,
            artifact_root=SANDBOX_ARTIFACTS_DIR,
        )
    except RuntimeError as e:
        # Docker 未启动或镜像不存在
        return f"[沙盒错误] {e}"

    lines = []
    if result.timed_out:
        lines.append(f"[超时] 执行超过 {timeout}s，已强制终止。")
    if result.stdout:
        lines.append(f"[stdout]\n{result.stdout}")
    if result.stderr:
        lines.append(f"[stderr]\n{result.stderr}")
    if result.artifacts:
        lines.append("[artifacts]\n" + "\n".join(result.artifacts))
    if not result.stdout and not result.stderr and not result.artifacts:
        lines.append("（无输出）")

    status = "成功" if result.success else f"失败 (exit_code={result.exit_code})"
    lines.insert(0, f"[执行{status}]")

    return "\n".join(lines)


@tool
def read_file(path: str) -> str:
    """
    读取项目目录下的文本文件内容。

    Args:
        path: 相对于项目根目录的文件路径，如 "eval/results/report.md"

    Returns:
        文件内容字符串（超过 4000 字符时截断并提示）
    """
    try:
        target = _safe_path(path)
        if not target.exists():
            return f"文件不存在: {path}"
        if not target.is_file():
            return f"路径不是文件: {path}"
        content = target.read_text(encoding="utf-8", errors="replace")
    except PermissionError as e:
        return f"[权限拒绝] {e}"
    except Exception as e:
        return f"[读取错误] {e}"

    if len(content) > 4000:
        return content[:4000] + f"\n\n... [内容已截断，共 {len(content)} 字符]"
    return content


@tool
def write_file(path: str, content: str) -> str:
    """
    将内容写入项目目录下的文件（不存在则创建，已存在则覆盖）。

    Args:
        path:    相对于项目根目录的文件路径，如 "outputs/solution.py"
        content: 要写入的文本内容

    Returns:
        操作结果描述
    """
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已写入 {path}（{len(content)} 字符）"
    except PermissionError as e:
        return f"[权限拒绝] {e}"
    except Exception as e:
        return f"[写入错误] {e}"


@tool
def run_shell(cmd: str) -> str:
    """
    在 Docker 沙盒中执行白名单 Shell 命令。

    允许的命令: ls, cat, head, tail, pwd, mkdir, cp, mv, echo, python, pip
    命令不会通过宿主机 shell 解释，而是分词后直接传给容器进程执行。

    Args:
        cmd: Shell 命令字符串

    Returns:
        命令的 stdout + stderr 输出
    """
    import shlex

    if any(op in cmd for op in ("||", ";", "|")):
        return "[拒绝] run_shell 暂不支持 '||'、';' 或 '|', 请改为单条命令或使用 '&&' 串联。"

    segments = [segment.strip() for segment in cmd.split("&&") if segment.strip()]
    if not segments:
        return "空命令。"

    outputs: list[str] = []
    error_prefixes = (
        "[拒绝]",
        "[命令解析错误]",
        "[超时]",
        "[执行失败]",
        "[沙盒错误]",
        "[执行错误]",
    )

    for segment in segments:
        try:
            tokens = shlex.split(segment)
        except ValueError as e:
            result = f"[命令解析错误] {e}"
        else:
            result = _run_shell_tokens(tokens)

        if len(segments) == 1:
            return result

        outputs.append(f"$ {segment}\n{result}")
        if result.startswith(error_prefixes):
            break

    return "\n\n".join(outputs)

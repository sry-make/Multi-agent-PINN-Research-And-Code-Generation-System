"""
Docker 沙盒执行器 — Coder Agent 的代码执行后端

安全机制:
    - 网络完全隔离 (--network none)
    - CPU / 内存硬限制
    - 执行超时强制终止
    - 非 root 用户运行（sandbox_user）
    - 代码写入临时卷，不污染宿主机
"""

from __future__ import annotations

import shutil
import os
import tempfile
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import docker
from docker.errors import DockerException, ImageNotFound

from config import (
    SANDBOX_ARTIFACTS_DIR,
    SANDBOX_IMAGE,
    SANDBOX_CPU_LIMIT,
    SANDBOX_MEM_LIMIT,
    SANDBOX_TIMEOUT_SEC,
    SANDBOX_NETWORK,
)

_PROJECT_MOUNT_PATH = "/workspace/project"


@dataclass
class ExecutionResult:
    success: bool
    stdout:  str
    stderr:  str
    exit_code: int
    timed_out: bool = False
    artifacts: list[str] = field(default_factory=list)


class DockerSandbox:
    """Docker 沙盒执行器（复用 client 连接）"""

    def __init__(self):
        try:
            self._client = docker.from_env()
        except DockerException as e:
            raise RuntimeError(
                "❌ 无法连接 Docker 守护进程，请确认 Docker Desktop 已启动。\n"
                f"原因: {e}"
            )

    # ── 公共 API ──────────────────────────────────────────────

    def run_python(
        self,
        code: str,
        timeout: int = SANDBOX_TIMEOUT_SEC,
        artifact_root: str | Path | None = None,
    ) -> ExecutionResult:
        """
        在沙盒容器中执行 Python 代码。

        Args:
            code:    要执行的 Python 源码字符串
            timeout: 执行超时秒数，默认使用 config.SANDBOX_TIMEOUT_SEC

        Returns:
            ExecutionResult
        """
        artifact_root_path = Path(artifact_root or SANDBOX_ARTIFACTS_DIR)
        export_dir = artifact_root_path / self._next_run_id()

        # 为当前执行创建独立的临时工作区和 /tmp，执行后导出宿主机产物。
        runtime_root = Path(tempfile.mkdtemp(prefix="pinn_agent_run_"))
        try:
            workspace_dir = runtime_root / "workspace"
            container_tmp_dir = runtime_root / "tmp"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            container_tmp_dir.mkdir(parents=True, exist_ok=True)

            # 非 root 容器用户需要可写执行目录与 /tmp。
            os.chmod(workspace_dir, 0o777)
            os.chmod(container_tmp_dir, 0o777)

            code_path = workspace_dir / "solution.py"
            code_path.write_text(textwrap.dedent(code), encoding="utf-8")
            os.chmod(code_path, 0o644)

            result = self._run_container(
                command=["python", "/workspace/solution.py"],
                timeout=timeout,
                volumes={
                    str(workspace_dir): {"bind": "/workspace", "mode": "rw"},
                    str(container_tmp_dir): {"bind": "/tmp", "mode": "rw"},
                },
                working_dir="/workspace",
            )
            result.artifacts = self._export_runtime_artifacts(
                workspace_dir=workspace_dir,
                container_tmp_dir=container_tmp_dir,
                export_dir=export_dir,
            )
            return result
        finally:
            self._cleanup_runtime_root(runtime_root)

    def run_command(
        self,
        command: Sequence[str],
        timeout: int = SANDBOX_TIMEOUT_SEC,
        mount_dir: str | Path | None = None,
    ) -> ExecutionResult:
        """
        在沙盒容器中执行白名单命令。

        Args:
            command:   已分词的命令列表，例如 ["pwd"] 或 ["ls", "-la"]
            timeout:   执行超时秒数
            mount_dir: 可选宿主机目录；提供时会挂载到容器内 /workspace/project

        Returns:
            ExecutionResult
        """
        volumes: dict[str, dict[str, str]] = {}
        working_dir: str | None = None

        if mount_dir is not None:
            host_dir = Path(mount_dir).resolve()
            volumes[str(host_dir)] = {
                "bind": _PROJECT_MOUNT_PATH,
                "mode": "rw",
            }
            working_dir = _PROJECT_MOUNT_PATH

        return self._run_container(
            command=[str(part) for part in command],
            timeout=timeout,
            volumes=volumes,
            working_dir=working_dir,
        )

    # ── 内部工具 ──────────────────────────────────────────────

    def _run_container(
        self,
        command: list[str],
        timeout: int,
        volumes: dict[str, dict[str, str]] | None = None,
        working_dir: str | None = None,
    ) -> ExecutionResult:
        """统一的容器执行入口，供 Python 与 shell 命令复用。"""
        self._ensure_image()

        container = None
        timed_out = False
        try:
            container = self._client.containers.run(
                image=SANDBOX_IMAGE,
                command=command,
                volumes=volumes or None,
                working_dir=working_dir,
                network_mode=SANDBOX_NETWORK,
                cpu_period=100_000,
                cpu_quota=int(float(SANDBOX_CPU_LIMIT) * 100_000),
                mem_limit=SANDBOX_MEM_LIMIT,
                detach=True,
                remove=False,           # 手动删除，以便读取日志
                user=self._resolve_container_user(),
            )

            try:
                exit_code = container.wait(timeout=timeout)["StatusCode"]
            except Exception:
                timed_out = True
                exit_code = -1
                try:
                    container.kill()
                except Exception:
                    pass

            stdout, stderr = self._split_logs(container)
            return ExecutionResult(
                success=exit_code == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                timed_out=timed_out,
                artifacts=[],
            )
        finally:
            if container is not None:
                try:
                    container.remove(force=timed_out)
                except Exception:
                    pass

    def _next_run_id(self) -> str:
        """生成产物导出目录名。"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{ts}_{uuid4().hex[:8]}"

    def _resolve_container_user(self) -> str:
        """
        优先使用宿主机当前 UID/GID 运行容器进程，避免挂载目录文件落盘后
        因属主不一致导致清理失败；获取失败时回退到镜像内 sandbox_user。
        """
        try:
            return f"{os.getuid()}:{os.getgid()}"
        except Exception:
            return "sandbox_user"

    def _cleanup_runtime_root(self, runtime_root: Path) -> None:
        """
        尽力清理临时运行目录。

        这里绝不能把清理异常向上抛出，否则会覆盖真实的执行结果。
        """
        try:
            shutil.rmtree(runtime_root)
        except Exception:
            pass

    def _ensure_image(self) -> None:
        """检查沙盒镜像是否存在，不存在则提示构建"""
        try:
            self._client.images.get(SANDBOX_IMAGE)
        except ImageNotFound:
            raise RuntimeError(
                f"❌ 沙盒镜像 '{SANDBOX_IMAGE}' 不存在。\n"
                "请先执行: docker build -t pinn_agent_sandbox:latest "
                "-f sandbox/Dockerfile.sandbox ."
            )

    def _split_logs(self, container) -> tuple[str, str]:
        """分别获取 stdout 和 stderr"""
        stdout = container.logs(stdout=True,  stderr=False).decode("utf-8", errors="replace")
        stderr = container.logs(stdout=False, stderr=True ).decode("utf-8", errors="replace")
        return stdout.strip(), stderr.strip()

    def _export_runtime_artifacts(
        self,
        workspace_dir: Path,
        container_tmp_dir: Path,
        export_dir: Path,
    ) -> list[str]:
        """
        将运行期生成的文件从临时挂载目录导出到宿主机 outputs/ 下。

        导出范围:
        - /workspace 下除 solution.py 外的所有文件
        - /tmp 下的所有文件（保存到 export_dir/tmp/...）
        """
        collected: list[str] = []

        def copy_tree(src_root: Path, dest_root: Path, skip_names: set[str] | None = None):
            for path in src_root.rglob("*"):
                if not path.is_file():
                    continue
                if skip_names and path.name in skip_names:
                    continue
                rel_path = path.relative_to(src_root)
                dest = dest_root / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                collected.append(str(dest.resolve()))

        copy_tree(
            workspace_dir,
            export_dir,
            skip_names={"solution.py"},
        )
        copy_tree(container_tmp_dir, export_dir / "tmp")

        if not collected and export_dir.exists():
            shutil.rmtree(export_dir, ignore_errors=True)

        return sorted(set(collected))


# 全局单例（延迟初始化，避免启动时 Docker 未就绪报错）
_sandbox: DockerSandbox | None = None


def get_sandbox() -> DockerSandbox:
    global _sandbox
    if _sandbox is None:
        _sandbox = DockerSandbox()
    return _sandbox

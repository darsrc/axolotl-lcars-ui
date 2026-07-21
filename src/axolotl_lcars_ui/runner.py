"""Axolotl subprocess runner with log capture."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


CONFIG_ACTIONS = (
    "preprocess",
    "train",
    "inference",
    "merge-lora",
    "merge-sharded-fsdp-weights",
    "evaluate",
    "lm-eval",
    "quantize",
)

UTILITY_ACTIONS = (
    "fetch",
    "delinearize-llama4",
)

AXOLOTL_ACTIONS = (
    *CONFIG_ACTIONS,
    *UTILITY_ACTIONS,
)

LAUNCHER_ACTIONS = {"train", "preprocess", "evaluate"}


@dataclass
class RunState:
    command: list[str] = field(default_factory=list)
    status: str = "idle"
    returncode: int | None = None
    started_at: float | None = None
    ended_at: float | None = None


class AxolotlRunner:
    """Starts and monitors one Axolotl CLI process at a time."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.state = RunState()
        self.logs: deque[str] = deque(maxlen=2000)
        self.process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    @property
    def axolotl_path(self) -> str | None:
        return shutil.which("axolotl")

    def is_running(self) -> bool:
        process = self.process
        return process is not None and process.poll() is None

    def start(
        self,
        action: str,
        config_path: Path,
        *,
        launcher: str = "",
        cli_args: str = "",
        launcher_args: str = "",
    ) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError("An Axolotl process is already running.")
            binary = self.axolotl_path
            if binary is None:
                raise RuntimeError("The axolotl CLI was not found on PATH.")
            if action not in AXOLOTL_ACTIONS:
                raise RuntimeError(f"Unsupported Axolotl action: {action}")
            if launcher and action not in LAUNCHER_ACTIONS:
                raise RuntimeError(f"{action} does not use an Axolotl launcher. Leave launcher unset.")
            command = [binary, action]
            if action in CONFIG_ACTIONS:
                command.append(str(config_path))
            elif not cli_args.strip():
                if action == "fetch":
                    raise RuntimeError("fetch requires Axolotl Args such as examples or deepspeed_configs.")
                raise RuntimeError("delinearize-llama4 requires Axolotl Args such as --model and --output.")
            if cli_args.strip():
                command.extend(shlex.split(cli_args))
            if launcher:
                command.extend(["--launcher", launcher])
            if launcher_args.strip():
                if not launcher:
                    raise RuntimeError("Launcher args require a launcher selection.")
                command.append("--")
                command.extend(shlex.split(launcher_args))

            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self.process = process
            self.state = RunState(
                command=command,
                status="running",
                returncode=None,
                started_at=time.time(),
                ended_at=None,
            )
            self.logs.append(f"[AXOLOTL] started: {shlex.join(command)}")

        thread = threading.Thread(target=self._reader, args=(process,), daemon=True)
        thread.start()

    def stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        self.logs.append("[AXOLOTL] termination requested.")
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.logs.append("[AXOLOTL] graceful termination timed out; killing process.")
            process.kill()
            process.wait(timeout=10)
        with self._lock:
            self.state.status = "stopped"
            self.state.returncode = process.returncode
            self.state.ended_at = time.time()

    def drain_logs(self) -> list[str]:
        lines = list(self.logs)
        self.logs.clear()
        return lines

    def status_label(self) -> str:
        if self.is_running():
            return "RUNNING"
        if self.state.status == "failed":
            return "FAILED"
        if self.state.status == "complete":
            return "COMPLETE"
        if self.state.status == "stopped":
            return "STOPPED"
        return "IDLE"

    def status_severity(self) -> str:
        if self.is_running() or self.state.status == "complete":
            return "ok"
        if self.state.status in {"failed", "stopped"}:
            return "warn"
        return "ok"

    def _reader(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self.logs.append(line.rstrip("\n"))
        returncode = process.wait()
        with self._lock:
            self.state.returncode = returncode
            self.state.ended_at = time.time()
            self.state.status = "complete" if returncode == 0 else "failed"
        self.logs.append(f"[AXOLOTL] exited with code {returncode}.")

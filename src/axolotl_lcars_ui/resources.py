"""System resource telemetry for the LCARS dashboard."""

from __future__ import annotations

import shutil
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import psutil


def format_bytes(value: float | int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if amount < 1024 or unit == "PB":
            return f"{amount:.1f}{unit}" if unit != "B" else f"{amount:.0f}B"
        amount /= 1024
    return f"{amount:.1f}PB"


@dataclass
class GpuInfo:
    index: int
    name: str
    utilization: float
    memory_used: int
    memory_total: int
    temperature: float | None = None
    power_draw: float | None = None
    power_limit: float | None = None

    @property
    def memory_percent(self) -> float:
        if self.memory_total <= 0:
            return 0.0
        return min(100.0, self.memory_used / self.memory_total * 100)


@dataclass
class DiskInfo:
    mountpoint: str
    device: str
    fstype: str
    total: int
    used: int
    free: int
    percent: float


@dataclass
class ResourceSnapshot:
    cpu_percent: float
    ram_percent: float
    ram_used: int
    ram_total: int
    swap_percent: float
    disks: list[DiskInfo]
    gpus: list[GpuInfo]


@dataclass
class TelemetrySampler:
    history_size: int = 72
    cpu_history: deque[float] = field(default_factory=lambda: deque(maxlen=72))
    ram_history: deque[float] = field(default_factory=lambda: deque(maxlen=72))
    gpu_history: deque[float] = field(default_factory=lambda: deque(maxlen=72))
    latest: ResourceSnapshot | None = None

    def sample(self) -> ResourceSnapshot:
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        snapshot = ResourceSnapshot(
            cpu_percent=psutil.cpu_percent(interval=None),
            ram_percent=float(virtual.percent),
            ram_used=int(virtual.used),
            ram_total=int(virtual.total),
            swap_percent=float(swap.percent),
            disks=self._disk_info(),
            gpus=self._gpu_info(),
        )
        self.latest = snapshot
        self.cpu_history.append(snapshot.cpu_percent)
        self.ram_history.append(snapshot.ram_percent)
        self.gpu_history.append(max((gpu.utilization for gpu in snapshot.gpus), default=0.0))
        return snapshot

    def chart_payload(self) -> dict[str, list[float]]:
        return {
            "CPU": list(self.cpu_history),
            "RAM": list(self.ram_history),
            "GPU": list(self.gpu_history),
        }

    def _disk_info(self) -> list[DiskInfo]:
        disks: list[DiskInfo] = []
        seen: set[str] = set()
        for partition in psutil.disk_partitions(all=False):
            mount = partition.mountpoint
            if mount in seen:
                continue
            seen.add(mount)
            try:
                usage = psutil.disk_usage(mount)
            except OSError:
                continue
            if usage.total <= 0:
                continue
            disks.append(
                DiskInfo(
                    mountpoint=mount,
                    device=partition.device,
                    fstype=partition.fstype,
                    total=int(usage.total),
                    used=int(usage.used),
                    free=int(usage.free),
                    percent=float(usage.percent),
                )
            )
        disks.sort(key=lambda item: (item.mountpoint != "/", item.mountpoint))
        return disks[:8]

    def _gpu_info(self) -> list[GpuInfo]:
        if shutil.which("nvidia-smi") is None:
            return []
        fields = (
            "index,name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,power.limit"
        )
        command = [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if completed.returncode != 0:
            return []

        gpus: list[GpuInfo] = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                gpus.append(
                    GpuInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        utilization=float(parts[2] or 0),
                        memory_used=int(float(parts[3] or 0) * 1024 * 1024),
                        memory_total=int(float(parts[4] or 0) * 1024 * 1024),
                        temperature=_optional_float(parts[5] if len(parts) > 5 else ""),
                        power_draw=_optional_float(parts[6] if len(parts) > 6 else ""),
                        power_limit=_optional_float(parts[7] if len(parts) > 7 else ""),
                    )
                )
            except ValueError:
                continue
        return gpus


def disk_rows(disks: list[DiskInfo]) -> list[dict[str, str]]:
    return [
        {
            "Mount": disk.mountpoint,
            "Used": format_bytes(disk.used),
            "Free": format_bytes(disk.free),
            "Total": format_bytes(disk.total),
            "Use": f"{disk.percent:.1f}%",
            "Type": disk.fstype,
        }
        for disk in disks
    ]


def gpu_rows(gpus: list[GpuInfo]) -> list[dict[str, str]]:
    if not gpus:
        return [{"GPU": "none", "Load": "0%", "VRAM": "0B / 0B", "Temp": "", "Power": ""}]
    rows = []
    for gpu in gpus:
        power = ""
        if gpu.power_draw is not None:
            power = f"{gpu.power_draw:.0f}W"
            if gpu.power_limit:
                power = f"{power}/{gpu.power_limit:.0f}W"
        rows.append(
            {
                "GPU": f"{gpu.index}: {gpu.name}",
                "Load": f"{gpu.utilization:.0f}%",
                "VRAM": f"{format_bytes(gpu.memory_used)} / {format_bytes(gpu.memory_total)}",
                "Temp": "" if gpu.temperature is None else f"{gpu.temperature:.0f}C",
                "Power": power,
            }
        )
    return rows


def process_rows(limit: int = 10) -> list[dict[str, str]]:
    """Return top local processes by resident memory, with sanitized command names."""

    rows = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "status", "cmdline"]):
        try:
            info = proc.info
            memory = info.get("memory_info")
            rss = int(getattr(memory, "rss", 0) or 0)
            cpu = float(info.get("cpu_percent") or 0.0)
            rows.append(
                {
                    "PID": str(info.get("pid") or ""),
                    "Process": _safe_process_name(info.get("name"), info.get("cmdline")),
                    "RAM": format_bytes(rss),
                    "CPU": f"{cpu:.0f}%",
                    "State": str(info.get("status") or ""),
                    "_rss": str(rss),
                    "_cpu": str(cpu),
                }
            )
        except (OSError, psutil.Error):
            continue
    rows.sort(key=lambda row: (int(row.pop("_rss")), float(row.pop("_cpu"))), reverse=True)
    return rows[:limit] or [{"PID": "", "Process": "No process data available", "RAM": "", "CPU": "", "State": ""}]


def gpu_process_rows(limit: int = 10) -> list[dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        return [{"PID": "", "GPU": "nvidia-smi unavailable", "Process": "", "VRAM": ""}]
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return [{"PID": "", "GPU": "GPU process query failed", "Process": "", "VRAM": ""}]
    if completed.returncode != 0 or not completed.stdout.strip():
        return [{"PID": "", "GPU": "No active GPU compute processes", "Process": "", "VRAM": ""}]
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            used = int(float(parts[3]) * 1024 * 1024)
        except ValueError:
            used = 0
        rows.append(
            {
                "PID": parts[1],
                "GPU": parts[0][-12:],
                "Process": Path(parts[2]).name or parts[2],
                "VRAM": format_bytes(used),
            }
        )
    rows.sort(key=lambda row: _parse_size(row["VRAM"]), reverse=True)
    return rows[:limit] or [{"PID": "", "GPU": "No active GPU compute processes", "Process": "", "VRAM": ""}]


def storage_hotspot_rows(
    project_root: Path,
    *,
    output_dir: str = "",
    prepared_path: str = "",
) -> list[dict[str, str]]:
    candidates: list[tuple[str, Path, str]] = []
    if output_dir:
        candidates.append(("Axolotl output_dir", _resolve_user_path(project_root, output_dir), "training outputs"))
    if prepared_path:
        candidates.append(("Prepared dataset cache", _resolve_user_path(project_root, prepared_path), "dataset preprocessing"))
    candidates.extend(
        [
            ("Outputs root", project_root / "outputs", "training outputs"),
            ("Default prepared cache", project_root / "last_run_prepared", "dataset preprocessing"),
            ("Config files", project_root / "configs", "local yaml configs"),
        ]
    )
    hf_cache = _hf_cache_path()
    if hf_cache is not None:
        candidates.append(("Hugging Face cache", hf_cache, "downloaded models/datasets"))

    rows = []
    seen: set[Path] = set()
    for label, path, role in candidates:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        size = directory_size(resolved)
        rows.append(
            {
                "Area": label,
                "Size": format_bytes(size),
                "Role": role,
                "Path": _display_path(resolved, project_root),
                "_bytes": str(size),
            }
        )
    rows.sort(key=lambda row: int(row.pop("_bytes")), reverse=True)
    return rows or [{"Area": "No monitored artifact paths", "Size": "", "Role": "", "Path": ""}]


def _optional_float(value: str) -> float | None:
    value = value.strip()
    if value in {"", "N/A", "[Not Supported]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _safe_process_name(name: object, cmdline: object) -> str:
    if isinstance(name, str) and name:
        return name[:48]
    if isinstance(cmdline, list) and cmdline:
        return Path(str(cmdline[0])).name[:48]
    return "unknown"


def _parse_size(value: str) -> float:
    match = value.strip().upper()
    factors = {"TB": 1024**4, "GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
    for unit, factor in factors.items():
        if match.endswith(unit):
            try:
                return float(match[: -len(unit)]) * factor
            except ValueError:
                return 0.0
    return 0.0


def _resolve_user_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _hf_cache_path() -> Path | None:
    try:
        from huggingface_hub import constants
    except Exception:
        return None
    cache_home = getattr(constants, "HF_HUB_CACHE", None)
    return Path(cache_home).expanduser() if cache_home else None

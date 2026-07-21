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

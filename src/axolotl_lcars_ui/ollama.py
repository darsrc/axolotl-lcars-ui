"""Ollama model discovery and Axolotl compatibility heuristics."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class OllamaModel:
    name: str
    size: int
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    format: str = ""
    modified_at: str = ""
    from_ref: str = ""
    compatible: bool = False
    compatible_path: str = ""
    reason: str = ""
    hf_hint: str = ""
    hf_query: str = ""
    next_step: str = ""


@dataclass(frozen=True)
class OllamaChatResult:
    """One non-streaming response with the useful generation counters."""

    model: str
    content: str
    eval_count: int = 0
    eval_duration: int = 0
    total_duration: int = 0

    @property
    def tokens_per_second(self) -> float | None:
        duration = self.eval_duration or self.total_duration
        if self.eval_count <= 0 or duration <= 0:
            return None
        seconds = duration / 1_000_000_000
        return self.eval_count / seconds if seconds > 0 else None


class OllamaManager:
    """Uses the local API, plus the CLI for Modelfile-based adapter creation."""

    def __init__(self, host: str = "http://127.0.0.1:11434") -> None:
        self.host = host.rstrip("/")
        self.models: list[OllamaModel] = []
        self.last_error = ""
        self.selected: OllamaModel | None = None
        self.logs: deque[str] = deque(maxlen=500)

    def refresh(self) -> list[OllamaModel]:
        self.last_error = ""
        try:
            payload = self._json("GET", "/api/tags")
        except OSError as exc:
            self.last_error = f"Ollama is not reachable at {self.host}: {exc}"
            self.models = []
            return self.models

        models = []
        for item in payload.get("models", []):
            details = item.get("details") or {}
            model = OllamaModel(
                name=str(item.get("name") or item.get("model") or ""),
                size=int(item.get("size") or 0),
                family=str(details.get("family") or ""),
                parameter_size=str(details.get("parameter_size") or ""),
                quantization=str(details.get("quantization_level") or ""),
                format=str(details.get("format") or ""),
                modified_at=str(item.get("modified_at") or "")[:19],
            )
            self._enrich_show(model)
            models.append(model)
        self.models = models
        return models

    def select(self, name: str) -> OllamaModel | None:
        if not self.models:
            self.refresh()
        for model in self.models:
            if model.name == name or (
                ":" not in name and model.name == f"{name}:latest"
            ):
                self.selected = model
                return model
        self.selected = None
        return None

    def rows(self) -> list[dict[str, str]]:
        if not self.models and not self.last_error:
            self.refresh()
        if self.last_error:
            return [{"Model": "Ollama unavailable", "Params": "", "Quant": "", "Size": "", "HF Search": "", "Axolotl": self.last_error}]
        if not self.models:
            return [{"Model": "No local Ollama models", "Params": "", "Quant": "", "Size": "", "HF Search": "", "Axolotl": ""}]
        return [
            {
                "Model": model.name,
                "Params": model.parameter_size,
                "Quant": model.quantization,
                "Size": _format_bytes(model.size),
                "Source": model.hf_hint or model.from_ref[:56],
                "HF Search": model.hf_query,
                "Axolotl": "readable" if model.compatible else model.next_step,
            }
            for model in self.models
        ]

    def create_adapter_model(
        self,
        *,
        project_root: Path,
        model_name: str,
        base_model: str,
        adapter_path: str,
        system_prompt: str = "",
        timeout: float = 600.0,
    ) -> OllamaModel:
        """Create a local Ollama model from an Axolotl Safetensors adapter."""

        clean_name = model_name.strip()
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?",
            clean_name,
        ):
            raise RuntimeError(
                "Ollama model names may contain letters, numbers, dots, dashes, slashes, "
                "and one optional :tag."
            )
        if self.select(clean_name) is not None:
            raise RuntimeError(
                f"An Ollama model named {clean_name} already exists. Choose a new test-model name."
            )
        clean_base = base_model.strip()
        if not clean_base:
            raise RuntimeError("Choose the installed Ollama base model used for this adapter.")
        if not any(model.name == clean_base for model in self.models):
            self.refresh()
        if not any(model.name == clean_base for model in self.models):
            raise RuntimeError(f"The Ollama base model is not installed: {clean_base}")

        adapter = Path(adapter_path).expanduser()
        if not adapter.is_absolute():
            adapter = project_root / adapter
        adapter = adapter.resolve()
        if not adapter.is_dir():
            raise RuntimeError(f"Adapter directory does not exist: {adapter}")
        if not (adapter / "adapter_config.json").is_file():
            raise RuntimeError(f"Missing adapter_config.json in {adapter}")
        if not (adapter / "adapter_model.safetensors").is_file():
            raise RuntimeError(f"Missing adapter_model.safetensors in {adapter}")

        binary = shutil.which("ollama")
        if binary is None:
            raise RuntimeError("The ollama CLI is not on PATH.")
        if '"""' in system_prompt:
            raise RuntimeError('The system prompt cannot contain a triple quote sequence (""").')

        safe_directory = re.sub(r"[^A-Za-z0-9._-]+", "-", clean_name).strip("-")
        build_dir = project_root / ".lora-studio" / "ollama" / safe_directory
        build_dir.mkdir(parents=True, exist_ok=True)
        modelfile = build_dir / "Modelfile"
        lines = [
            f"FROM {clean_base}",
            f"ADAPTER {json.dumps(str(adapter))}",
        ]
        if system_prompt.strip():
            lines.append(f'SYSTEM """{system_prompt.strip()}"""')
        modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")

        command = [binary, "create", clean_name, "-f", str(modelfile)]
        self.logs.append(f"[OLLAMA] creating {clean_name} from {clean_base}")
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Ollama model creation failed: {exc}") from exc
        for line in [*completed.stdout.splitlines(), *completed.stderr.splitlines()]:
            if line.strip():
                self.logs.append(f"[OLLAMA] {line.strip()}")
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(detail or f"ollama create exited with code {completed.returncode}")

        self.refresh()
        created = self.select(clean_name)
        if created is None:
            raise RuntimeError(
                f"Ollama reported success, but {clean_name} was not found after refresh."
            )
        self.logs.append(f"[OLLAMA] model ready: {clean_name}")
        return created

    def chat(
        self,
        model_name: str,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        timeout: float = 300.0,
    ) -> OllamaChatResult:
        """Run one non-streaming local Ollama chat turn."""

        clean_model = model_name.strip()
        clean_prompt = prompt.strip()
        if not clean_model:
            raise RuntimeError("Choose an Ollama model.")
        if not clean_prompt:
            raise RuntimeError("Enter a test prompt.")
        messages: list[dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": clean_prompt})
        self.logs.append(f"[OLLAMA] testing {clean_model}")
        try:
            payload = self._json(
                "POST",
                "/api/chat",
                {
                    "model": clean_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": max(0.0, min(2.0, float(temperature)))},
                },
                timeout=timeout,
            )
        except OSError as exc:
            raise RuntimeError(f"Ollama test failed: {exc}") from exc
        message = payload.get("message") or {}
        if not isinstance(message, dict):
            raise RuntimeError("Ollama returned an unexpected chat response.")
        content = str(message.get("content") or "")
        result = OllamaChatResult(
            model=clean_model,
            content=content,
            eval_count=int(payload.get("eval_count") or 0),
            eval_duration=int(payload.get("eval_duration") or 0),
            total_duration=int(payload.get("total_duration") or 0),
        )
        self.logs.append(
            f"[OLLAMA] response complete: {result.eval_count} generated token(s)"
        )
        return result

    def drain_logs(self) -> list[str]:
        lines = list(self.logs)
        self.logs.clear()
        return lines

    def _enrich_show(self, model: OllamaModel) -> None:
        try:
            payload = self._json("POST", "/api/show", {"model": model.name})
        except OSError as exc:
            model.reason = f"show failed: {exc}"
            model.hf_query = _hf_search_query(model)
            _set_next_step(model)
            return
        modelfile = str(payload.get("modelfile") or "")
        model.from_ref = _from_line(modelfile)
        model.hf_hint = _hf_reference(model.from_ref)
        model.hf_query = _hf_search_query(model)
        info = payload.get("model_info") or {}
        if not model.format:
            model.format = str(info.get("general.file_type") or "")
        _mark_compatibility(model)
        _set_next_step(model)

    def _json(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        timeout: float = 2.0,
    ) -> dict[str, object]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except (OSError, UnicodeError):
                detail = ""
            raise OSError(detail or str(exc)) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OSError(exc) from exc
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise OSError("invalid JSON returned by Ollama") from exc
        if not isinstance(parsed, dict):
            raise OSError("unexpected Ollama response")
        return parsed


def _from_line(modelfile: str) -> str:
    for line in modelfile.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            return stripped[5:].strip()
    return ""


def _mark_compatibility(model: OllamaModel) -> None:
    source = model.from_ref.strip().strip('"')
    if source:
        path = Path(source).expanduser()
        exists = _path_exists(path)
        if exists and _is_transformers_model_dir(path):
            model.compatible = True
            model.compatible_path = str(path)
            model.reason = "local HF/safetensors directory"
            return
        if exists and path.suffix.lower() == ".gguf":
            model.reason = "GGUF is runnable in Ollama, not trainable by Axolotl"
            return
        if source.startswith("/usr/share/ollama/") or "/.ollama/models/blobs/" in source:
            model.reason = "Ollama internal blob path is not an Axolotl model directory"
            return
    if model.format.lower() == "gguf" or model.quantization:
        model.reason = "Ollama quantized/GGUF store is not an Axolotl base_model path"
        return
    model.reason = "No local Transformers/safetensors source path exposed"


def _set_next_step(model: OllamaModel) -> None:
    if model.compatible:
        model.next_step = "Use compatible local source"
        return
    if model.hf_hint:
        model.next_step = "Search HF for source/fine-tunes; avoid GGUF-only files"
        return
    if model.hf_query:
        model.next_step = "Search HF for matching Transformers/safetensors repo"
        return
    model.next_step = model.reason or "No Axolotl-readable source detected"


def _hf_reference(source: str) -> str:
    clean = source.strip().strip('"')
    if not clean:
        return ""
    if clean.startswith("hf.co/"):
        clean = "https://" + clean
    if clean.startswith("huggingface.co/"):
        clean = "https://" + clean
    if clean.startswith(("http://", "https://")):
        parsed = urlparse(clean)
        if parsed.netloc not in {"hf.co", "huggingface.co", "www.huggingface.co"}:
            return ""
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return "/".join(parts[:2]).split(":", 1)[0]
        return ""
    match = re.match(r"^([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)(?::[^/\s]+)?$", clean)
    return match.group(1) if match else ""


def _hf_search_query(model: OllamaModel) -> str:
    if model.hf_hint:
        repo = model.hf_hint
        name = repo.split("/", 1)[1]
        return _strip_runtime_quant_terms(name) or repo
    name = model.name.split(":", 1)[0]
    pieces = [name, model.family, model.parameter_size]
    query = " ".join(piece for piece in pieces if piece).strip()
    return _strip_runtime_quant_terms(query)


def _strip_runtime_quant_terms(value: str) -> str:
    text = re.sub(r"(?i)\bq[2-8](?:[-_\s][a-z0-9]+)*\b", " ", value)
    text = re.sub(r"(?i)\b(gguf|exl2|gptq|awq)\b", " ", text)
    text = re.sub(r"(?i)\b(4bit|8bit|int4|int8|quantized|ollama)\b", " ", text)
    text = re.sub(r"[-_]+", " ", text)
    return " ".join(text.split())


def _is_transformers_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    has_tokenizer = any((path / name).exists() for name in ("tokenizer.json", "tokenizer.model"))
    return has_config and has_weights and has_tokenizer


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f}{unit}" if unit != "B" else f"{amount:.0f}B"
        amount /= 1024
    return f"{amount:.1f}TB"

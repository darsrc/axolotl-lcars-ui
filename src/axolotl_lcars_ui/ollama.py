"""Ollama model discovery and Axolotl compatibility heuristics."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
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


class OllamaManager:
    """Reads the local Ollama API without taking a dependency on the ollama CLI."""

    def __init__(self, host: str = "http://127.0.0.1:11434") -> None:
        self.host = host.rstrip("/")
        self.models: list[OllamaModel] = []
        self.last_error = ""
        self.selected: OllamaModel | None = None

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
            if model.name == name:
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

    def _json(self, method: str, path: str, body: dict[str, object] | None = None) -> dict[str, object]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=2.0) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OSError(exc) from exc
        parsed = json.loads(raw or "{}")
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

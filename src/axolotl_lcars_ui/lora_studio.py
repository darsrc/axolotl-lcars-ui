"""Beginner-facing LoRA project, dataset, and artifact helpers."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


class LoraStudioError(RuntimeError):
    """Raised when a guided LoRA operation cannot be completed safely."""


LORA_GOALS = (
    "Personality / voice",
    "Agent behavior",
    "Personality + agent",
)

LORA_MEMORY_PROFILES = (
    "Balanced LoRA · easiest Ollama path",
    "QLoRA · lowest VRAM",
)

LORA_BASE_MODELS: tuple[tuple[str, str], ...] = (
    (
        "Llama 3.2 1B · smallest starter",
        "unsloth/Llama-3.2-1B-Instruct",
    ),
    (
        "Llama 3.2 3B Instruct · stronger",
        "meta-llama/Llama-3.2-3B-Instruct",
    ),
    (
        "Gemma 2 2B Instruct · compact",
        "google/gemma-2-2b-it",
    ),
    (
        "Mistral 7B Instruct · larger",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ),
)

OLLAMA_BASE_HINTS = {
    "unsloth/Llama-3.2-1B-Instruct": "llama3.2:1b",
    "NousResearch/Llama-3.2-1B": "llama3.2:1b",
    "meta-llama/Llama-3.2-1B-Instruct": "llama3.2:1b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2:3b",
    "google/gemma-2-2b-it": "gemma2:2b",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral:7b",
}

_PROJECT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
_DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}\.jsonl$")
_PLACEHOLDER_MARKERS = ("[edit me", "<edit me", "replace this", "todo:")


@dataclass(frozen=True)
class DatasetReport:
    """Plain-language result of inspecting one configured dataset."""

    source: str
    source_kind: str
    dataset_type: str
    example_count: int | None = None
    message_count: int = 0
    placeholder_count: int = 0
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        if self.source_kind == "hub":
            return not self.errors
        return (
            not self.errors
            and self.example_count is not None
            and self.example_count > 0
            and self.placeholder_count == 0
        )

    @property
    def status(self) -> str:
        if self.errors:
            return "NEEDS ATTENTION"
        if self.source_kind == "hub":
            return "CHECKED AT TRAIN TIME"
        if self.placeholder_count:
            return "DRAFT"
        if self.ready:
            return "READY"
        return "NOT CONFIGURED"


@dataclass(frozen=True)
class AdapterArtifact:
    """A PEFT adapter directory produced by Axolotl."""

    path: Path
    size_bytes: int
    modified_at: float
    base_model: str = ""

    @property
    def modified_text(self) -> str:
        return datetime.fromtimestamp(self.modified_at).strftime("%Y-%m-%d %H:%M:%S")


def normalize_project_name(value: str) -> str:
    """Return a safe project slug, rejecting surprising filesystem names."""

    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not _PROJECT_NAME_PATTERN.fullmatch(slug):
        raise LoraStudioError(
            "Use 2–63 letters, numbers, or hyphens, starting with a letter or number."
        )
    return slug


def beginner_config_updates(
    project_name: str,
    *,
    base_model: str,
    memory_profile: str,
) -> dict[str, Any]:
    """Translate the short setup wizard into conservative Axolotl settings."""

    slug = normalize_project_name(project_name)
    if memory_profile not in LORA_MEMORY_PROFILES:
        raise LoraStudioError("Choose one of the guided memory profiles.")
    qlora = memory_profile == LORA_MEMORY_PROFILES[1]
    return {
        "base_model": base_model.strip(),
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "adapter": "qlora" if qlora else "lora",
        "load_in_8bit": not qlora,
        "load_in_4bit": qlora,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "datasets.0.path": f"./data/{slug}.jsonl",
        "datasets.0.type": "chat_template",
        "datasets.0.ds_type": "json",
        "datasets.0.field_messages": "messages",
        "datasets.0.chat_template": "tokenizer_default",
        "datasets.0.train_on_eos": "turn",
        "dataset_prepared_path": f"./prepared/{slug}",
        "val_set_size": 0.1,
        "sequence_len": 2048,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        "output_dir": f"./outputs/{slug}",
        "save_safetensors": True,
        "micro_batch_size": 1 if qlora else 2,
        "gradient_accumulation_steps": 8 if qlora else 4,
        "num_epochs": 3,
        "learning_rate": 0.0002 if qlora else 0.0001,
        "optimizer": "paged_adamw_8bit" if qlora else "adamw_bnb_8bit",
        "lr_scheduler": "cosine",
        "warmup_steps": 10,
        "bf16": "auto",
        "fp16": False,
        "gradient_checkpointing": "true",
        "attn_implementation": "sdpa",
        "logging_steps": 1,
        "save_steps": 100,
        "eval_steps": 100,
        "strict": False,
    }


def starter_dataset_template(goal: str, assistant_name: str) -> str:
    """Create an intentionally unfinished JSONL teaching template."""

    name = assistant_name.strip() or "the assistant"
    system = (
        f"[EDIT ME: describe {name}'s voice, values, boundaries, and response style in one "
        "short paragraph.]"
    )
    if goal == LORA_GOALS[0]:
        prompts = (
            "Introduce yourself to a new user.",
            "Explain a difficult idea in your characteristic voice.",
            "Disagree politely when the user is mistaken.",
            "Respond when you do not know the answer.",
            "Help a frustrated user recover from a mistake.",
            "End a successful conversation naturally.",
        )
    elif goal == LORA_GOALS[1]:
        prompts = (
            "A request is ambiguous. What do you do first?",
            "Plan a task that requires several tools.",
            "A tool returns an error. How do you recover?",
            "A requested action could be destructive.",
            "Report progress during a long-running task.",
            "Summarize what was completed and what remains.",
        )
    else:
        prompts = (
            "Introduce yourself and explain how you help.",
            "Plan a multi-step task in your characteristic voice.",
            "Ask one useful question when a request is ambiguous.",
            "Recover gracefully after a tool fails.",
            "Set a warm but firm safety boundary.",
            "Report a completed task concisely.",
        )

    records = []
    for index, prompt in enumerate(prompts, start=1):
        records.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": (
                            f"[EDIT ME {index}: write the exact ideal answer you want the model "
                            "to imitate. Do not describe the answer—write it.]"
                        ),
                    },
                ]
            }
        )
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)


def inspect_configured_dataset(project_root: Path, cfg: dict[str, Any]) -> DatasetReport:
    """Inspect the first Axolotl dataset without downloading remote content."""

    datasets = cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets or not isinstance(datasets[0], dict):
        return DatasetReport(
            source="",
            source_kind="missing",
            dataset_type="",
            errors=("Choose or create a dataset first.",),
        )
    dataset = datasets[0]
    source = str(dataset.get("path") or "").strip()
    dataset_type = str(dataset.get("type") or "").strip()
    if not source:
        return DatasetReport(
            source="",
            source_kind="missing",
            dataset_type=dataset_type,
            errors=("The configured dataset path is empty.",),
        )
    if not _looks_local_dataset(source):
        return DatasetReport(
            source=source,
            source_kind="hub",
            dataset_type=dataset_type,
            warnings=(
                "Hub dataset contents are loaded and validated by Axolotl during preprocessing.",
            ),
        )

    path = _resolve_project_path(project_root, source)
    if not path.exists():
        return DatasetReport(
            source=str(path),
            source_kind="local",
            dataset_type=dataset_type,
            errors=("The local dataset file does not exist yet.",),
        )
    if not path.is_file():
        return DatasetReport(
            source=str(path),
            source_kind="local",
            dataset_type=dataset_type,
            errors=("The guided data page expects one JSON or JSONL file.",),
        )
    if path.suffix.lower() == ".jsonl":
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return DatasetReport(
                source=str(path),
                source_kind="local",
                dataset_type=dataset_type,
                errors=(f"Could not read the dataset: {exc}",),
            )
        return inspect_jsonl_text(
            text,
            source=str(path),
            dataset_type=dataset_type,
            messages_field=str(dataset.get("field_messages") or "messages"),
            completion_field=str(dataset.get("field") or "text"),
        )
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return DatasetReport(
                source=str(path),
                source_kind="local",
                dataset_type=dataset_type,
                errors=(f"Could not parse the JSON dataset: {exc}",),
            )
        if not isinstance(payload, list):
            return DatasetReport(
                source=str(path),
                source_kind="local",
                dataset_type=dataset_type,
                errors=("A guided JSON dataset must contain a top-level list of examples.",),
            )
        return _inspect_records(
            payload,
            source=str(path),
            dataset_type=dataset_type,
            messages_field=str(dataset.get("field_messages") or "messages"),
            completion_field=str(dataset.get("field") or "text"),
        )
    return DatasetReport(
        source=str(path),
        source_kind="local",
        dataset_type=dataset_type,
        example_count=None,
        warnings=(
            f"{path.suffix or 'This file type'} is left to Axolotl's preprocessing validator.",
        ),
    )


def inspect_jsonl_text(
    text: str,
    *,
    source: str = "editor",
    dataset_type: str = "chat_template",
    messages_field: str = "messages",
    completion_field: str = "text",
) -> DatasetReport:
    """Parse JSONL with line-specific errors and inspect its training shape."""

    records: list[Any] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"Line {line_number}: invalid JSON ({exc.msg}).")
            if len(errors) >= 20:
                errors.append("More JSON errors were omitted.")
                break
    if errors:
        return DatasetReport(
            source=source,
            source_kind="local",
            dataset_type=dataset_type,
            example_count=len(records),
            errors=tuple(errors),
        )
    return _inspect_records(
        records,
        source=source,
        dataset_type=dataset_type,
        messages_field=messages_field,
        completion_field=completion_field,
    )


def save_chat_jsonl(project_root: Path, filename: str, text: str) -> tuple[Path, DatasetReport]:
    """Validate and save a guided dataset, preserving one recoverable prior draft."""

    clean_name = Path(filename.strip()).name
    if clean_name != filename.strip() or not _DATASET_NAME_PATTERN.fullmatch(clean_name):
        raise LoraStudioError(
            "Use a simple .jsonl filename containing letters, numbers, dots, dashes, or underscores."
        )
    report = inspect_jsonl_text(text, source=f"./data/{clean_name}")
    if report.errors:
        raise LoraStudioError(report.errors[0])
    if not report.example_count:
        raise LoraStudioError("Add at least one JSONL training example before saving.")

    data_dir = (project_root / "data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    target = (data_dir / clean_name).resolve()
    if target.parent != data_dir:
        raise LoraStudioError("The dataset must stay inside this project's data directory.")
    if target.exists():
        backup = target.with_suffix(target.suffix + ".bak")
        shutil.copy2(target, backup)
    target.write_text(text.rstrip() + "\n", encoding="utf-8")
    return target, report


def discover_adapter_artifacts(
    project_root: Path,
    cfg: dict[str, Any],
    *,
    limit: int = 25,
) -> list[AdapterArtifact]:
    """Find recent PEFT adapter directories under the configured output path."""

    raw_output = str(cfg.get("output_dir") or "").strip()
    if not raw_output:
        return []
    output_dir = _resolve_project_path(project_root, raw_output)
    if not output_dir.exists() or not output_dir.is_dir():
        return []
    if output_dir == Path(output_dir.anchor):
        return []

    artifacts: list[AdapterArtifact] = []
    config_files: list[Path] = []
    seen: set[Path] = set()
    for pattern in (
        "adapter_config.json",
        "checkpoint-*/adapter_config.json",
        "*/adapter_config.json",
        "*/*/adapter_config.json",
    ):
        for config_path in output_dir.glob(pattern):
            resolved = config_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            config_files.append(config_path)
            if len(config_files) >= 200:
                break
        if len(config_files) >= 200:
            break
    for config_path in config_files:
        directory = config_path.parent
        weight_files = [
            path
            for path in (directory / "adapter_model.safetensors",)
            if path.is_file()
        ]
        if not weight_files:
            continue
        files = [config_path, *weight_files]
        stats = [stat for path in files if (stat := _safe_stat(path)) is not None]
        size = sum(stat.st_size for stat in stats)
        modified = max((stat.st_mtime for stat in stats), default=0.0)
        base_model = ""
        try:
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                base_model = str(metadata.get("base_model_name_or_path") or "")
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        artifacts.append(
            AdapterArtifact(
                path=directory.resolve(),
                size_bytes=size,
                modified_at=modified,
                base_model=base_model,
            )
        )
    artifacts.sort(key=lambda item: item.modified_at, reverse=True)
    return artifacts[:limit]


def suggested_ollama_model(base_model: str, available_names: Iterable[str]) -> str:
    """Pick an installed Ollama model matching the training base when possible."""

    names = [str(name) for name in available_names if str(name)]
    hint = OLLAMA_BASE_HINTS.get(base_model.strip(), "")
    if hint:
        for name in names:
            if name == hint:
                return name
            if ":" in hint and name.startswith(hint + "-"):
                return name
        # A merely similar family member is not safe for a known adapter base.
        return ""
    lowered = base_model.lower()
    family_hints = (
        ("llama", "llama"),
        ("gemma", "gemma"),
        ("mistral", "mistral"),
    )
    for needle, family in family_hints:
        if needle not in lowered:
            continue
        for name in names:
            if family in name.lower():
                return name
    return names[0] if names else ""


def _inspect_records(
    records: list[Any],
    *,
    source: str,
    dataset_type: str,
    messages_field: str,
    completion_field: str,
) -> DatasetReport:
    errors: list[str] = []
    warnings: list[str] = []
    message_count = 0
    placeholder_count = 0
    normalized_type = dataset_type or "chat_template"

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            errors.append(f"Example {index}: expected a JSON object.")
            continue
        strings: list[str] = []
        if normalized_type in {"chat_template", "sharegpt"}:
            messages = record.get(messages_field)
            if not isinstance(messages, list) or not messages:
                errors.append(f"Example {index}: missing a non-empty '{messages_field}' list.")
                continue
            for message_index, message in enumerate(messages, start=1):
                if not isinstance(message, dict):
                    errors.append(
                        f"Example {index}, message {message_index}: expected an object."
                    )
                    continue
                role = str(message.get("role") or "").strip()
                content = message.get("content")
                if not role:
                    errors.append(f"Example {index}, message {message_index}: role is empty.")
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"Example {index}, message {message_index}: content is empty.")
                elif isinstance(content, str):
                    strings.append(content)
                message_count += 1
        elif normalized_type == "completion":
            content = record.get(completion_field)
            if not isinstance(content, str) or not content.strip():
                errors.append(f"Example {index}: missing non-empty '{completion_field}' text.")
            else:
                strings.append(content)
        elif normalized_type == "alpaca":
            instruction = record.get("instruction")
            output = record.get("output")
            if not isinstance(instruction, str) or not instruction.strip():
                errors.append(f"Example {index}: missing non-empty 'instruction'.")
            if not isinstance(output, str) or not output.strip():
                errors.append(f"Example {index}: missing non-empty 'output'.")
            strings.extend(value for value in (instruction, output) if isinstance(value, str))
        else:
            warnings.append(
                f"Dataset type '{normalized_type}' has only basic JSON object validation here."
            )
            strings.extend(value for value in record.values() if isinstance(value, str))

        placeholder_count += sum(
            1
            for value in strings
            if any(marker in value.lower() for marker in _PLACEHOLDER_MARKERS)
        )
        if len(errors) >= 20:
            errors.append("More shape errors were omitted.")
            break

    if not records:
        errors.append("The dataset has no examples.")
    if records and len(records) < 20:
        warnings.append(
            "This is a small draft. Aim for at least 20–50 varied, high-quality examples "
            "before trusting the result."
        )
    if placeholder_count:
        warnings.append(
            f"Replace {placeholder_count} EDIT ME/TODO placeholder(s) with ideal model responses."
        )
    return DatasetReport(
        source=source,
        source_kind="local",
        dataset_type=normalized_type,
        example_count=len(records),
        message_count=message_count,
        placeholder_count=placeholder_count,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _looks_local_dataset(value: str) -> bool:
    path = Path(value).expanduser()
    return (
        value.startswith((".", "/", "~"))
        or path.suffix.lower()
        in {".json", ".jsonl", ".csv", ".parquet", ".arrow", ".txt", ".text"}
    )


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _safe_stat(path: Path) -> Any | None:
    try:
        return path.stat()
    except OSError:
        return None


__all__ = [
    "AdapterArtifact",
    "DatasetReport",
    "LORA_BASE_MODELS",
    "LORA_GOALS",
    "LORA_MEMORY_PROFILES",
    "LoraStudioError",
    "beginner_config_updates",
    "discover_adapter_artifacts",
    "inspect_configured_dataset",
    "inspect_jsonl_text",
    "normalize_project_name",
    "save_chat_jsonl",
    "starter_dataset_template",
    "suggested_ollama_model",
]

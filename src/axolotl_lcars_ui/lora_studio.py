"""Beginner-facing LoRA project, dataset, and artifact helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


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

LORA_GOAL_HINTS: Mapping[str, str] = {
    "Personality / voice": (
        "Teach tone, phrasing, boundaries, and a consistent conversational identity."
    ),
    "Agent behavior": (
        "Teach planning, tool-use patterns, clarification, recovery, and concise progress reports."
    ),
    "Personality + agent": (
        "Blend a recognizable voice with repeatable planning and tool-use behavior."
    ),
}


@dataclass(frozen=True)
class LoraDatasetFormat:
    """One common Hugging Face dataset record shape translated for Axolotl."""

    key: str
    label: str
    summary: str
    record_shape: str
    settings: Mapping[str, Any]


LORA_HF_DATASET_FORMATS: tuple[LoraDatasetFormat, ...] = (
    LoraDatasetFormat(
        key="openai-messages",
        label="OpenAI messages · recommended for chat",
        summary=(
            "Each record has a messages list containing role/content objects. "
            "This is the same shape created by the Studio's local builder."
        ),
        record_shape='{"messages": [{"role": "user", "content": "..."}, ...]}',
        settings={
            "datasets.0.type": "chat_template",
            "datasets.0.field_messages": "messages",
            "datasets.0.roles_to_train": ("assistant",),
            "datasets.0.train_on_eos": "turn",
        },
    ),
    LoraDatasetFormat(
        key="sharegpt",
        label="ShareGPT conversations · from/value",
        summary=(
            "Each record has a conversations list whose messages use from/value "
            "instead of role/content."
        ),
        record_shape='{"conversations": [{"from": "human", "value": "..."}, ...]}',
        settings={
            "datasets.0.type": "chat_template",
            "datasets.0.field_messages": "conversations",
            "datasets.0.message_property_mappings": {
                "role": "from",
                "content": "value",
            },
            "datasets.0.roles_to_train": ("assistant",),
            "datasets.0.train_on_eos": "turn",
        },
    ),
    LoraDatasetFormat(
        key="alpaca",
        label="Alpaca instructions · instruction/input/output",
        summary=(
            "Instruction-tuning rows with instruction and output fields, plus an optional input."
        ),
        record_shape='{"instruction": "...", "input": "...", "output": "..."}',
        settings={
            "datasets.0.type": "alpaca",
        },
    ),
    LoraDatasetFormat(
        key="plain-text",
        label="Plain text · one text field",
        summary=(
            "Each row already contains the complete training text in a field named text. "
            "Use advanced config when the field has another name."
        ),
        record_shape='{"text": "complete training text"}',
        settings={
            "datasets.0.type": "completion",
            "datasets.0.field": "text",
        },
    ),
)

_LORA_HF_DATASET_FORMATS_BY_KEY = {
    dataset_format.key: dataset_format for dataset_format in LORA_HF_DATASET_FORMATS
}
_HF_DATASET_REPO_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)
_HF_DATASET_FORMAT_RESET_KEYS = (
    "datasets.0.name",
    "datasets.0.data_files",
    "datasets.0.ds_type",
    "datasets.0.field",
    "datasets.0.field_messages",
    "datasets.0.chat_template",
    "datasets.0.chat_template_jinja",
    "datasets.0.train_on_eos",
    "datasets.0.train_on_eot",
    "datasets.0.roles_to_train",
    "datasets.0.roles",
    "datasets.0.revision",
    "datasets.0.input_transform",
    "datasets.0.shards",
    "datasets.0.shards_idx",
    "datasets.0.preprocess_shards",
    "datasets.0.conversation",
    "datasets.0.input_format",
    "datasets.0.field_human",
    "datasets.0.field_model",
    "datasets.0.field_tools",
    "datasets.0.field_thinking",
    "datasets.0.template_thinking_key",
    "datasets.0.message_field_role",
    "datasets.0.message_field_content",
    "datasets.0.message_property_mappings",
    "datasets.0.message_field_training",
    "datasets.0.message_field_training_detail",
    "datasets.0.split_thinking",
    "datasets.0.logprobs_field",
    "datasets.0.temperature",
    "datasets.0.drop_system_message",
)


@dataclass(frozen=True)
class LoraModelTemplate:
    """Architecture-aware defaults for one supported chat-model checkpoint."""

    key: str
    label: str
    model_id: str
    family: str
    architecture: str
    summary: str
    hardware: str
    default_preset: str
    parameter_billions: float
    qlora_below_vram_gb: float
    high_detail_vram_gb: float
    min_axolotl: str
    settings: Mapping[str, Any]
    moe: bool = False


_QWEN_TEXT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "down_proj",
    "up_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
)

_QWEN_TEXT_TEMPLATE_SETTINGS: Mapping[str, Any] = {
    # Axolotl selects the Qwen conditional-generation class from the model config.
    "model_type": None,
    "tokenizer_type": None,
    "datasets.0.chat_template": None,
    "chat_template": "qwen3_5",
    "lora_target_linear": False,
    "lora_target_modules": _QWEN_TEXT_LORA_TARGETS,
    # Packed Qwen hybrid-attention training requires the optional FLA install.
    # The guided default favors a first run that works without that extra dependency.
    "sample_packing": False,
    "pad_to_sequence_len": False,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
}

_GEMMA4_TEXT_TARGET_REGEX = (
    r"model.language_model.layers.[\d]+.(_checkpoint_wrapped_module.)?"
    r"(mlp|self_attn).(up|down|gate|q|k|v|o)_proj"
)

_GEMMA4_TEXT_TEMPLATE_SETTINGS: Mapping[str, Any] = {
    # Gemma 4 is multimodal even for a text-only Studio dataset. Restrict the
    # adapter to its language backbone instead of targeting every linear layer.
    "model_type": None,
    "tokenizer_type": None,
    "datasets.0.chat_template": None,
    "chat_template": "gemma4",
    "eot_tokens": ("<turn|>",),
    "lora_target_linear": False,
    "lora_target_modules": _GEMMA4_TEXT_TARGET_REGEX,
    "pad_to_sequence_len": False,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "lora_dropout": 0.0,
}

LORA_MODEL_TEMPLATES: tuple[LoraModelTemplate, ...] = (
    LoraModelTemplate(
        key="qwen35-2b",
        label="Qwen 3.5 2B · smallest Qwen",
        model_id="Qwen/Qwen3.5-2B",
        family="Qwen 3.5",
        architecture="2B dense hybrid",
        summary=(
            "Best Qwen starter for prototypes and narrow behavior adapters. "
            "The template trains text/chat behavior and leaves multimodal components untouched."
        ),
        hardware="Balanced LoRA on a modest GPU; QLoRA when memory is tight",
        default_preset="balanced",
        parameter_billions=2.0,
        qlora_below_vram_gb=10,
        high_detail_vram_gb=24,
        min_axolotl="0.16.0",
        settings=_QWEN_TEXT_TEMPLATE_SETTINGS,
    ),
    LoraModelTemplate(
        key="qwen35-4b",
        label="Qwen 3.5 4B · everyday Qwen",
        model_id="Qwen/Qwen3.5-4B",
        family="Qwen 3.5",
        architecture="4B dense hybrid",
        summary=(
            "A practical quality/size balance for personality, support, coding, and agent behavior."
        ),
        hardware="Balanced LoRA with comfortable memory; QLoRA on smaller GPUs",
        default_preset="balanced",
        parameter_billions=4.0,
        qlora_below_vram_gb=14,
        high_detail_vram_gb=32,
        min_axolotl="0.16.0",
        settings=_QWEN_TEXT_TEMPLATE_SETTINGS,
    ),
    LoraModelTemplate(
        key="qwen35-9b",
        label="Qwen 3.5 9B · stronger Qwen",
        model_id="Qwen/Qwen3.5-9B",
        family="Qwen 3.5",
        architecture="9B dense hybrid",
        summary=(
            "The strongest requested Qwen 3.5 dense option. Start in QLoRA mode unless the GPU "
            "has ample headroom."
        ),
        hardware="QLoRA is the conservative default; standard LoRA needs substantially more VRAM",
        default_preset="low-vram",
        parameter_billions=9.0,
        qlora_below_vram_gb=32,
        high_detail_vram_gb=64,
        min_axolotl="0.16.0",
        settings=_QWEN_TEXT_TEMPLATE_SETTINGS,
    ),
    LoraModelTemplate(
        key="qwen36-27b",
        label="Qwen 3.6 27B · dense flagship",
        model_id="Qwen/Qwen3.6-27B",
        family="Qwen 3.6",
        architecture="27B dense hybrid",
        summary=(
            "A large dense coding/agent model. The guided default is QLoRA with a micro batch of "
            "one; multi-GPU or high-memory hardware is still expected."
        ),
        hardware="High-memory workstation/server; begin with QLoRA",
        default_preset="low-vram",
        parameter_billions=27.0,
        qlora_below_vram_gb=80,
        high_detail_vram_gb=120,
        min_axolotl="0.16.0",
        settings=_QWEN_TEXT_TEMPLATE_SETTINGS,
    ),
    LoraModelTemplate(
        key="qwen36-35b-a3b",
        label="Qwen 3.6 35B-A3B · MoE",
        model_id="Qwen/Qwen3.6-35B-A3B",
        family="Qwen 3.6",
        architecture="35B total / 3B active MoE",
        summary=(
            "The requested A3B model: 35B total parameters with 3B active. The safe template "
            "adapts attention and linear-attention paths; routed experts stay frozen by default."
        ),
        hardware="Server-class memory despite only 3B active parameters; begin with QLoRA",
        default_preset="low-vram",
        parameter_billions=35.0,
        qlora_below_vram_gb=96,
        high_detail_vram_gb=160,
        min_axolotl="0.16.0",
        settings=_QWEN_TEXT_TEMPLATE_SETTINGS,
        moe=True,
    ),
    LoraModelTemplate(
        key="gemma4-e2b",
        label="Gemma 4 E2B IT · compact",
        model_id="google/gemma-4-E2B-it",
        family="Gemma 4",
        architecture="2.3B effective / 5.1B total",
        summary=(
            "Compact instruction-tuned Gemma 4. The template targets only the text decoder and "
            "uses Gemma 4's native system-role chat format."
        ),
        hardware="Balanced LoRA with comfortable memory; QLoRA on smaller GPUs",
        default_preset="balanced",
        parameter_billions=5.1,
        qlora_below_vram_gb=16,
        high_detail_vram_gb=32,
        min_axolotl="0.16.1",
        settings=_GEMMA4_TEXT_TEMPLATE_SETTINGS,
    ),
    LoraModelTemplate(
        key="gemma4-e4b",
        label="Gemma 4 E4B IT · stronger",
        model_id="google/gemma-4-E4B-it",
        family="Gemma 4",
        architecture="4.5B effective / 8B total",
        summary=(
            "The stronger on-device Gemma 4 option. Its per-layer embeddings make the actual "
            "checkpoint larger than the E4B name suggests, so QLoRA is the safer first run."
        ),
        hardware="QLoRA is the conservative default; total checkpoint size is about 8B parameters",
        default_preset="low-vram",
        parameter_billions=8.0,
        qlora_below_vram_gb=24,
        high_detail_vram_gb=48,
        min_axolotl="0.16.1",
        settings=_GEMMA4_TEXT_TEMPLATE_SETTINGS,
    ),
)

LORA_BASE_MODELS: tuple[tuple[str, str], ...] = (
    *((template.label, template.model_id) for template in LORA_MODEL_TEMPLATES),
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

LORA_BASE_MODEL_HINTS: Mapping[str, str] = {
    **{
        template.model_id: (
            f"{template.summary} Default recipe: {template.default_preset.replace('-', ' ')}. "
            f"Requires Axolotl {template.min_axolotl}+."
        )
        for template in LORA_MODEL_TEMPLATES
    },
    "unsloth/Llama-3.2-1B-Instruct": (
        "Best first run: small, quick to iterate, and easy to package with a matching Llama base."
    ),
    "meta-llama/Llama-3.2-3B-Instruct": (
        "A stronger small model. It needs more VRAM and may require Hugging Face access approval."
    ),
    "google/gemma-2-2b-it": (
        "A compact non-Llama alternative with good instruction behavior; keep the exact Gemma "
        "family when testing the adapter."
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": (
        "The largest guided choice. Prefer QLoRA unless the detected GPU has generous memory."
    ),
}

OLLAMA_BASE_HINTS = {
    "unsloth/Llama-3.2-1B-Instruct": "llama3.2:1b",
    "NousResearch/Llama-3.2-1B": "llama3.2:1b",
    "meta-llama/Llama-3.2-1B-Instruct": "llama3.2:1b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2:3b",
    "google/gemma-2-2b-it": "gemma2:2b",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral:7b",
    "Qwen/Qwen3.5-2B": "qwen3.5:2b",
    "Qwen/Qwen3.5-4B": "qwen3.5:4b",
    "Qwen/Qwen3.5-9B": "qwen3.5:9b",
    "Qwen/Qwen3.6-27B": "qwen3.6:27b",
    "Qwen/Qwen3.6-35B-A3B": "qwen3.6:35b-a3b",
    "google/gemma-4-E2B-it": "gemma4:e2b",
    "google/gemma-4-E4B-it": "gemma4:e4b",
}

_PROJECT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
_DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}\.jsonl$")
_PLACEHOLDER_MARKERS = ("[edit me", "<edit me", "replace this", "todo:")
_LORA_MODEL_TEMPLATES_BY_ID = {
    template.model_id.casefold(): template for template in LORA_MODEL_TEMPLATES
}

_MODEL_TEMPLATE_RESET_KEYS = (
    "processor_type",
    "chat_template",
    "eot_tokens",
    "lora_target_modules",
    "lora_target_parameters",
    "quantize_moe_experts",
    "activation_offloading",
    "freeze_mm_modules",
    "skip_prepare_dataset",
    "remove_unused_columns",
    "gradient_checkpointing_kwargs",
    "gemma4_hybrid_attn_impl",
    "plugins",
    "use_kernels",
    "use_scattermoe",
    "experts_implementation",
    "lora_qkv_kernel",
    "lora_o_kernel",
    "lora_mlp_kernel",
)


@dataclass(frozen=True)
class LoraPreset:
    """One conservative, plain-language recipe for a common LoRA run."""

    key: str
    label: str
    summary: str
    best_for: str
    hardware: str
    speed: str
    settings: Mapping[str, Any]

    @property
    def method(self) -> str:
        return str(self.settings.get("adapter") or "lora").upper()


LORA_PRESETS: tuple[LoraPreset, ...] = (
    LoraPreset(
        key="quick-check",
        label="Quick check · fastest first run",
        summary=(
            "A short, low-capacity run that proves the model, data, and training pipeline work."
        ),
        best_for="New projects, format checks, and catching setup errors before a long run",
        hardware="1–3B model on a modest GPU",
        speed="Fastest · 1 epoch · 1K context",
        settings={
            "adapter": "lora",
            "load_in_8bit": True,
            "load_in_4bit": False,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "sequence_len": 1024,
            "sample_packing": False,
            "pad_to_sequence_len": False,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_epochs": 1,
            "learning_rate": 0.0002,
            "optimizer": "adamw_bnb_8bit",
            "warmup_steps": 5,
        },
    ),
    LoraPreset(
        key="balanced",
        label="Everyday chat · safe default",
        summary=(
            "The general-purpose starter for voice, assistant behavior, and small instruction sets."
        ),
        best_for="Most personality, assistant, support, and tool-behavior adapters",
        hardware="1–3B model with roughly 10–24 GB VRAM",
        speed="Balanced · 3 epochs · 2K context",
        settings={
            "adapter": "lora",
            "load_in_8bit": True,
            "load_in_4bit": False,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "sequence_len": 2048,
            "sample_packing": True,
            "pad_to_sequence_len": True,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_epochs": 3,
            "learning_rate": 0.0001,
            "optimizer": "adamw_bnb_8bit",
            "warmup_steps": 10,
        },
    ),
    LoraPreset(
        key="low-vram",
        label="Low VRAM · 4-bit QLoRA",
        summary=(
            "Loads the base in 4-bit and trades some speed for the smallest practical GPU footprint."
        ),
        best_for="Laptop GPUs, larger base models, or an out-of-memory balanced run",
        hardware="About 6–14 GB for common 1–8B models; model and context still matter",
        speed="Memory saver · 3 epochs · 2K context",
        settings={
            "adapter": "qlora",
            "load_in_8bit": False,
            "load_in_4bit": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "sequence_len": 2048,
            "sample_packing": True,
            "pad_to_sequence_len": True,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_epochs": 3,
            "learning_rate": 0.0002,
            "optimizer": "paged_adamw_8bit",
            "warmup_steps": 10,
        },
    ),
    LoraPreset(
        key="high-detail",
        label="High detail · more adapter capacity",
        summary=(
            "Uses a larger rank, longer context, and larger effective batch for nuanced behavior."
        ),
        best_for="Complex response formats or behavior that clearly underfits the balanced preset",
        hardware="24+ GB for 1–3B models; substantially more for a 7B model",
        speed="Slowest · 3 epochs · 4K context",
        settings={
            "adapter": "lora",
            "load_in_8bit": True,
            "load_in_4bit": False,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "sequence_len": 4096,
            "sample_packing": True,
            "pad_to_sequence_len": True,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "num_epochs": 3,
            "learning_rate": 0.0001,
            "optimizer": "adamw_bnb_8bit",
            "warmup_steps": 10,
        },
    ),
)

LORA_PRESET_KEYS = tuple(preset.key for preset in LORA_PRESETS)
DEFAULT_LORA_PRESET = "balanced"


@dataclass(frozen=True)
class LoraTuningHint:
    """Beginner-readable guidance for one setting exposed by the presets."""

    key: str
    label: str
    purpose: str
    starter_range: str
    tune_when: str
    tradeoff: str

    @property
    def inline_hint(self) -> str:
        return (
            f"{self.purpose} Starter: {self.starter_range}. "
            f"Tune when: {self.tune_when} Tradeoff: {self.tradeoff}"
        )


LORA_TUNING_HINTS: tuple[LoraTuningHint, ...] = (
    LoraTuningHint(
        "adapter",
        "Method",
        "LoRA is the straightforward path; QLoRA loads the base model in 4-bit to save VRAM.",
        "LoRA first, QLoRA after an out-of-memory error",
        "the selected model does not fit",
        "QLoRA uses much less memory but can train more slowly and needs careful adapter export.",
    ),
    LoraTuningHint(
        "lora_r",
        "LoRA rank",
        "Controls how much new behavior the adapter can represent.",
        "8 for a test, 16 for most runs, 32 for proven underfitting",
        "good data still produces responses that are too generic or misses a complex format",
        "Higher rank increases trainable parameters, memory use, adapter size, and overfit risk.",
    ),
    LoraTuningHint(
        "lora_alpha",
        "LoRA alpha",
        "Scales the adapter update relative to the base model.",
        "Usually 2 × rank",
        "you deliberately change rank",
        "Changing alpha alone can make training too weak or too aggressive.",
    ),
    LoraTuningHint(
        "lora_dropout",
        "Dropout",
        "Randomly hides a small share of adapter activations to reduce memorization.",
        "0.0–0.1; 0.05 is a safe default",
        "a small dataset copies training wording but fails held-out prompts",
        "More dropout can improve generalization but can also cause underfitting.",
    ),
    LoraTuningHint(
        "sequence_len",
        "Context length",
        "Sets the maximum token length of each prepared training sample.",
        "1024 for quick tests, 2048 normally, 4096 only when examples need it",
        "important examples are being truncated",
        "Longer context can increase activation memory and training time sharply.",
    ),
    LoraTuningHint(
        "micro_batch_size",
        "Micro batch",
        "Controls how many samples each GPU processes at once.",
        "1–2",
        "GPU memory is underused, or the run reports out of memory",
        "Raise for throughput; lower first after an out-of-memory error.",
    ),
    LoraTuningHint(
        "gradient_accumulation_steps",
        "Gradient accumulation",
        "Builds a larger effective batch without keeping every sample in VRAM at once.",
        "4–16",
        "you lower micro batch or loss is very noisy",
        "Higher values stabilize updates but add time between optimizer steps.",
    ),
    LoraTuningHint(
        "num_epochs",
        "Epochs",
        "Counts complete passes through the training dataset.",
        "1 to prove the pipeline; 2–3 for a real first run",
        "held-out behavior still underfits, or training wording is being memorized",
        "More epochs can strengthen behavior but quickly overfit small datasets.",
    ),
    LoraTuningHint(
        "learning_rate",
        "Learning rate",
        "Controls the size of each optimizer update.",
        "0.0001–0.0003 for LoRA-style SFT",
        "loss is flat after checking labels, or loss spikes and becomes unstable",
        "Higher learns faster but can overshoot; lower is steadier but may underfit.",
    ),
    LoraTuningHint(
        "sample_packing",
        "Sample packing",
        "Combines short examples into fuller sequences for better GPU utilization.",
        "On for normal runs; off for the first debug run",
        "you are diagnosing malformed samples or unexplained loss spikes",
        "Packing is faster, while unpacked samples are easier to inspect.",
    ),
)

_LORA_PRESETS_BY_KEY = {preset.key: preset for preset in LORA_PRESETS}
_LORA_TUNING_HINTS_BY_KEY = {hint.key: hint for hint in LORA_TUNING_HINTS}


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
class DatasetImport:
    """Validated local JSON/JSONL data normalized for Axolotl."""

    source_name: str
    suggested_filename: str
    detected_format: str
    format_key: str
    jsonl_text: str
    report: DatasetReport
    preview_rows: tuple[Mapping[str, str], ...] = ()


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
    preset: str | None = None,
    memory_profile: str | None = None,
) -> dict[str, Any]:
    """Translate the short setup wizard into a conservative Axolotl config."""

    slug = normalize_project_name(project_name)
    # Keep the original memory-profile API working for saved sessions and integrations.
    if preset is None and memory_profile is not None:
        if memory_profile not in LORA_MEMORY_PROFILES:
            raise LoraStudioError("Choose one of the guided memory profiles.")
        preset = "low-vram" if memory_profile == LORA_MEMORY_PROFILES[1] else "balanced"
    selected = get_lora_preset(preset or DEFAULT_LORA_PRESET)
    model_template = get_lora_model_template(base_model)
    updates: dict[str, Any] = {
        # Clear architecture-specific fields left by a previously selected model.
        **{key: None for key in _MODEL_TEMPLATE_RESET_KEYS},
        "base_model": base_model.strip(),
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "datasets.0.path": f"./data/{slug}.jsonl",
        "datasets.0.type": "chat_template",
        "datasets.0.ds_type": "json",
        "datasets.0.field_messages": "messages",
        "datasets.0.chat_template": "tokenizer_default",
        "datasets.0.roles_to_train": "assistant",
        "datasets.0.train_on_eos": "turn",
        "dataset_prepared_path": f"./prepared/{slug}",
        "val_set_size": 0.1,
        "output_dir": f"./outputs/{slug}",
        "save_safetensors": True,
        "lora_target_linear": True,
        "train_on_inputs": False,
        "max_steps": None,
        "batch_size": None,
        "lr_scheduler": "cosine",
        "warmup_ratio": None,
        "weight_decay": 0.01,
        "bf16": "auto",
        "fp16": False,
        "gradient_checkpointing": "true",
        "attn_implementation": "sdpa",
        "logging_steps": 1,
        "save_steps": 100,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "eval_steps": 100,
        "eval_strategy": "epoch",
        "strict": False,
    }
    updates.update(selected.settings)
    if model_template is not None:
        updates.update(model_template.settings)
        if model_template.moe:
            # Axolotl forbids the generic all-linear target when expert quantization
            # is enabled. The model template already uses explicit attention targets.
            updates["quantize_moe_experts"] = selected.method == "QLORA"
    return updates


def get_lora_preset(key: str) -> LoraPreset:
    """Return a preset by stable key with a beginner-readable error."""

    try:
        return _LORA_PRESETS_BY_KEY[key]
    except KeyError as exc:
        raise LoraStudioError("Choose one of the guided smart presets.") from exc


def get_lora_model_template(base_model: str) -> LoraModelTemplate | None:
    """Return architecture defaults for a known official model checkpoint."""

    return _LORA_MODEL_TEMPLATES_BY_ID.get(base_model.strip().casefold())


def get_lora_dataset_format(key: str) -> LoraDatasetFormat:
    """Return a downloaded-dataset shape by stable key."""

    try:
        return _LORA_HF_DATASET_FORMATS_BY_KEY[key]
    except KeyError as exc:
        raise LoraStudioError("Choose one of the common downloaded dataset formats.") from exc


def downloaded_dataset_config_updates(
    repo_id: str,
    format_key: str,
    *,
    split: str = "train",
    subset: str = "",
    use_top_level_chat_template: bool = False,
) -> dict[str, Any]:
    """Translate a cached Hugging Face dataset choice into a clean Axolotl source."""

    normalized_repo = repo_id.strip()
    if not _HF_DATASET_REPO_PATTERN.fullmatch(normalized_repo):
        raise LoraStudioError("Choose a downloaded Hugging Face dataset in owner/name form.")
    selected_format = get_lora_dataset_format(format_key)
    normalized_split = _clean_hf_dataset_option(split, fallback="train", label="split")
    normalized_subset = _clean_hf_dataset_option(subset, fallback="", label="subset")
    updates: dict[str, Any] = {
        **{key: None for key in _HF_DATASET_FORMAT_RESET_KEYS},
        "datasets.0.path": normalized_repo,
        "datasets.0.split": normalized_split,
        "datasets.0.name": normalized_subset or None,
    }
    updates.update(selected_format.settings)
    if selected_format.settings.get("datasets.0.type") == "chat_template":
        updates["datasets.0.chat_template"] = (
            None if use_top_level_chat_template else "tokenizer_default"
        )
    return updates


def local_dataset_config_updates(
    filename: str,
    format_key: str,
    *,
    use_top_level_chat_template: bool = False,
) -> dict[str, Any]:
    """Translate a validated local JSONL file into a clean Axolotl source."""

    clean_name = _validated_dataset_filename(filename)
    selected_format = get_lora_dataset_format(format_key)
    updates: dict[str, Any] = {
        **{key: None for key in _HF_DATASET_FORMAT_RESET_KEYS},
        "datasets.0.path": f"./data/{clean_name}",
        "datasets.0.split": None,
        "datasets.0.ds_type": "json",
    }
    updates.update(selected_format.settings)
    if selected_format.settings.get("datasets.0.type") == "chat_template":
        updates["datasets.0.chat_template"] = (
            None if use_top_level_chat_template else "tokenizer_default"
        )
    return updates


def infer_lora_preset(cfg: Mapping[str, Any]) -> str:
    """Infer the closest smart preset from an existing Axolotl config."""

    if str(cfg.get("adapter") or "").lower() == "qlora" or bool(cfg.get("load_in_4bit")):
        return "low-vram"
    try:
        rank = int(cfg.get("lora_r") or 0)
        context = int(cfg.get("sequence_len") or 0)
        epochs = float(cfg.get("num_epochs") or 0)
    except (TypeError, ValueError):
        return DEFAULT_LORA_PRESET
    if epochs and epochs <= 1 and rank <= 8 and context <= 1024:
        return "quick-check"
    if rank >= 32 or context >= 4096:
        return "high-detail"
    return DEFAULT_LORA_PRESET


def _clean_hf_dataset_option(value: str, *, fallback: str, label: str) -> str:
    normalized = value.strip() or fallback
    if len(normalized) > 128 or any(character in normalized for character in ("\n", "\r")):
        raise LoraStudioError(f"Dataset {label} must be one short line.")
    return normalized


def recommend_lora_preset(
    gpu_vram_gb: float | None,
    base_model: str,
) -> str:
    """Recommend a recipe from detected VRAM and the selected model size."""

    model_template = get_lora_model_template(base_model)
    if model_template is not None:
        if gpu_vram_gb is None or gpu_vram_gb <= 0:
            return model_template.default_preset
        if gpu_vram_gb < model_template.qlora_below_vram_gb:
            return "low-vram"
        if gpu_vram_gb >= model_template.high_detail_vram_gb:
            return "high-detail"
        return "balanced"

    model_size = _model_size_billions(base_model)
    if gpu_vram_gb is None or gpu_vram_gb <= 0:
        return DEFAULT_LORA_PRESET
    if model_size >= 6:
        if gpu_vram_gb < 24:
            return "low-vram"
        return "high-detail" if gpu_vram_gb >= 48 else DEFAULT_LORA_PRESET
    if gpu_vram_gb < 10:
        return "low-vram"
    if gpu_vram_gb >= 24:
        return "high-detail"
    return DEFAULT_LORA_PRESET


def lora_tuning_hint(key: str) -> str:
    """Return concise inline help for a common LoRA field."""

    hint = _LORA_TUNING_HINTS_BY_KEY.get(key)
    return hint.inline_hint if hint is not None else ""


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


def chat_example_line(
    user_prompt: str,
    ideal_response: str,
    *,
    system_prompt: str = "",
) -> str:
    """Build one OpenAI-messages JSONL line from a beginner form."""

    user = user_prompt.strip()
    assistant = ideal_response.strip()
    system = system_prompt.strip()
    if not user:
        raise LoraStudioError("Write what the user says in this example.")
    if not assistant:
        raise LoraStudioError("Write the exact ideal answer the model should learn.")
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(
        (
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        )
    )
    return json.dumps({"messages": messages}, ensure_ascii=False)


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


def parse_json_dataset(filename: str, content: bytes) -> DatasetImport:
    """Parse, normalize, validate, and preview an uploaded JSON or JSONL dataset."""

    source_name = Path(filename.strip()).name
    suffix = Path(source_name).suffix.lower()
    if not source_name or suffix not in {".json", ".jsonl"}:
        raise LoraStudioError("Choose a .json or .jsonl dataset file.")
    if not content:
        raise LoraStudioError("The selected dataset file is empty.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LoraStudioError(
            f"The dataset must be UTF-8 text (invalid byte near position {exc.start})."
        ) from exc

    container_warning = ""
    if suffix == ".jsonl":
        records = _parse_uploaded_jsonl(text)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LoraStudioError(
                f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}."
            ) from exc
        records, container_name = _uploaded_json_records(payload)
        if container_name:
            container_warning = (
                f"Loaded examples from the top-level '{container_name}' array."
            )

    if not records:
        raise LoraStudioError("The dataset does not contain any examples.")
    shapes = {
        _uploaded_record_shape(record, index)
        for index, record in enumerate(records, start=1)
    }
    if len(shapes) != 1:
        labels = ", ".join(sorted(_uploaded_shape_label(shape) for shape in shapes))
        raise LoraStudioError(
            "The file mixes incompatible record shapes "
            f"({labels}). Convert every example to one consistent shape."
        )
    shape = next(iter(shapes))
    normalized = [
        _normalize_uploaded_record(record, shape, index)
        for index, record in enumerate(records, start=1)
    ]
    format_key = "plain-text" if shape[0] == "plain-text" else "openai-messages"
    dataset_type = "completion" if format_key == "plain-text" else "chat_template"
    lines = [json.dumps(record, ensure_ascii=False) for record in normalized]
    jsonl_text = "\n".join(lines)
    report = _inspect_records(
        normalized,
        source=source_name,
        dataset_type=dataset_type,
        messages_field="messages",
        completion_field="text",
    )
    warnings = list(report.warnings)
    if container_warning:
        warnings.append(container_warning)
    if shape[0] not in {"openai-messages", "plain-text"}:
        warnings.append(
            f"Detected {_uploaded_shape_label(shape)} and normalized it to OpenAI messages."
        )
    duplicate_count = len(lines) - len(set(lines))
    if duplicate_count:
        warnings.append(
            f"Found {duplicate_count} exact duplicate example(s); remove duplicates "
            "before a long training run."
        )
    report = DatasetReport(
        source=report.source,
        source_kind=report.source_kind,
        dataset_type=report.dataset_type,
        example_count=report.example_count,
        message_count=report.message_count,
        placeholder_count=report.placeholder_count,
        errors=report.errors,
        warnings=tuple(dict.fromkeys(warnings)),
    )
    return DatasetImport(
        source_name=source_name,
        suggested_filename=_suggested_jsonl_filename(source_name),
        detected_format=_uploaded_shape_label(shape),
        format_key=format_key,
        jsonl_text=jsonl_text,
        report=report,
        preview_rows=_uploaded_preview_rows(normalized, format_key),
    )


def save_jsonl_dataset(
    project_root: Path,
    filename: str,
    text: str,
    *,
    dataset_type: str = "chat_template",
    messages_field: str = "messages",
    completion_field: str = "text",
) -> tuple[Path, DatasetReport]:
    """Validate and save normalized JSONL while preserving a recoverable backup."""

    clean_name = _validated_dataset_filename(filename)
    report = inspect_jsonl_text(
        text,
        source=f"./data/{clean_name}",
        dataset_type=dataset_type,
        messages_field=messages_field,
        completion_field=completion_field,
    )
    if report.errors:
        raise LoraStudioError(report.errors[0])
    if not report.example_count:
        raise LoraStudioError("Add at least one JSONL training example before saving.")

    data_dir = (project_root / "data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    target = (data_dir / clean_name).resolve()
    if target.parent != data_dir:
        raise LoraStudioError("The dataset must stay inside this project's data directory.")
    handle, temp_name = tempfile.mkstemp(
        dir=data_dir,
        prefix=".lcars-dataset-",
        suffix=".tmp",
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text.rstrip() + "\n")
        if target.exists():
            backup = target.with_suffix(target.suffix + ".bak")
            shutil.copy2(target, backup)
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return target, report


def save_chat_jsonl(project_root: Path, filename: str, text: str) -> tuple[Path, DatasetReport]:
    """Validate and save a guided dataset, preserving one recoverable prior draft."""

    return save_jsonl_dataset(project_root, filename, text)


def _parse_uploaded_jsonl(text: str) -> list[Any]:
    records: list[Any] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"Line {line_number}, column {exc.colno}: {exc.msg}.")
            if len(errors) >= 5:
                errors.append("More JSONL syntax errors were omitted.")
                break
    if errors:
        raise LoraStudioError(" ".join(errors))
    return records


def _uploaded_json_records(payload: Any) -> tuple[list[Any], str]:
    if isinstance(payload, list):
        return payload, ""
    if not isinstance(payload, dict):
        raise LoraStudioError(
            "A JSON dataset must be an example object, a list of examples, or an object "
            "containing a data/records/examples/train list."
        )
    for key in ("data", "records", "examples", "train"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return candidate, key
    return [payload], ""


def _uploaded_record_shape(record: Any, index: int) -> tuple[str, ...]:
    if not isinstance(record, dict):
        raise LoraStudioError(f"Example {index}: expected a JSON object.")
    if "messages" in record:
        return ("openai-messages",)
    if "conversations" in record:
        return ("sharegpt",)
    if "instruction" in record and "output" in record:
        return ("alpaca",)
    for prompt_key, response_key in (
        ("prompt", "response"),
        ("question", "answer"),
        ("input", "output"),
    ):
        if prompt_key in record and response_key in record:
            return ("prompt-response", prompt_key, response_key)
    if "text" in record:
        return ("plain-text",)
    keys = ", ".join(str(key) for key in list(record)[:8]) or "none"
    raise LoraStudioError(
        f"Example {index}: unsupported columns ({keys}). Expected messages, conversations, "
        "instruction/output, prompt/response, question/answer, input/output, or text."
    )


def _uploaded_shape_label(shape: tuple[str, ...]) -> str:
    labels = {
        "openai-messages": "OpenAI messages",
        "sharegpt": "ShareGPT conversations",
        "alpaca": "Alpaca instruction/input/output",
        "plain-text": "plain text",
    }
    if shape[0] == "prompt-response":
        return f"{shape[1]}/{shape[2]} pairs"
    return labels[shape[0]]


def _normalize_uploaded_record(
    record: Any,
    shape: tuple[str, ...],
    index: int,
) -> dict[str, Any]:
    assert isinstance(record, dict)
    if shape[0] == "openai-messages":
        return {
            "messages": _normalize_uploaded_messages(
                record.get("messages"),
                index,
                role_key="role",
                content_key="content",
            )
        }
    if shape[0] == "sharegpt":
        return {
            "messages": _normalize_uploaded_messages(
                record.get("conversations"),
                index,
                role_key="from",
                content_key="value",
            )
        }
    if shape[0] == "alpaca":
        instruction = _required_uploaded_text(
            record.get("instruction"),
            index,
            "instruction",
        )
        output = _required_uploaded_text(record.get("output"), index, "output")
        optional_input = record.get("input")
        if optional_input is not None and not isinstance(optional_input, str):
            raise LoraStudioError(f"Example {index}: 'input' must be text when present.")
        user_content = instruction
        if isinstance(optional_input, str) and optional_input.strip():
            user_content = f"{instruction}\n\nInput:\n{optional_input.strip()}"
        return {
            "messages": _prompt_response_messages(
                user_content,
                output,
                record.get("system"),
                index,
            )
        }
    if shape[0] == "prompt-response":
        prompt = _required_uploaded_text(record.get(shape[1]), index, shape[1])
        response = _required_uploaded_text(record.get(shape[2]), index, shape[2])
        return {
            "messages": _prompt_response_messages(
                prompt,
                response,
                record.get("system"),
                index,
            )
        }
    text = _required_uploaded_text(record.get("text"), index, "text")
    return {"text": text}


def _normalize_uploaded_messages(
    messages: Any,
    example_index: int,
    *,
    role_key: str,
    content_key: str,
) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise LoraStudioError(
            f"Example {example_index}: '{role_key}/{content_key}' messages must be "
            "a non-empty list."
        )
    role_aliases = {
        "human": "user",
        "gpt": "assistant",
        "bot": "assistant",
        "model": "assistant",
    }
    allowed_roles = {"system", "user", "assistant", "tool"}
    normalized: list[dict[str, str]] = []
    for message_index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise LoraStudioError(
                f"Example {example_index}, message {message_index}: expected an object."
            )
        role = str(message.get(role_key) or "").strip().lower()
        role = role_aliases.get(role, role)
        if role not in allowed_roles:
            raise LoraStudioError(
                f"Example {example_index}, message {message_index}: unsupported role "
                f"{role!r}; use system, user/human, assistant/gpt, or tool."
            )
        content = _required_uploaded_text(
            message.get(content_key),
            example_index,
            f"message {message_index} {content_key}",
        )
        normalized.append({"role": role, "content": content})
    roles = {message["role"] for message in normalized}
    if "user" not in roles or "assistant" not in roles:
        raise LoraStudioError(
            f"Example {example_index}: each chat example needs at least one user and "
            "one assistant message."
        )
    return normalized


def _prompt_response_messages(
    prompt: str,
    response: str,
    system: Any,
    example_index: int,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system is not None:
        if not isinstance(system, str):
            raise LoraStudioError(
                f"Example {example_index}: optional 'system' must be text."
            )
        if system.strip():
            messages.append({"role": "system", "content": system.strip()})
    messages.extend(
        (
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        )
    )
    return messages


def _required_uploaded_text(value: Any, example_index: int, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LoraStudioError(
            f"Example {example_index}: '{field}' must contain non-empty text."
        )
    return value.strip()


def _suggested_jsonl_filename(source_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(source_name).stem).strip("._-")
    if not stem or not stem[0].isalnum():
        stem = f"dataset-{stem}" if stem else "imported-dataset"
    return f"{stem[:118]}.jsonl"


def _validated_dataset_filename(filename: str) -> str:
    clean_name = Path(filename.strip()).name
    if clean_name != filename.strip() or not _DATASET_NAME_PATTERN.fullmatch(clean_name):
        raise LoraStudioError(
            "Use a simple .jsonl filename containing letters, numbers, dots, "
            "dashes, or underscores."
        )
    return clean_name


def _uploaded_preview_rows(
    records: list[dict[str, Any]],
    format_key: str,
    *,
    limit: int = 8,
) -> tuple[Mapping[str, str], ...]:
    rows: list[Mapping[str, str]] = []
    for index, record in enumerate(records[:limit], start=1):
        if format_key == "plain-text":
            rows.append(
                {
                    "Example": str(index),
                    "Input": _compact_preview(str(record.get("text") or "")),
                    "Ideal output": "Completion text",
                    "Messages": "1 text",
                }
            )
            continue
        messages = record.get("messages")
        assert isinstance(messages, list)
        user = next(
            (
                str(message.get("content") or "")
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            "",
        )
        assistant = next(
            (
                str(message.get("content") or "")
                for message in reversed(messages)
                if isinstance(message, dict) and message.get("role") == "assistant"
            ),
            "",
        )
        rows.append(
            {
                "Example": str(index),
                "Input": _compact_preview(user),
                "Ideal output": _compact_preview(assistant),
                "Messages": str(len(messages)),
            }
        )
    return tuple(rows)


def _compact_preview(value: str, *, limit: int = 180) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


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


def _model_size_billions(model_name: str) -> float:
    """Best-effort parameter count from common Hugging Face model-id tokens."""

    matches = re.findall(
        r"(?:^|[-_/])(\d+(?:\.\d+)?)b(?:$|[-_/])",
        model_name.lower(),
    )
    if not matches:
        return 0.0
    try:
        return float(matches[-1])
    except ValueError:
        return 0.0


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
    "DEFAULT_LORA_PRESET",
    "DatasetImport",
    "DatasetReport",
    "LORA_BASE_MODELS",
    "LORA_BASE_MODEL_HINTS",
    "LORA_GOALS",
    "LORA_GOAL_HINTS",
    "LORA_HF_DATASET_FORMATS",
    "LORA_MEMORY_PROFILES",
    "LORA_MODEL_TEMPLATES",
    "LORA_PRESETS",
    "LORA_PRESET_KEYS",
    "LORA_TUNING_HINTS",
    "LoraPreset",
    "LoraDatasetFormat",
    "LoraModelTemplate",
    "LoraStudioError",
    "LoraTuningHint",
    "beginner_config_updates",
    "chat_example_line",
    "discover_adapter_artifacts",
    "downloaded_dataset_config_updates",
    "get_lora_dataset_format",
    "get_lora_preset",
    "get_lora_model_template",
    "infer_lora_preset",
    "inspect_configured_dataset",
    "inspect_jsonl_text",
    "local_dataset_config_updates",
    "lora_tuning_hint",
    "normalize_project_name",
    "parse_json_dataset",
    "recommend_lora_preset",
    "save_chat_jsonl",
    "save_jsonl_dataset",
    "starter_dataset_template",
    "suggested_ollama_model",
]

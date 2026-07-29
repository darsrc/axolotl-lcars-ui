"""LCARS application entry point for Axolotl management."""

from __future__ import annotations

import argparse
import asyncio
import html
import math
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import lcars_ui as lcars
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from starlette.datastructures import State

from axolotl_lcars_ui.config_store import FIELD_SPECS, ConfigError, ConfigStore, FieldSpec
from axolotl_lcars_ui.hf_manager import (
    HuggingFaceManager,
    cache_summary_text,
    sorted_search_results,
)
from axolotl_lcars_ui.lora_studio import (
    DEFAULT_LORA_PRESET,
    LORA_BASE_MODELS,
    LORA_BASE_MODEL_HINTS,
    LORA_GOALS,
    LORA_GOAL_HINTS,
    LORA_HF_DATASET_FORMATS,
    LORA_MODEL_TEMPLATES,
    LORA_PRESETS,
    LORA_PRESET_KEYS,
    LORA_TUNING_HINTS,
    DatasetReport,
    LoraStudioError,
    beginner_config_updates,
    chat_example_line,
    discover_adapter_artifacts,
    downloaded_dataset_config_updates,
    get_lora_dataset_format,
    get_lora_model_template,
    get_lora_preset,
    infer_lora_preset,
    inspect_configured_dataset,
    inspect_jsonl_text,
    lora_tuning_hint,
    normalize_project_name,
    recommend_lora_preset,
    save_chat_jsonl,
    starter_dataset_template,
    suggested_ollama_model,
)
from axolotl_lcars_ui.ollama import OllamaManager
from axolotl_lcars_ui.resources import (
    DiskInfo,
    TelemetrySampler,
    disk_rows,
    format_bytes,
    gpu_process_rows,
    gpu_rows,
    process_rows,
    storage_hotspot_rows,
)
from axolotl_lcars_ui.runner import AXOLOTL_ACTIONS, CONFIG_ACTIONS, LAUNCHER_ACTIONS, AxolotlRunner
from axolotl_lcars_ui.ui_state import PERSISTED_WIDGET_IDS, UiStateStore
from axolotl_lcars_ui.validator import AxolotlPreflight, PreflightIssue, issue_rows
from axolotl_lcars_ui.workflow import WorkflowError, WorkflowManager

from lcars_ui.app import create_app
from lcars_ui.dsl._builder import _ManifestBuilder
from lcars_ui.dsl._state import Mode, _LCARSContext, get_ctx, get_session_state, set_ctx
from lcars_ui.dsl.api import _index_form_children


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERNAL_NAVIGATION_SCRIPT = Path(__file__).with_name("static") / "internal_navigation.js"
LOG_AXOLOTL = "axolotl-run-log"
LOG_HF = "hf-log"
LOG_OLLAMA = "ollama-test-log"
WORKFLOW_CANVAS_ID = "axolotl-workflow-canvas"
HF_RESULTS_TABLE_ID = "hf-results-table"
HF_RESULTS_PAGE_KEY = "hf-results-page"
HF_RESULTS_PAGE_SIZE_KEY = "hf-results-page-size"
HF_RESULTS_PAGE_SIZE = 10
HF_RESULTS_PAGE_SIZES = (10, 25, 50, 100)
SEARCH_INPUT_OPTIONS = lcars.TextInputOptions(
    input_type="search",
    commit="enter",
    debounce_ms=250,
)
COMMAND_INPUT_OPTIONS = lcars.TextInputOptions(
    commit="blur",
    debounce_ms=250,
)
SEARCHABLE_CHOICES = lcars.ChoiceOptions(searchable=True)
COLLAPSIBLE_PANEL_OPTIONS = lcars.ContainerOptions(
    density="compact",
    overflow="auto",
    collapsible=True,
)
COLLAPSED_PANEL_OPTIONS = lcars.ContainerOptions(
    density="compact",
    overflow="auto",
    collapsible=True,
    initial_collapsed=True,
)
DENSE_PANEL_OPTIONS = lcars.ContainerOptions(
    density="compact",
    overflow="auto",
)
LOG_VIEW_OPTIONS = lcars.LogOptions(
    toolbar=True,
    search=True,
    line_numbers=True,
    wrap=False,
)
CONFIG_GROUP_ORDER = (
    "Run Safety",
    "Model",
    "Dataset",
    "Sequence / Packing",
    "Training",
    "Adapter / PEFT",
    "Optimizer",
    "Precision / Memory",
    "Attention / Kernels",
    "Distributed",
    "Tracking",
    "Integrations",
    "RL / Evaluation",
)

CONFIG_GROUP_NOTES = {
    "Run Safety": "Resume and strictness controls that can change whether a run starts or restarts cleanly.",
    "Model": "Axolotl expects Hugging Face/Transformers model ids or local directories with config/tokenizer/weights.",
    "Dataset": "Dataset source, Axolotl formatter type, split, local file hints, and preprocessing controls.",
    "Sequence / Packing": "Context length, padding, packing, and truncation controls that drive memory and throughput.",
    "Training": "Output destination, model-hub publishing, checkpoint format, and adapter merge controls.",
    "Adapter / PEFT": "LoRA, QLoRA, IA3, GPTQ, bitsandbytes, and adapter-specific settings.",
    "Optimizer": "Batch sizing, epochs or step limits, learning rate schedule, optimizer, and optimizer kwargs.",
    "Precision / Memory": "Quantization, precision, checkpointing, offload, and memory ceilings.",
    "Attention / Kernels": "Modern attention backend selection plus legacy Axolotl switches for compatibility.",
    "Distributed": "DeepSpeed, FSDP, tensor/context parallel, and DDP controls.",
    "Tracking": "Logging, eval, checkpoint cadence, sample generation, and best-model selection.",
    "Integrations": "Weights & Biases, TensorBoard, MLflow, Comet, OpenTelemetry, and Hugging Face auth.",
    "RL / Evaluation": "TRL/RL modes, vLLM knobs, reward-model flags, and lm-eval settings.",
}

SETUP_REQUIRED_KEYS = {"base_model", "datasets.0.path"}
HF_SORT_OPTIONS = ["downloads", "likes", "last_modified", "trending_score"]
HF_QUERY_MODE_OPTIONS = ["Search Hub", "Inspect Exact Repository"]
HF_COMPATIBILITY_OPTIONS = ["compatible files only", "include warnings and blocked"]
HF_ARTIFACT_FILTER_OPTIONS = [
    "any artifact",
    "base/trainable models",
    "PEFT adapters",
    "datasets",
    "runtime only",
]
HF_QUANT_FILTER_OPTIONS = [
    "any weight format",
    "Transformers safetensors",
    "full precision fp16/bf16",
    "4-bit quantized",
    "8-bit quantized",
    "GPTQ quantized",
    "AWQ quantized",
    "GGUF runtime files",
]
HF_FIT_FILTER_OPTIONS = ["any", "known size", "fits vram"]
HF_LIMIT_OPTIONS = ["12", "25", "50"]
SETUP_FIELD_KEYS = {
    "strict",
    "resume_from_checkpoint",
    "auto_resume_from_checkpoints",
    "save_only_model",
    "base_model",
    "revision_of_model",
    "base_model_config",
    "base_model_ignore_patterns",
    "tokenizer_config",
    "model_type",
    "tokenizer_type",
    "trust_remote_code",
    "datasets.0.path",
    "datasets.0.type",
    "datasets.0.split",
    "datasets.0.name",
    "datasets.0.data_files",
    "datasets.0.ds_type",
    "datasets.0.field",
    "datasets.0.field_messages",
    "datasets.0.chat_template",
    "datasets.0.chat_template_jinja",
    "datasets.0.train_on_eos",
    "dataset_prepared_path",
    "val_set_size",
    "streaming",
    "dataset_processes",
    "dataset_num_proc",
    "sequence_len",
    "eval_sequence_len",
    "excess_length_strategy",
    "max_prompt_len",
    "sample_packing",
    "eval_sample_packing",
    "pad_to_sequence_len",
    "pad_to_multiple_of",
}

MODEL_PRESETS = [
    "unsloth/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b",
    "mistralai/Mistral-7B-v0.1",
]

DATASET_PRESETS = {
    "teknium/GPT4-LLM-Cleaned | alpaca": ("teknium/GPT4-LLM-Cleaned", "alpaca"),
    "tatsu-lab/alpaca | alpaca": ("tatsu-lab/alpaca", "alpaca"),
    "HuggingFaceH4/ultrachat_200k | chat_template": (
        "HuggingFaceH4/ultrachat_200k",
        "chat_template",
    ),
    "./data/train.jsonl | completion": ("./data/train.jsonl", "completion"),
}

LORA_BASE_MODEL_VALUES = tuple(value for _, value in LORA_BASE_MODELS)
LORA_HF_DATASET_FORMAT_KEYS = tuple(
    dataset_format.key for dataset_format in LORA_HF_DATASET_FORMATS
)

CONFIG_VALUE_HINTS: dict[str, dict[str, str]] = {
    "adapter": {
        "": "No adapter: all model weights may be trained. This is not a beginner LoRA run.",
        "lora": "Standard LoRA. Faster and simpler when the base model fits in GPU memory.",
        "qlora": "4-bit QLoRA. Choose this when standard LoRA runs out of GPU memory.",
        "ia3": "A different parameter-efficient method; use only for an intentional IA3 config.",
    },
    "bf16": {
        "auto": "Let Axolotl use BF16 when the GPU supports it.",
        "true": "Require BF16. Training will fail on unsupported hardware.",
        "false": "Disable BF16; another precision mode must cover training.",
    },
    "fp16": {
        "unset": "Leave precision selection to BF16/other configured defaults.",
        "true": "Use FP16 on GPUs without BF16 support.",
        "false": "Explicitly disable FP16.",
    },
    "gradient_checkpointing": {
        "": "Use Axolotl's default behavior.",
        "true": "Save VRAM by recomputing activations; usually worth the speed cost.",
        "false": "Use more VRAM for faster training.",
        "offload": "Move checkpointed activations off GPU; slower, but can rescue a tight run.",
        "offload_disk": "Last-resort disk offload with a large performance cost.",
    },
    "attn_implementation": {
        "": "Use the model/Axolotl default attention implementation.",
        "sdpa": "PyTorch scaled-dot-product attention; portable and a safe guided default.",
        "flash_attention_2": "Often faster and leaner, but requires a compatible FlashAttention install.",
        "eager": "Most compatible and easiest to debug, usually slower and more memory-hungry.",
    },
    "save_strategy": {
        "": "Use the trainer default.",
        "no": "Do not create intermediate checkpoints.",
        "epoch": "Save after each full pass through the dataset; simple for small LoRA runs.",
        "steps": "Save on a fixed step cadence for long datasets.",
        "best": "Keep checkpoints according to evaluation quality; requires a working eval setup.",
    },
    "eval_strategy": {
        "": "Use the trainer default.",
        "no": "Skip validation during training.",
        "epoch": "Evaluate after each full dataset pass; simple for small LoRA runs.",
        "steps": "Evaluate on a fixed step cadence for long runs.",
    },
    "chat_template": {
        "qwen3_5": (
            "Axolotl's Qwen 3.5/3.6 role and control-token template. Required by the guided "
            "Qwen model templates."
        ),
        "gemma4": (
            "Axolotl's Gemma 4 conversation format, including Gemma 4 turn boundaries."
        ),
    },
}

CONFIG_FIELD_HINTS: dict[str, str] = {
    "base_model": (
        "The Hugging Face model id or local Transformers directory that the adapter modifies. "
        "An adapter must later be used with this same model lineage."
    ),
    "datasets.0.path": (
        "Training examples to imitate. Use a project-local file for the easiest validation, "
        "or an exact Hugging Face dataset id."
    ),
    "datasets.0.type": (
        "Tells Axolotl how each dataset record is shaped. Use chat_template for OpenAI-style "
        "messages and completion for a single text field."
    ),
    "datasets.0.field_messages": (
        "The record key containing the ordered role/content messages. Keep messages for the "
        "guided JSONL template."
    ),
    "datasets.0.chat_template": (
        "Converts messages to the model's exact control-token format. tokenizer_default is safest "
        "when the chosen tokenizer includes a template."
    ),
    "datasets.0.roles_to_train": (
        "Which message roles contribute to loss. Assistant-only prevents the model from learning "
        "to imitate user prompts."
    ),
    "datasets.0.train_on_eos": (
        "Controls which end-of-sequence tokens are learned. turn matches each trainable assistant "
        "turn in the guided chat format."
    ),
    "processor_type": (
        "Loads a multimodal processor for image/audio/video datasets. The Studio's default "
        "conversation builder is text-only, so known Qwen/Gemma text templates leave this unset."
    ),
    "chat_template": (
        "Top-level architecture-aware message formatter. Qwen 3.5 and 3.6 use qwen3_5; "
        "Gemma 4 uses gemma4."
    ),
    "eot_tokens": (
        "Extra end-of-turn tokens Axolotl should recognize. Gemma 4 needs <turn|> so assistant "
        "turn boundaries are trained correctly."
    ),
    "val_set_size": (
        "Fraction held out for validation during training. Around 0.05–0.1 is useful once the "
        "dataset is large enough; keep separate held-out prompts for the final test too."
    ),
    "output_dir": (
        "Directory for checkpoints and the final adapter. Use a unique project path so one run "
        "cannot overwrite another."
    ),
    "lora_target_linear": (
        "Applies LoRA to the model's linear layers without architecture-specific module names. "
        "This is the portable guided default, but multimodal Qwen 3.5/3.6 and Gemma 4 templates "
        "replace it with explicit text-backbone targets."
    ),
    "lora_target_modules": (
        "The exact model modules that receive adapter weights. Qwen hybrid models include both "
        "normal and Gated DeltaNet projections; Gemma 4 uses a regex to avoid its media encoders."
    ),
    "lora_target_parameters": (
        "Targets raw parameter tensors such as routed MoE experts. The Qwen A3B beginner template "
        "leaves these frozen; add expert targets only when you intentionally need expert LoRA."
    ),
    "quantize_moe_experts": (
        "Quantizes frozen routed experts in a QLoRA MoE run. It reduces memory for Qwen A3B but "
        "must not be paired with the generic lora_target_linear switch."
    ),
    "load_in_8bit": (
        "Loads base weights in 8-bit for a standard guided LoRA run. Disable it when QLoRA's "
        "4-bit loading is enabled."
    ),
    "load_in_4bit": (
        "Loads base weights in 4-bit to minimize VRAM. Pair it with adapter=qlora, never with "
        "8-bit loading."
    ),
    "pad_to_sequence_len": (
        "Pads prepared batches to a consistent length. It can improve kernel efficiency but may "
        "waste memory when example lengths vary widely."
    ),
    "warmup_steps": (
        "Uses small optimizer updates at the beginning to reduce instability. Ten steps is a "
        "conservative small-run default."
    ),
    "weight_decay": (
        "Regularizes trainable weights. About 0.01 is a common default; high values can suppress "
        "the small LoRA update."
    ),
    "save_strategy": (
        "Controls when recoverable checkpoints are written. Epoch-based saving is easiest for "
        "small datasets; step-based saving is better for long runs."
    ),
    "eval_strategy": (
        "Controls when the validation split is measured. Match it to the save strategy if you "
        "intend to choose the best checkpoint."
    ),
}

SETUP_RECIPES: dict[str, dict[str, Any]] = {
    "LoRA SFT starter": {
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
        "lr_scheduler": "cosine",
        "bf16": "auto",
        "fp16": False,
        "gradient_checkpointing": "true",
        "strict": False,
    },
    "QLoRA 4-bit starter": {
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
        "learning_rate": 0.0002,
        "optimizer": "paged_adamw_8bit",
        "bf16": "auto",
        "gradient_checkpointing": "true",
        "strict": False,
    },
    "Chat template SFT": {
        "datasets.0.type": "chat_template",
        "datasets.0.field_messages": "messages",
        "datasets.0.chat_template": "tokenizer_default",
        "datasets.0.train_on_eos": "turn",
        "sample_packing": True,
        "pad_to_sequence_len": True,
    },
    "Local completion JSONL": {
        "datasets.0.path": "./data/train.jsonl",
        "datasets.0.type": "completion",
        "datasets.0.ds_type": "json",
        "datasets.0.field": "text",
        "datasets.0.split": "train",
        "sample_packing": True,
    },
}


@dataclass
class AppState:
    config_store: ConfigStore
    telemetry: TelemetrySampler
    hf: HuggingFaceManager
    ollama: OllamaManager
    runner: AxolotlRunner
    workflow: WorkflowManager
    preflight: AxolotlPreflight
    preflight_issues: list[PreflightIssue] = field(default_factory=list)
    resource_tick: int = 0
    lora_base_response: str = ""
    lora_tuned_response: str = ""
    lora_base_response_meta: str = "Not tested yet."
    lora_tuned_response_meta: str = "Not tested yet."

    def refresh_preflight(self) -> list[PreflightIssue]:
        try:
            cfg = self.config_store.load()
        except ConfigError as exc:
            self.preflight_issues = [PreflightIssue("error", "YAML", str(exc))]
            return self.preflight_issues
        self.preflight_issues = self.preflight.validate(cfg)
        if self.runner.axolotl_path is None:
            self.preflight_issues = [
                PreflightIssue(
                    "error",
                    "Axolotl CLI",
                    "The axolotl executable is not on PATH. Install Axolotl in this environment before launching runs.",
                ),
                *[
                    issue
                    for issue in self.preflight_issues
                    if not (issue.severity == "ok" and issue.check == "Launch Gate")
                ],
            ]
        return self.preflight_issues


CONFIG_STORE = ConfigStore(PROJECT_ROOT)
STATE = AppState(
    config_store=CONFIG_STORE,
    telemetry=TelemetrySampler(),
    hf=HuggingFaceManager(),
    ollama=OllamaManager(),
    runner=AxolotlRunner(PROJECT_ROOT),
    workflow=WorkflowManager(CONFIG_STORE.active_name),
    preflight=AxolotlPreflight(PROJECT_ROOT, OllamaManager()),
)
STATE.preflight = AxolotlPreflight(PROJECT_ROOT, STATE.ollama)
UI_STATE = UiStateStore(PROJECT_ROOT)


def _detected_gpu_vram_gb() -> float | None:
    snapshot = STATE.telemetry.latest
    if snapshot is None or not snapshot.gpus:
        return None
    total = snapshot.gpus[0].memory_total
    return total / (1024**3) if total > 0 else None


def _detected_gpu_label() -> str:
    snapshot = STATE.telemetry.latest
    if snapshot is None or not snapshot.gpus:
        return "GPU VRAM not detected · using the balanced safe default"
    gpu = snapshot.gpus[0]
    return f"{gpu.name} · {gpu.memory_total / (1024**3):.1f} GB VRAM detected"


def _persisted_widget_defaults() -> dict[str, Any]:
    """Typed fallbacks for every preference mirrored outside the active YAML."""

    try:
        cfg = STATE.config_store.load()
    except ConfigError:
        cfg = {}
    project_name = Path(STATE.config_store.active_name).stem
    base_model = str(cfg.get("base_model") or LORA_BASE_MODEL_VALUES[0])
    recommended_preset = recommend_lora_preset(
        _detected_gpu_vram_gb(),
        base_model,
    )
    ollama_name = (
        STATE.ollama.selected.name
        if STATE.ollama.selected is not None
        else (STATE.ollama.models[0].name if STATE.ollama.models else "")
    )
    suggested_base = suggested_ollama_model(
        base_model,
        [model.name for model in STATE.ollama.models],
    )
    artifacts = discover_adapter_artifacts(PROJECT_ROOT, cfg, limit=1)
    adapter_path = str(artifacts[0].path) if artifacts else str(cfg.get("output_dir") or "")
    test_model_name = f"{project_name}-lora"
    installed_names = {model.name for model in STATE.ollama.models}
    chat_model = test_model_name if test_model_name in installed_names else suggested_base
    return {
        "active-config-select": STATE.config_store.active_name,
        "new-config-name": "experiment.yml",
        "lora-project-name": project_name,
        "lora-goal": LORA_GOALS[0],
        "lora-base-model": base_model,
        "lora-preset": recommended_preset,
        "lora-data-filename": f"{project_name}.jsonl",
        "lora-test-base-model": suggested_base,
        "lora-test-adapter-path": adapter_path,
        "lora-test-model-name": test_model_name,
        "lora-test-chat-model": chat_model,
        "lora-test-compare-base": suggested_base,
        "lora-test-system": "",
        "lora-test-prompt": "Introduce yourself and explain how you would help me.",
        "setup-recipe": next(iter(SETUP_RECIPES)),
        "setup-model-preset": MODEL_PRESETS[0],
        "setup-dataset-preset": next(iter(DATASET_PRESETS)),
        "run-action": "train",
        "run-launcher": "",
        "run-cli-args": "",
        "run-launcher-args": "",
        "hf-query": "llama instruct",
        "hf-query-mode": HF_QUERY_MODE_OPTIONS[0],
        "hf-search-repo-type": STATE.hf.last_repo_type,
        "hf-repo-type": STATE.hf.last_repo_type,
        "hf-sort": HF_SORT_OPTIONS[0],
        "hf-compatibility": HF_COMPATIBILITY_OPTIONS[0],
        "hf-limit": HF_LIMIT_OPTIONS[0],
        "hf-vram-limit": float(STATE.hf.vram_limit_gb or 24),
        "hf-sift": "",
        "hf-artifact-filter": HF_ARTIFACT_FILTER_OPTIONS[0],
        "hf-quant-filter": HF_QUANT_FILTER_OPTIONS[0],
        "hf-fit-filter": HF_FIT_FILTER_OPTIONS[0],
        HF_RESULTS_PAGE_KEY: "1",
        HF_RESULTS_PAGE_SIZE_KEY: str(HF_RESULTS_PAGE_SIZE),
        "hf-repo-id": STATE.hf.last_repo_id,
        "hf-revision": "",
        "delete-repo-id": STATE.hf.last_repo_id,
        "delete-repo-type": STATE.hf.last_repo_type,
        "ollama-model-name": ollama_name,
    }


def _persisted_widget_choices() -> dict[str, tuple[str, ...]]:
    ollama_names = tuple(model.name for model in STATE.ollama.models)
    try:
        current_base = str(STATE.config_store.load().get("base_model") or "").strip()
    except ConfigError:
        current_base = ""
    lora_base_choices = (
        (*LORA_BASE_MODEL_VALUES, current_base)
        if current_base and current_base not in LORA_BASE_MODEL_VALUES
        else LORA_BASE_MODEL_VALUES
    )
    return {
        "active-config-select": tuple(STATE.config_store.list_configs()),
        "lora-goal": tuple(LORA_GOALS),
        "lora-base-model": lora_base_choices,
        "lora-preset": tuple(LORA_PRESET_KEYS),
        "lora-test-base-model": ("", *ollama_names),
        "lora-test-chat-model": ("", *ollama_names),
        "lora-test-compare-base": ("", *ollama_names),
        "setup-recipe": tuple(SETUP_RECIPES),
        "setup-model-preset": tuple(MODEL_PRESETS),
        "setup-dataset-preset": tuple(DATASET_PRESETS),
        "run-action": tuple(AXOLOTL_ACTIONS),
        "run-launcher": ("", "python", "accelerate", "torchrun"),
        "hf-query-mode": tuple(HF_QUERY_MODE_OPTIONS),
        "hf-search-repo-type": ("model", "dataset"),
        "hf-repo-type": ("model", "dataset"),
        "hf-sort": tuple(HF_SORT_OPTIONS),
        "hf-compatibility": tuple(HF_COMPATIBILITY_OPTIONS),
        "hf-limit": tuple(HF_LIMIT_OPTIONS),
        "hf-artifact-filter": tuple(HF_ARTIFACT_FILTER_OPTIONS),
        "hf-quant-filter": tuple(HF_QUANT_FILTER_OPTIONS),
        "hf-fit-filter": tuple(HF_FIT_FILTER_OPTIONS),
        HF_RESULTS_PAGE_SIZE_KEY: tuple(str(size) for size in HF_RESULTS_PAGE_SIZES),
        "delete-repo-type": ("model", "dataset"),
    }


def _normalized_persisted_widget_value(
    widget_id: str,
    value: Any,
    *,
    defaults: dict[str, Any] | None = None,
    choices: dict[str, tuple[str, ...]] | None = None,
) -> Any:
    """Reject stale or malformed preferences before they reach LCARS controls."""

    defaults = defaults or _persisted_widget_defaults()
    choices = choices or _persisted_widget_choices()
    default = defaults[widget_id]
    if widget_id == "hf-vram-limit":
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(number):
            return float(default)
        return max(1.0, min(256.0, number))
    if widget_id == HF_RESULTS_PAGE_KEY:
        try:
            page = int(str(value).strip())
        except (TypeError, ValueError):
            return str(default)
        return str(max(1, min(1000, page)))
    if widget_id in choices:
        selected = str(value) if value is not None else str(default)
        return selected if selected in choices[widget_id] else default
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return str(default)
    text = str(value)
    limits = {
        "new-config-name": 128,
        "lora-project-name": 64,
        "lora-data-filename": 128,
        "lora-test-adapter-path": 1024,
        "lora-test-model-name": 256,
        "lora-test-system": 8192,
        "lora-test-prompt": 8192,
        "run-cli-args": 4096,
        "run-launcher-args": 4096,
        "hf-query": 512,
        "hf-sift": 512,
        "hf-repo-id": 512,
        "hf-revision": 256,
        "delete-repo-id": 512,
        "ollama-model-name": 512,
    }
    if len(text) > limits.get(widget_id, 1024):
        return str(default)
    if widget_id == "new-config-name" and (
        not text.endswith((".yml", ".yaml")) or "/" in text or "\\" in text
    ):
        return str(default)
    if widget_id == "lora-data-filename" and (
        not text.endswith(".jsonl") or "/" in text or "\\" in text
    ):
        return str(default)
    return text


def _restore_persisted_state() -> None:
    """Replay saved selections into the managers that own them."""

    active = str(UI_STATE.get("active_config", "") or "")
    if active:
        try:
            STATE.config_store.set_active(active)
        except ConfigError:
            UI_STATE.set("active_config", STATE.config_store.active_name)

    saved_widgets = UI_STATE.widget_values()
    repo_type = str(
        _normalized_persisted_widget_value(
            "hf-repo-type",
            UI_STATE.get(
                "hf_repo_type",
                saved_widgets.get("hf-repo-type", STATE.hf.last_repo_type),
            ),
        )
    )
    if repo_type in {"model", "dataset"}:
        STATE.hf.last_repo_type = repo_type  # type: ignore[assignment]
    STATE.hf.last_repo_id = str(UI_STATE.get("hf_repo_id", "") or "")
    STATE.hf.local_sort = str(
        UI_STATE.get("hf_local_sort", STATE.hf.local_sort) or STATE.hf.local_sort
    )
    STATE.hf.local_sort_desc = bool(UI_STATE.get("hf_local_sort_desc", STATE.hf.local_sort_desc))
    expanded = UI_STATE.get("hf_expanded_result_ids", [])
    if isinstance(expanded, list):
        STATE.hf.set_expanded_result_ids(
            [str(row_id) for row_id in expanded if isinstance(row_id, str)]
        )
    vram = UI_STATE.get("hf_vram_limit")
    if isinstance(vram, (int, float)) and vram > 0:
        STATE.hf.vram_limit_gb = float(vram)

    saved_workflow = UI_STATE.get("workflow_document")
    try:
        STATE.workflow = WorkflowManager(
            STATE.config_store.active_name,
            saved_workflow if isinstance(saved_workflow, dict) else None,
        )
    except Exception:
        STATE.workflow = WorkflowManager(STATE.config_store.active_name)
        UI_STATE.set(
            "workflow_document",
            STATE.workflow.document.model_dump(mode="json"),
        )


_restore_persisted_state()
STATE.telemetry.sample()
STATE.ollama.refresh()
STATE.refresh_preflight()


def build_ui() -> None:
    """Build the static LCARS manifest and handle rerun actions."""

    _hydrate_widget_state()

    lcars.config(
        "AXOLOTL LCARS",
        theme="galaxy",
        subtitle="CONFIGURATION / TELEMETRY / CONTENT OPS",
        header_color="tanoi",
        sound_enabled=True,
    )

    lcars.nav("LoRA Studio", page="lora", color="pale-canary")
    lcars.nav("1 · Setup", page="lora-setup", color="tanoi")
    lcars.nav("2 · Data", page="lora-data", color="golden-tanoi")
    lcars.nav("3 · Train", page="lora-train", color="red")
    lcars.nav("4 · Test", page="lora-test", color="lilac")
    lcars.nav("Command", page="command", color="tanoi")
    lcars.nav("Config", page="config", color="golden-tanoi")
    lcars.nav("Setup", page="config-setup", color="pale-canary")
    lcars.nav("Train", page="config-train", color="tanoi")
    lcars.nav("Hardware", page="config-hardware", color="anakiwa")
    lcars.nav("Tracking", page="config-tracking", color="lilac")
    lcars.nav("Advanced", page="config-advanced", color="blue-bell")
    lcars.nav("Workflow", page="run", color="red")
    lcars.nav("Console", page="console", color="hopbush")
    lcars.nav("Resources", page="resources", color="anakiwa")
    lcars.nav("HF Hub", page="hub", color="lilac")
    lcars.nav("Content", page="content", color="blue-bell")
    lcars.nav("Ollama", page="ollama", color="pale-canary")

    _lora_home_page()
    _lora_setup_page()
    _lora_data_page()
    _lora_train_page()
    _lora_test_page()
    _command_page()
    _config_page()
    _config_setup_page()
    _config_train_page()
    _config_hardware_page()
    _config_tracking_page()
    _config_advanced_page()
    _run_page()
    _console_page()
    _resources_page()
    _hub_page()
    _content_page()
    _ollama_page()


def _lora_home_page() -> None:
    cfg = _load_config_or_empty()
    dataset = inspect_configured_dataset(PROJECT_ROOT, cfg)
    artifacts = discover_adapter_artifacts(PROJECT_ROOT, cfg)
    steps = _lora_journey_rows(cfg, dataset, bool(artifacts))
    completed = sum(1 for row in steps if row["Status"] == "READY")

    with lcars.page("LoRA Studio", id="lora", layout="grid", fillers=False):
        with lcars.data_panel(
            "Your First LoRA",
            color="pale-canary",
            id="lora-home-intro-panel",
            weight=12,
            aspect="wide",
            group="lora-home",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.header(
                "Teach a small behavior layer—not a whole model",
                size="h2",
                color="pale-canary",
                id="lora-home-title",
            )
            lcars.markdown(
                "A **LoRA** is a compact set of learned changes attached to a base model. "
                "Use it for a repeatable voice, response style, decision pattern, or narrow skill. "
                "You do not need to understand rank, alpha, quantization, or batch math to start: "
                "the Setup page recommends a safe recipe for the detected GPU. The four guided "
                "pages still write a normal Axolotl config, so every advanced field remains "
                "available when you are ready.",
                id="lora-home-explainer",
            )
            lcars.progress(
                "Guided journey",
                completed / len(steps) * 100,
                color="tanoi",
                id="lora-home-progress",
                options=lcars.MeterOptions(
                    unit="%",
                    segments=4,
                    ticks=True,
                    description=f"{completed} of {len(steps)} stages ready",
                ),
            )
            _enhanced_table(
                steps,
                title="Four Steps",
                id="lora-journey-table",
                filter_columns={"Step", "Status"},
            )
            lcars.markdown(
                "[Start with setup](?page=lora-setup) · "
                "[Open training data](?page=lora-data) · "
                "[Monitor training](?page=lora-train) · "
                "[Test with Ollama](?page=lora-test)",
                id="lora-home-links",
                options=lcars.MarkdownOptions(link_target="_self"),
            )

        with lcars.data_panel(
            "What You Can Teach",
            color="tanoi",
            id="lora-home-teach-panel",
            weight=8,
            aspect="wide",
            group="lora-concepts",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                [
                    {
                        "Goal": "Personality",
                        "LoRA learns": "Voice, tone, phrasing, boundaries, conversational habits",
                        "Runtime still supplies": "Facts, context window, memory",
                    },
                    {
                        "Goal": "Agent behavior",
                        "LoRA learns": "Planning style, when to ask, tool-call patterns, recovery habits",
                        "Runtime still supplies": "Actual tools, permissions, memory, execution loop",
                    },
                    {
                        "Goal": "Narrow skill",
                        "LoRA learns": "A repeated input → ideal-output pattern",
                        "Runtime still supplies": "General reasoning and outside knowledge",
                    },
                ],
                title="LoRA Boundaries",
                id="lora-teaching-table",
                filter_columns={"Goal"},
            )

        with lcars.data_panel(
            "Before Spending GPU Time",
            color="golden-tanoi",
            id="lora-home-advice-panel",
            weight=7,
            aspect="wide",
            group="lora-concepts",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "1. Try the behavior as a **system prompt** first.\n"
                "2. Fine-tune only when you have examples of the exact answers you want.\n"
                "3. Prefer a smaller test run before a long run.\n"
                "4. Keep a few prompts out of training so you can test generalization.\n\n"
                "LoRA improves consistency; it does not verify facts or make unsafe tool access safe.",
                id="lora-home-advice",
            )


def _lora_setup_page() -> None:
    cfg = _load_config_or_empty()
    current_base = str(cfg.get("base_model") or LORA_BASE_MODEL_VALUES[0])
    base_options = [
        lcars.SelectOption(
            label=label,
            value=value,
            description=LORA_BASE_MODEL_HINTS.get(value),
        )
        for label, value in LORA_BASE_MODELS
    ]
    if current_base and current_base not in LORA_BASE_MODEL_VALUES:
        base_options.insert(
            0,
            lcars.SelectOption(
                label=f"{current_base} · current custom model",
                value=current_base,
                description=(
                    "Preserved from the active YAML. Confirm its architecture, access terms, "
                    "chat template, and GPU fit before training."
                ),
            ),
        )
    project_default = Path(STATE.config_store.active_name).stem
    _seed_text("lora-project-name", project_default)
    recommended_key = recommend_lora_preset(
        _detected_gpu_vram_gb(),
        _widget_value("lora-base-model", current_base),
    )
    preset_default = _widget_value(
        "lora-preset",
        infer_lora_preset(cfg) if cfg else recommended_key,
    )
    if preset_default not in LORA_PRESET_KEYS:
        preset_default = recommended_key

    with lcars.page("LoRA Setup", id="lora-setup", layout="grid", fillers=False):
        with lcars.control_panel(
            "Make My LoRA",
            color="tanoi",
            id="lora-setup-wizard-panel",
            zone="full",
            span=(4, 7),
            weight=12,
            aspect="wide",
            group="lora-setup-wizard",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "**The choices are already safe.** Give the project a name, pick what it should "
                "learn, and press the button. The smart preset fills the adapter, rank, context, "
                "batch, optimizer, precision, checkpoints, and output paths.",
                id="lora-setup-help",
            )
            with lcars.form(
                "LoRA Project",
                action_id="lora-setup-save",
                submit_label="Create Project With Smart Defaults",
                id="lora-setup-form",
                color="tanoi",
                options=lcars.FormOptions(layout="grid", columns=2),
            ):
                project_name = lcars.text_input(
                    "Project Name",
                    value=project_default,
                    placeholder="helpful-captain",
                    autocomplete=False,
                    id="lora-project-name",
                    options=lcars.TextInputOptions(
                        description="Becomes configs/name.yml, data/name.jsonl, and outputs/name.",
                        validation=lcars.ValidationOptions(
                            required=True,
                            pattern=r"^[A-Za-z0-9][A-Za-z0-9 _-]{1,62}$",
                            message="Use 2–63 letters, numbers, spaces, dashes, or underscores.",
                        ),
                    ),
                    hint=(
                        "This safe project slug names the starter config, local JSONL dataset, "
                        "training output directory, and eventual Ollama adapter."
                    ),
                )
                goal = lcars.radio(
                    "What Are You Teaching?",
                    [
                        lcars.SelectOption(
                            label=item,
                            value=item,
                            description=LORA_GOAL_HINTS[item],
                        )
                        for item in LORA_GOALS
                    ],
                    value=LORA_GOALS[0],
                    id="lora-goal",
                    settings=lcars.ChoiceOptions(
                        description=(
                            "This selects the starter-example template. It does not lock the "
                            "adapter to one narrow capability."
                        )
                    ),
                    hint=(
                        "Choose the closest teaching goal. It changes the starter examples and "
                        "guidance, while the resulting LoRA can still learn varied behavior."
                    ),
                )
                base_model = lcars.select(
                    "Base Model Template",
                    base_options,
                    value=current_base,
                    id="lora-base-model",
                    settings=lcars.ChoiceOptions(
                        searchable=True,
                        description=(
                            "Known Qwen 3.5/3.6 and Gemma 4 choices also apply their required chat "
                            "format and safe text-backbone LoRA targets automatically."
                        ),
                    ),
                    hint=(
                        "Known templates safely coordinate chat format, architecture-specific target "
                        "modules, and the recommended first recipe."
                    ),
                )
                recommended_key = recommend_lora_preset(
                    _detected_gpu_vram_gb(),
                    str(base_model),
                )
                preset_key = lcars.select(
                    "How Should It Train?",
                    _lora_preset_options(recommended_key),
                    value=preset_default,
                    id="lora-preset",
                    settings=lcars.ChoiceOptions(
                        description=(
                            "Start with the recommended recipe. Change one setting only after a "
                            "measured problem such as out-of-memory, truncation, or underfitting."
                        )
                    ),
                    hint=(
                        "A smart preset sets quantization, rank, context, effective batch, optimizer, "
                        "precision, checkpoints, and output paths as one coherent recipe."
                    ),
                )
            if _is_active_action("lora-setup-save"):
                _lora_setup_action(project_name, goal, base_model, preset_key)
            chosen_preset = get_lora_preset(
                preset_key if preset_key in LORA_PRESET_KEYS else DEFAULT_LORA_PRESET
            )
            lcars.text(
                f"Selected: {chosen_preset.label}. {chosen_preset.summary}",
                id="lora-setup-selected-summary",
                options=lcars.TextOptions(
                    description=(
                        f"Best for: {chosen_preset.best_for}. "
                        f"Hardware guidance: {chosen_preset.hardware}."
                    ),
                    wrap="wrap",
                ),
            )
            chosen_model_template = get_lora_model_template(str(base_model))
            if chosen_model_template is not None:
                lcars.text(
                    (
                        f"Model template: {chosen_model_template.family} · "
                        f"{chosen_model_template.architecture}. "
                        f"{chosen_model_template.summary}"
                    ),
                    id="lora-setup-model-summary",
                    options=lcars.TextOptions(
                        description=(
                            f"Training scope: text/chat adapter. "
                            f"Minimum supported Axolotl: {chosen_model_template.min_axolotl}."
                        ),
                        wrap="wrap",
                    ),
                )

        with lcars.data_panel(
            "Active Project At A Glance",
            color="pale-canary",
            id="lora-setup-plan-panel",
            zone="full",
            span=(4, 1),
            weight=10,
            aspect="wide",
            group="lora-setup-active",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                _lora_training_plan_rows(cfg),
                title="Plain-Language Plan",
                id="lora-setup-plan-table",
                filter_columns={"Choice", "Why"},
                copy_columns={"Value"},
            )
            lcars.markdown(
                "[Next: write training examples](?page=lora-data) · "
                "[Browse base models](?page=hub) · "
                "[Open every Axolotl field](?page=config)",
                id="lora-setup-links",
                options=lcars.MarkdownOptions(link_target="_self"),
            )

        with lcars.data_panel(
            "Qwen 3.5 / 3.6 + Gemma 4 Model Templates",
            color="lilac",
            id="lora-model-template-panel",
            zone="full",
            span=(4, 3),
            weight=12,
            aspect="wide",
            group="lora-model-templates",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "These are **text/chat LoRA defaults** for the official post-trained checkpoints. "
                "They deliberately avoid training vision/audio encoders. Choosing one in the form "
                "applies its Axolotl chat template, turn tokens, target modules, and MoE guardrails.",
                id="lora-model-template-help",
            )
            _enhanced_table(
                _lora_model_template_rows(),
                title="Architecture-Aware Defaults",
                id="lora-model-template-table",
                filter_columns={"Family", "Model", "First recipe"},
                copy_columns={"Model"},
            )

        with lcars.data_panel(
            "Smart Presets",
            color="anakiwa",
            id="lora-preset-help-panel",
            zone="full",
            span=(4, 2),
            weight=12,
            aspect="wide",
            group="lora-presets",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            recommended = get_lora_preset(recommended_key)
            lcars.metric(
                "Recommended",
                recommended.label,
                status="ok",
                color="anakiwa",
                id="lora-preset-recommendation",
                options=lcars.MetricOptions(
                    secondary_value=_detected_gpu_label(),
                ),
            )
            _enhanced_table(
                _lora_preset_rows(recommended_key),
                title="Pick By Outcome, Not Acronym",
                id="lora-preset-table",
                filter_columns={"Preset", "Method", "Fit"},
            )

        with lcars.data_panel(
            "What To Tune—And When",
            color="lilac",
            id="lora-tuning-help-panel",
            zone="full",
            span=(4, 2),
            weight=12,
            aspect="wide",
            group="lora-tuning",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "Leave these alone for the first real run. Change **one variable at a time**, "
                "compare on held-out prompts, and keep the old config so you can undo the change.",
                id="lora-tuning-help",
            )
            _enhanced_table(
                _lora_tuning_rows(cfg),
                title="Beginner Tuning Guide",
                id="lora-tuning-table",
                filter_columns={"Setting", "Current", "Starter range"},
            )


def _lora_data_page() -> None:
    cfg = _load_config_or_empty()
    report = inspect_configured_dataset(PROJECT_ROOT, cfg)
    dataset_cache_rows = _lora_dataset_cache_rows()
    cached_datasets = _lora_downloaded_dataset_cache_rows(dataset_cache_rows)
    cached_dataset_ids = tuple(row["Repo"] for row in cached_datasets)
    configured_dataset = str(_config_path_value(cfg, "datasets.0.path") or "").strip()
    downloaded_dataset_default = _widget_value(
        "lora-hf-dataset",
        (
            configured_dataset
            if configured_dataset in cached_dataset_ids
            else (cached_dataset_ids[0] if cached_dataset_ids else "")
        ),
    )
    if downloaded_dataset_default not in cached_dataset_ids:
        downloaded_dataset_default = cached_dataset_ids[0] if cached_dataset_ids else ""
    downloaded_format_default = _widget_value(
        "lora-hf-dataset-format",
        _infer_lora_hf_dataset_format(cfg),
    )
    if downloaded_format_default not in LORA_HF_DATASET_FORMAT_KEYS:
        downloaded_format_default = LORA_HF_DATASET_FORMAT_KEYS[0]
    split_default = str(_config_path_value(cfg, "datasets.0.split") or "train")
    subset_default = str(_config_path_value(cfg, "datasets.0.name") or "")
    _seed_text("lora-hf-dataset-split", split_default)
    _seed_text("lora-hf-dataset-subset", subset_default)
    project_name = _widget_value(
        "lora-project-name",
        Path(STATE.config_store.active_name).stem,
    )
    goal = _widget_value("lora-goal", LORA_GOALS[0])
    filename_default = f"{project_name}.jsonl"
    _seed_text("lora-data-filename", filename_default)
    _seed_text(
        "lora-data-editor",
        starter_dataset_template(goal, project_name.replace("-", " ").title()),
    )

    with lcars.page("LoRA Data", id="lora-data", layout="grid", fillers=False):
        with lcars.data_panel(
            "Current Training Dataset",
            color="golden-tanoi",
            id="lora-data-status-panel",
            zone="full",
            span=(4, 5),
            weight=10,
            aspect="wide",
            group="lora-data",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "**This is the one dataset the Train page will use.** Choose either the downloaded "
                "dataset route or the local example builder below; selecting one replaces the "
                "other as the active source.",
                id="lora-data-current-help",
            )
            lcars.text(
                _lora_active_dataset_summary(cfg, report, cached_dataset_ids),
                id="lora-data-active-source",
                options=lcars.TextOptions(
                    description=(
                        "This summary comes directly from datasets[0] in the active YAML."
                    ),
                    selectable=True,
                    copyable=True,
                    wrap="wrap",
                ),
            )
            lcars.metric(
                "Status",
                _lora_dataset_status(cfg, report),
                status=(
                    "ok"
                    if _lora_dataset_trainable(cfg, report)
                    else ("crit" if report.errors else "warn")
                ),
                color="golden-tanoi",
                id="lora-data-status",
                options=lcars.MetricOptions(secondary_value=report.source or "No source"),
            )
            _enhanced_table(
                _lora_dataset_issue_rows(report),
                title="Data Checks",
                id="lora-data-checks-table",
                filter_columns={"Level", "Detail"},
            )

        with lcars.control_panel(
            "Option A · Use A Downloaded Dataset",
            color="anakiwa",
            id="lora-downloaded-dataset-panel",
            zone="full",
            span=(4, 9),
            weight=11,
            aspect="wide",
            group="lora-data-downloaded",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "Datasets downloaded through **HF Hub** live in the Hugging Face cache. Pick one "
                "below and tell the Studio what one row looks like. The config uses its stable "
                "`owner/name`; Hugging Face reuses the downloaded files automatically. You do "
                "**not** paste downloaded data into the JSONL editor.",
                id="lora-downloaded-dataset-help",
            )
            _enhanced_table(
                _lora_dataset_download_rows(dataset_cache_rows, configured_dataset),
                title="Downloaded / In-Progress Datasets",
                id="lora-downloaded-dataset-table",
                filter_columns={"Dataset", "Status", "Revision", "What to do"},
                copy_columns={"Dataset", "Cache path"},
            )
            with lcars.form(
                "Use Downloaded Dataset",
                action_id="lora-use-downloaded-dataset",
                submit_label="Use This Downloaded Dataset",
                id="lora-downloaded-dataset-form",
                color="anakiwa",
                disabled=not bool(cached_datasets),
                options=lcars.FormOptions(
                    layout="grid",
                    columns=2,
                    description=(
                        "This changes only the active YAML dataset source and format. "
                        "It does not copy or delete cached files."
                    ),
                ),
            ):
                downloaded_dataset = lcars.select(
                    "Downloaded Dataset",
                    _lora_downloaded_dataset_options(cached_datasets),
                    value=downloaded_dataset_default,
                    id="lora-hf-dataset",
                    disabled=not bool(cached_datasets),
                    settings=lcars.ChoiceOptions(
                        searchable=True,
                        description=(
                            "Only completed Hugging Face dataset downloads appear here."
                        ),
                    ),
                    hint=(
                        "This list includes only complete, readable dataset snapshots from the "
                        "standard Hugging Face cache; interrupted downloads stay unavailable."
                    ),
                )
                downloaded_format = lcars.select(
                    "What Does One Row Look Like?",
                    _lora_hf_dataset_format_options(),
                    value=downloaded_format_default,
                    id="lora-hf-dataset-format",
                    settings=lcars.ChoiceOptions(
                        description=(
                            "Pick by column names, not by the subject of the dataset. "
                            "The descriptions show the expected fields."
                        )
                    ),
                    hint=(
                        "Inspect one dataset row and choose its actual column shape: OpenAI messages, "
                        "ShareGPT conversations, Alpaca instructions, or plain text."
                    ),
                )
                downloaded_split = lcars.text_input(
                    "Training Split",
                    value=split_default,
                    placeholder="train",
                    autocomplete=False,
                    id="lora-hf-dataset-split",
                    options=lcars.TextInputOptions(
                        description=(
                            "Usually train. Slices such as train[:10%] are useful for a quick run."
                        ),
                        validation=lcars.ValidationOptions(required=True),
                    ),
                    hint=(
                        "Use train for the full split, or a datasets slice such as train[:10%] for "
                        "a fast pipeline check before committing to the full corpus."
                    ),
                )
                downloaded_subset = lcars.text_input(
                    "Dataset Subset / Config · optional",
                    value=subset_default,
                    placeholder="Leave blank unless the dataset card names a subset",
                    autocomplete=False,
                    id="lora-hf-dataset-subset",
                    options=lcars.TextInputOptions(
                        description=(
                            "Some repositories contain named configurations such as default, "
                            "en, or cleaned. This is not the train/validation split."
                        )
                    ),
                    hint=(
                        "Only set this when the dataset card lists a named subset/configuration. "
                        "It is separate from the train, validation, or test split."
                    ),
                )
            if _is_active_action("lora-use-downloaded-dataset"):
                _lora_use_downloaded_dataset_action(
                    downloaded_dataset,
                    downloaded_format,
                    downloaded_split,
                    downloaded_subset,
                )
            lcars.alert(
                _lora_dataset_cache_notice(dataset_cache_rows, cached_datasets),
                level="yellow",
                id="lora-no-downloaded-datasets",
                visible=(
                    not bool(cached_datasets)
                    or any(
                        row.get("Status") == "INCOMPLETE"
                        for row in dataset_cache_rows
                    )
                ),
            )
            lcars.markdown(
                "[Find datasets in HF Hub](?page=hub) · "
                "[Watch downloads and cache](?page=content) · "
                "[Unusual columns? Open full Dataset settings](?page=setup)",
                id="lora-downloaded-dataset-links",
                options=lcars.MarkdownOptions(link_target="_self"),
            )

        with lcars.control_panel(
            "Option B · Build My Own Dataset",
            color="tanoi",
            id="lora-example-builder-panel",
            zone="full",
            span=(4, 10),
            weight=12,
            aspect="wide",
            group="lora-data-builder",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "Use this route when you want to author behavior examples yourself. Write one "
                "realistic prompt and the **exact answer you want the model to imitate**. Press "
                "**Add & Save Local Example**, then repeat. Saving here switches the active "
                "dataset from any Hub download to this project's local JSONL file.",
                id="lora-example-builder-help",
            )
            with lcars.form(
                "One Training Conversation",
                action_id="lora-example-add",
                submit_label="Add & Save Local Example",
                id="lora-example-form",
                color="tanoi",
                options=lcars.FormOptions(layout="grid", columns=2),
            ):
                example_user = lcars.text_input(
                    "What The User Says",
                    value="",
                    placeholder="Ask a realistic question or make a realistic request",
                    autocomplete=False,
                    id="lora-example-user",
                    options=lcars.TextInputOptions(
                        multiline=True,
                        rows=2,
                        description=(
                            "Use natural wording. Vary easy, hard, ambiguous, and edge-case prompts."
                        ),
                        validation=lcars.ValidationOptions(required=True),
                    ),
                )
                example_answer = lcars.text_input(
                    "Exact Ideal Answer",
                    value="",
                    placeholder="Write the full answer—not instructions such as “be warm”",
                    autocomplete=False,
                    id="lora-example-answer",
                    options=lcars.TextInputOptions(
                        multiline=True,
                        rows=3,
                        description=(
                            "This is the behavior the model copies. Include tone, reasoning style, "
                            "format, uncertainty, and boundaries directly in the response."
                        ),
                        validation=lcars.ValidationOptions(required=True),
                    ),
                )
                example_system = lcars.text_input(
                    "System Instruction · optional",
                    value="",
                    placeholder="Only add this when the example needs a system role",
                    autocomplete=False,
                    id="lora-example-system",
                    options=lcars.TextInputOptions(
                        multiline=True,
                        rows=1,
                        description=(
                            "Usually leave this blank; repeated system text can make the adapter "
                            "depend on it."
                        ),
                    ),
                )
            if _is_active_action("lora-example-add"):
                _lora_add_example_action(
                    project_name=project_name,
                    filename=_widget_value("lora-data-filename", filename_default),
                    user_prompt=example_user,
                    ideal_response=example_answer,
                    system_prompt=example_system,
                )
            lcars.text(
                "Your first form example replaces the untouched EDIT ME starter template. "
                "Later examples are appended. The raw editor below is optional and belongs only "
                "to this local-dataset route.",
                id="lora-example-builder-note",
            )

        with lcars.control_panel(
            "Option B Advanced · Raw JSONL",
            color="golden-tanoi",
            id="lora-data-editor-panel",
            zone="full",
            span=(4, 1),
            weight=2,
            aspect="wide",
            group="lora-data-advanced",
            options=COLLAPSED_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "Use this only when you want to bulk-edit or paste JSONL. Each line is one "
                "conversation. Replace every **EDIT ME** answer with the exact response the model "
                "should imitate.",
                id="lora-data-editor-help",
            )
            with lcars.form(
                "JSONL Dataset",
                action_id="lora-data-save",
                submit_label="Validate & Save Raw JSONL",
                id="lora-data-form",
                color="golden-tanoi",
            ):
                filename = lcars.text_input(
                    "Dataset Filename",
                    value=filename_default,
                    placeholder="helpful-captain.jsonl",
                    autocomplete=False,
                    id="lora-data-filename",
                    options=lcars.TextInputOptions(
                        validation=lcars.ValidationOptions(
                            required=True,
                            pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}\.jsonl$",
                            message="Use a simple .jsonl filename.",
                        )
                    ),
                )
                editor_text = lcars.text_input(
                    "One JSON Conversation Per Line",
                    value="",
                    placeholder='{"messages": [{"role": "user", "content": "..."}, ...]}',
                    autocomplete=False,
                    id="lora-data-editor",
                    options=lcars.TextInputOptions(
                        multiline=True,
                        rows=12,
                        commit="blur",
                        validation=lcars.ValidationOptions(required=True),
                    ),
                )
            if _is_active_action("lora-data-save"):
                _lora_save_dataset_action(filename, editor_text)
            if lcars.button(
                "Replace Editor With Fresh Template",
                color="lilac",
                id="lora-data-reset-template",
                options=lcars.ButtonOptions(
                    confirm="Replace the unsaved editor contents with a fresh EDIT ME template?"
                ),
            ):
                _lora_reset_dataset_template_action(goal, project_name)
            if lcars.button(
                "Validate Configured File",
                color="anakiwa",
                id="lora-data-validate",
            ):
                _lora_validate_dataset_action()
            lcars.text(
                "When overwriting an existing dataset, the previous version is kept beside it as "
                "filename.jsonl.bak.",
                id="lora-data-backup-note",
            )

        with lcars.data_panel(
            "Quality Beats Quantity",
            color="tanoi",
            id="lora-data-quality-panel",
            zone="full",
            span=(4, 2),
            weight=9,
            aspect="wide",
            group="lora-data-guide",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                [
                    {
                        "Check": "Ideal outputs",
                        "Good": "Every answer is something you genuinely want copied",
                        "Avoid": "Descriptions like “respond warmly” instead of the warm response",
                    },
                    {
                        "Check": "Variety",
                        "Good": "Different wording, difficulty, moods, successes, and failures",
                        "Avoid": "Near-duplicate prompts",
                    },
                    {
                        "Check": "Boundaries",
                        "Good": "Examples of uncertainty, refusal, correction, and tool failure",
                        "Avoid": "Only happy-path greetings",
                    },
                    {
                        "Check": "Evaluation",
                        "Good": "Keep several realistic prompts out of the training file",
                        "Avoid": "Testing only memorized training prompts",
                    },
                    {
                        "Check": "Privacy",
                        "Good": "Synthetic or authorized data with secrets removed",
                        "Avoid": "Passwords, tokens, private chats, or unlicensed content",
                    },
                ],
                title="Dataset Checklist",
                id="lora-data-quality-table",
                filter_columns={"Check"},
            )


def _lora_train_page() -> None:
    cfg = _load_config_or_empty()
    dataset = inspect_configured_dataset(PROJECT_ROOT, cfg)
    artifacts = discover_adapter_artifacts(PROJECT_ROOT, cfg)
    errors = [issue for issue in STATE.preflight_issues if issue.severity == "error"]
    dataset_trainable = _lora_dataset_trainable(cfg, dataset)
    can_train = not errors and dataset_trainable and not STATE.workflow.is_active
    snapshot = STATE.telemetry.latest or STATE.telemetry.sample()
    gpu = snapshot.gpus[0] if snapshot.gpus else None

    with lcars.page("LoRA Train", id="lora-train", layout="console", fillers=False):
        with lcars.data_panel(
            "Beginner Launch Gate",
            color="red",
            id="lora-train-gate-panel",
            zone="side",
            span=(2, 2),
            weight=8,
            aspect="tall",
            group="lora-training",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.metric(
                "Run Gate",
                "READY" if can_train else "BLOCKED",
                status="ok" if can_train else "crit",
                color="red",
                id="lora-train-gate",
                options=lcars.MetricOptions(
                    secondary_value=_lora_gate_detail(errors, dataset, cfg)
                ),
            )
            lcars.metric(
                "Dataset",
                _lora_dataset_status(cfg, dataset),
                status="ok" if dataset_trainable else "warn",
                color="golden-tanoi",
                id="lora-train-data-status",
                options=lcars.MetricOptions(
                    secondary_value=(
                        "remote"
                        if dataset.example_count is None
                        else f"{dataset.example_count} examples"
                    )
                ),
            )
            _enhanced_table(
                _lora_training_brief_rows(cfg, dataset),
                title="Training Plan",
                id="lora-train-plan-table",
                filter_columns={"Choice"},
                copy_columns={"Value"},
            )

        with lcars.data_panel(
            "Live Training Monitor",
            color="red",
            id="lora-train-monitor-panel",
            zone="primary",
            span=(4, 3),
            weight=12,
            aspect="wide",
            group="lora-training",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.metric(
                "Axolotl",
                STATE.runner.status_label(),
                status=STATE.runner.status_severity(),
                color="red",
                id="lora-train-status",
                options=lcars.MetricOptions(
                    secondary_value=STATE.config_store.active_name,
                ),
            )
            lcars.metric(
                "Elapsed",
                _lora_elapsed_text(),
                status="ok",
                color="tanoi",
                id="lora-train-elapsed",
                options=lcars.MetricOptions(secondary_value="current / most recent run"),
            )
            lcars.metric(
                "GPU",
                f"{gpu.utilization:.0f}%" if gpu is not None else "NOT DETECTED",
                status=(
                    _percent_status(gpu.utilization) if gpu is not None else "warn"
                ),
                color="anakiwa",
                id="lora-train-gpu",
                options=lcars.MetricOptions(
                    secondary_value=(
                        f"{format_bytes(gpu.memory_used)} / {format_bytes(gpu.memory_total)}"
                        if gpu is not None
                        else "training may be unavailable or CPU-only"
                    )
                ),
            )
            lcars.metric(
                "RAM",
                f"{snapshot.ram_percent:.0f}%",
                status=_percent_status(snapshot.ram_percent),
                color="lilac",
                id="lora-train-ram",
                options=lcars.MetricOptions(
                    secondary_value=(
                        f"{format_bytes(snapshot.ram_used)} / "
                        f"{format_bytes(snapshot.ram_total)}"
                    )
                ),
            )
            lcars.progress(
                "Training process",
                _lora_process_progress(),
                color="red",
                id="lora-train-progress",
                options=lcars.MeterOptions(
                    unit="%",
                    indeterminate=STATE.runner.is_running(),
                    description=(
                        "Axolotl's detailed step and loss output appears in the log below."
                    ),
                ),
            )
            lcars.log(
                LOG_AXOLOTL,
                max_lines=1000,
                title="Axolotl Training Output",
                id="lora-training-log",
                options=LOG_VIEW_OPTIONS,
            )

        with lcars.control_panel(
            "Train",
            color="golden-tanoi",
            id="lora-train-controls-panel",
            zone="dock",
            span=(4, 2),
            weight=9,
            aspect="wide",
            group="lora-training",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "**You only need Start Training.** It reruns every readiness check before launch. "
                "The other buttons are optional diagnostics.",
                id="lora-train-controls-help",
            )
            if lcars.button(
                "Start Training",
                color="red",
                id="lora-train-start",
                disabled=not can_train or STATE.runner.is_running(),
                options=lcars.ButtonOptions(
                    confirm=(
                        "Run the final safety checks and start LoRA training? "
                        "Training can run for a long time."
                    ),
                    debounce_ms=750,
                    busy_label="Starting",
                ),
            ):
                _lora_start_action("train")
            if lcars.button(
                "Check Readiness",
                color="anakiwa",
                id="lora-train-preflight",
            ):
                _run_preflight_action()
                _update_lora_widgets()
            if lcars.button(
                "Prepare Data Only · optional",
                color="golden-tanoi",
                id="lora-train-preprocess",
                disabled=not dataset_trainable or STATE.runner.is_running(),
                options=lcars.ButtonOptions(
                    confirm=(
                        "Run Axolotl preprocessing without starting training? "
                        "This is useful for inspecting tokenized samples."
                    ),
                    debounce_ms=750,
                    busy_label="Starting",
                ),
            ):
                _lora_start_action("preprocess")
            if lcars.button(
                "Stop Training",
                color="red",
                id="lora-train-stop",
                disabled=not STATE.runner.is_running(),
                options=lcars.ButtonOptions(
                    confirm="Ask the active Axolotl process to stop?",
                    debounce_ms=750,
                    busy_label="Stopping",
                ),
            ):
                _stop_axolotl_action()
            lcars.markdown(
                "[Fix the dataset](?page=lora-data) · "
                "[Open full console controls](?page=console) · "
                "[Inspect system resources](?page=resources)",
                id="lora-train-links",
                options=lcars.MarkdownOptions(link_target="_self"),
            )

        with lcars.data_panel(
            "Adapter Outputs",
            color="tanoi",
            id="lora-artifacts-panel",
            zone="primary",
            span=(4, 2),
            weight=10,
            aspect="wide",
            group="lora-artifacts",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                _lora_artifact_rows(artifacts),
                title="Detected Safetensors Adapters",
                id="lora-artifacts-table",
                filter_columns={"State", "Base Model"},
                copy_columns={"Path", "Base Model"},
            )
            lcars.markdown(
                "[Next: package and compare in Ollama](?page=lora-test)",
                id="lora-artifacts-next",
                options=lcars.MarkdownOptions(link_target="_self"),
            )


def _lora_test_page() -> None:
    cfg = _load_config_or_empty()
    artifacts = discover_adapter_artifacts(PROJECT_ROOT, cfg)
    model_names = [model.name for model in STATE.ollama.models]
    trained_base = str(cfg.get("base_model") or "")
    base_default = suggested_ollama_model(trained_base, model_names)
    adapter_default = str(artifacts[0].path) if artifacts else str(cfg.get("output_dir") or "")
    project_name = _widget_value(
        "lora-project-name",
        Path(STATE.config_store.active_name).stem,
    )
    tuned_default = f"{project_name}-lora"
    chat_default = tuned_default if tuned_default in model_names else base_default
    _seed_text("lora-test-adapter-path", adapter_default)
    _seed_text("lora-test-model-name", tuned_default)
    _seed_text("lora-test-system", "")
    _seed_text(
        "lora-test-prompt",
        "Introduce yourself and explain how you would help me.",
    )
    select_options = _lora_ollama_select_options(model_names)

    with lcars.page("LoRA Test", id="lora-test", layout="grid", fillers=False):
        with lcars.data_panel(
            "Ollama Adapter Gate",
            color="lilac",
            id="lora-test-gate-panel",
            weight=10,
            aspect="wide",
            group="lora-test",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.metric(
                "Ollama",
                "READY" if model_names else "NOT CONNECTED",
                status="ok" if model_names else "crit",
                color="lilac",
                id="lora-test-ollama-status",
                options=lcars.MetricOptions(
                    secondary_value=(
                        f"{len(model_names)} local model(s)"
                        if model_names
                        else STATE.ollama.last_error or "No local models detected"
                    )
                ),
            )
            lcars.metric(
                "Adapter",
                "FOUND" if artifacts else "NOT FOUND",
                status="ok" if artifacts else "warn",
                color="tanoi",
                id="lora-test-adapter-status",
                options=lcars.MetricOptions(
                    secondary_value=adapter_default or "Finish training first"
                ),
            )
            _enhanced_table(
                [
                    {
                        "Requirement": "Exact base",
                        "Your training value": trained_base,
                        "Why": "A different Ollama base can produce erratic behavior",
                    },
                    {
                        "Requirement": "Adapter format",
                        "Your training value": (
                            "Safetensors detected" if artifacts else "Not detected"
                        ),
                        "Why": "Ollama imports the PEFT adapter directory",
                    },
                    {
                        "Requirement": "Architecture",
                        "Your training value": _lora_architecture_hint(trained_base),
                        "Why": "Ollama documents adapter import for Llama, Mistral, and Gemma",
                    },
                ],
                title="Compatibility Checklist",
                id="lora-test-compatibility-table",
                filter_columns={"Requirement"},
                copy_columns={"Your training value"},
            )

        with lcars.control_panel(
            "1 · Build Tuned Ollama Model",
            color="lilac",
            id="lora-test-build-panel",
            weight=7,
            aspect="tall",
            group="lora-test",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "This writes a managed Modelfile and runs `ollama create`. Choose the **same base "
                "model lineage** used for training. "
                "[Ollama adapter import guide](https://docs.ollama.com/import)",
                id="lora-test-build-help",
                options=lcars.MarkdownOptions(link_target="_blank"),
            )
            with lcars.form(
                "Ollama Adapter Model",
                action_id="lora-test-build",
                submit_label="Build Test Model",
                id="lora-test-build-form",
                color="lilac",
            ):
                ollama_base = lcars.select(
                    "Installed Ollama Base",
                    select_options,
                    value=base_default,
                    id="lora-test-base-model",
                    disabled=not model_names,
                )
                adapter_path = lcars.text_input(
                    "Safetensors Adapter Directory",
                    value=adapter_default,
                    placeholder="./outputs/my-lora or checkpoint directory",
                    autocomplete=False,
                    id="lora-test-adapter-path",
                    options=lcars.TextInputOptions(
                        validation=lcars.ValidationOptions(required=True)
                    ),
                )
                model_name = lcars.text_input(
                    "New Ollama Model Name",
                    value=tuned_default,
                    placeholder="helpful-captain-lora",
                    autocomplete=False,
                    id="lora-test-model-name",
                    options=lcars.TextInputOptions(
                        validation=lcars.ValidationOptions(
                            required=True,
                            pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?$",
                            message="Enter a valid local Ollama name with an optional :tag.",
                        )
                    ),
                )
            if _is_active_action("lora-test-build"):
                _lora_build_ollama_action(
                    ollama_base,
                    adapter_path,
                    model_name,
                )
            if lcars.button(
                "Refresh Ollama Models",
                color="anakiwa",
                id="lora-test-refresh",
            ):
                _lora_refresh_ollama_action()

        with lcars.control_panel(
            "2 · Compare Base And LoRA",
            color="tanoi",
            id="lora-test-chat-panel",
            weight=9,
            aspect="tall",
            group="lora-compare",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "Use the same prompt and low temperature for both models. Leave the system prompt "
                "blank first to see what the adapter itself learned.",
                id="lora-test-chat-help",
            )
            with lcars.form(
                "Side-by-Side Test",
                action_id="lora-test-compare",
                submit_label="Run Side-by-Side Comparison",
                id="lora-test-compare-form",
                color="tanoi",
            ):
                compare_base = lcars.select(
                    "Base Model",
                    select_options,
                    value=base_default,
                    id="lora-test-compare-base",
                    disabled=not model_names,
                )
                tuned_model = lcars.select(
                    "Tuned Model",
                    select_options,
                    value=chat_default,
                    id="lora-test-chat-model",
                    disabled=not model_names,
                )
                system_prompt = lcars.text_input(
                    "Optional System Prompt",
                    value="",
                    placeholder="Leave blank to test what the LoRA learned by itself",
                    autocomplete=False,
                    id="lora-test-system",
                    options=lcars.TextInputOptions(multiline=True, rows=2),
                )
                test_prompt = lcars.text_input(
                    "Held-Out Test Prompt",
                    value="Introduce yourself and explain how you would help me.",
                    placeholder="Use a prompt that is not in the training file",
                    autocomplete=False,
                    id="lora-test-prompt",
                    options=lcars.TextInputOptions(
                        multiline=True,
                        rows=3,
                        validation=lcars.ValidationOptions(required=True),
                    ),
                )
            if _is_active_action("lora-test-compare"):
                _lora_compare_action(
                    compare_base,
                    tuned_model,
                    test_prompt,
                    system_prompt,
                )

        with lcars.data_panel(
            "Base Response",
            color="anakiwa",
            id="lora-base-response-panel",
            weight=10,
            aspect="wide",
            group="lora-responses",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            lcars.text(
                STATE.lora_base_response or "Run a comparison to see the unchanged base response.",
                size="body",
                id="lora-base-response",
                options=lcars.TextOptions(
                    description=STATE.lora_base_response_meta,
                    selectable=True,
                    copyable=bool(STATE.lora_base_response),
                    wrap="pre",
                ),
            )

        with lcars.data_panel(
            "LoRA Response",
            color="lilac",
            id="lora-tuned-response-panel",
            weight=10,
            aspect="wide",
            group="lora-responses",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            lcars.text(
                STATE.lora_tuned_response or "Build the adapter model, then run a comparison.",
                size="body",
                id="lora-tuned-response",
                options=lcars.TextOptions(
                    description=STATE.lora_tuned_response_meta,
                    selectable=True,
                    copyable=bool(STATE.lora_tuned_response),
                    wrap="pre",
                ),
            )

        with lcars.data_panel(
            "Ollama Activity",
            color="blue-bell",
            id="lora-test-log-panel",
            weight=12,
            aspect="wide",
            group="lora-compare",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            lcars.log(
                LOG_OLLAMA,
                max_lines=500,
                title="Ollama Build And Test Output",
                id="lora-test-log",
                options=LOG_VIEW_OPTIONS,
            )


def _command_page() -> None:
    with lcars.page("Command", id="command", layout="grid", fillers=False):
        with lcars.data_panel(
            "Launch Readiness",
            color="tanoi",
            id="command-readiness-panel",
            weight=12,
            aspect="wide",
            group="command-overview",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            issues = STATE.preflight_issues
            errors = sum(1 for issue in issues if issue.severity == "error")
            warnings = sum(1 for issue in issues if issue.severity == "warn")
            lcars.metric(
                "Run Gate",
                "BLOCKED" if errors else "READY",
                status="crit" if errors else "ok",
                color="red" if errors else "tanoi",
                id="run-gate-metric",
                options=lcars.MetricOptions(
                    secondary_value=f"{errors} blocking issue(s)",
                    trend="down" if errors else "flat",
                ),
            )
            lcars.metric(
                "Warnings",
                str(warnings),
                status="warn" if warnings else "ok",
                color="golden-tanoi",
                id="warning-count-metric",
                options=lcars.MetricOptions(secondary_value="preflight advisories"),
            )
            lcars.metric(
                "Axolotl CLI",
                "FOUND" if STATE.runner.axolotl_path else "MISSING",
                status="ok" if STATE.runner.axolotl_path else "crit",
                color="anakiwa",
                id="axolotl-cli-metric",
                options=lcars.MetricOptions(
                    secondary_value=STATE.runner.axolotl_path or "not on PATH",
                ),
            )
            _enhanced_table(
                issue_rows(issues),
                title="Preflight Matrix",
                id="preflight-table",
                filter_columns={"Level", "Check", "Detail"},
            )

        with lcars.control_panel(
            "Primary Actions",
            color="golden-tanoi",
            id="command-actions-panel",
            weight=6,
            aspect="tall",
            group="command-overview",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            if lcars.button("Run Preflight", color="anakiwa", id="run-preflight"):
                _run_preflight_action()
            if lcars.button("Save Structured Config", color="tanoi", id="save-structured-config"):
                _save_config_action()
            if lcars.button(
                "Start Single Training",
                color="red",
                id="quick-start-training",
                options=lcars.ButtonOptions(
                    confirm="Start one Axolotl training action with the active config?",
                    debounce_ms=750,
                    busy_label="Starting",
                ),
            ):
                _start_axolotl_action("train")
            if lcars.button(
                "Stop Axolotl",
                color="red",
                id="quick-stop-axolotl",
                options=lcars.ButtonOptions(
                    confirm="Stop the active Axolotl process?",
                    debounce_ms=750,
                    busy_label="Stopping",
                ),
            ):
                _stop_axolotl_action()
            lcars.markdown(
                f"Active config: `{STATE.config_store.active_path}`\n\n"
                "[Open raw YAML editor](/raw)",
                id="command-raw-link",
                options=lcars.MarkdownOptions(copy_code=True),
            )

        with lcars.data_panel(
            "Current Config Summary",
            color="blue-bell",
            id="command-config-panel",
            weight=9,
            aspect="wide",
            group="command-config",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                STATE.config_store.summary_rows(),
                title="Active YAML",
                id="config-summary-table",
                filter_columns={"Key", "Value"},
                copy_columns={"Key", "Value"},
            )


def _config_page() -> None:
    with lcars.page("Config", id="config", layout="grid", fillers=False):
        with lcars.control_panel(
            "Config Files",
            color="golden-tanoi",
            id="config-files-panel",
            weight=7,
            aspect="tall",
            group="config-manager",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            lcars.markdown(
                "Structured pages cover the high-impact Axolotl surface. The raw YAML editor remains "
                "the complete escape hatch for deeply nested or experimental options.",
                id="config-note",
            )
            configs = STATE.config_store.list_configs()
            with lcars.form(
                "Active Config Selection",
                action_id="config-switch",
                submit_label="Switch Config",
                id="config-switch-form",
                color="anakiwa",
            ):
                selected = lcars.select(
                    "Active Config",
                    configs,
                    value=STATE.config_store.active_name,
                    id="active-config-select",
                    settings=SEARCHABLE_CHOICES,
                )
            if _is_active_action("config-switch"):
                _switch_config_action(selected)
            with lcars.form(
                "New Starter Config",
                action_id="config-create",
                submit_label="Create Starter",
                id="config-create-form",
                color="tanoi",
            ):
                _seed_text("new-config-name", "experiment.yml")
                new_name = lcars.text_input(
                    "New Config Name",
                    value="experiment.yml",
                    placeholder="experiment.yml",
                    autocomplete=False,
                    id="new-config-name",
                    options=lcars.TextInputOptions(
                        commit="enter",
                        validation=lcars.ValidationOptions(
                            required=True,
                            pattern=r"^[^/\\]+\.ya?ml$",
                            message="Use a YAML filename without directories.",
                        ),
                    ),
                )
            if _is_active_action("config-create"):
                _create_config_action(new_name)
            if lcars.button("Duplicate Active", color="lilac", id="config-duplicate"):
                _duplicate_config_action()
            if lcars.button("Save All Structured", color="tanoi", id="config-save-all"):
                _save_config_action()
            if lcars.button("Validate Config", color="anakiwa", id="config-validate"):
                _run_preflight_action()

        with lcars.data_panel(
            "Coverage Map",
            color="blue-bell",
            id="config-coverage-panel",
            weight=12,
            aspect="wide",
            group="config-manager",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                _coverage_rows(),
                title="Structured Surface",
                id="config-coverage-table",
                filter_columns={"Page", "Group"},
                numeric_columns={"Fields"},
            )
            _enhanced_table(
                STATE.config_store.summary_rows(),
                title="Summary",
                id="config-page-summary-table",
                filter_columns={"Key", "Value"},
                copy_columns={"Key", "Value"},
            )


def _config_setup_page() -> None:
    with lcars.page("Setup", id="config-setup", layout="grid", fillers=False):
        _setup_smart_panel()
        with lcars.data_panel(
            "Defaults / Examples",
            color="blue-bell",
            id="setup-defaults-panel",
            weight=12,
            aspect="wide",
            group="setup-reference",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                _setup_default_rows(),
                title="Axolotl Defaults And Starters",
                id="setup-defaults-table",
                filter_columns={"Field", "Need", "Role"},
                copy_columns={"Field"},
                page_size=25,
            )
        for group, color in (
            ("Model", "pale-canary"),
            ("Dataset", "golden-tanoi"),
            ("Sequence / Packing", "anakiwa"),
            ("Run Safety", "lilac"),
        ):
            group_id = group.lower().replace(" ", "-").replace("/", "")
            with lcars.padd(
                f"{group} Essentials",
                color=color,
                id=f"setup-{group_id}-panel",
                weight=7,
                aspect="tall",
                group="setup-fields",
                options=COLLAPSIBLE_PANEL_OPTIONS,
            ):
                _render_config_fields({group}, keys=SETUP_FIELD_KEYS, include_headers=False)
        _config_page_actions("setup")


def _config_train_page() -> None:
    _config_group_page(
        "Train",
        "config-train",
        "tanoi",
        ("Training", "Adapter / PEFT", "Optimizer"),
        "train",
    )


def _config_hardware_page() -> None:
    _config_group_page(
        "Hardware",
        "config-hardware",
        "anakiwa",
        ("Precision / Memory", "Attention / Kernels", "Distributed"),
        "hardware",
    )


def _config_tracking_page() -> None:
    _config_group_page(
        "Tracking",
        "config-tracking",
        "lilac",
        ("Tracking", "Integrations", "RL / Evaluation"),
        "tracking",
    )


def _config_group_page(
    title: str,
    page_id: str,
    color: str,
    groups: tuple[str, ...],
    suffix: str,
) -> None:
    with lcars.page(title, id=page_id, layout="grid", fillers=False):
        for group_name in groups:
            group_id = group_name.lower().replace(" ", "-").replace("/", "")
            with lcars.padd(
                group_name,
                color=color,
                id=f"{suffix}-{group_id}-panel",
                weight=8,
                aspect="tall",
                group=f"{suffix}-fields",
                options=COLLAPSIBLE_PANEL_OPTIONS,
            ):
                _render_config_fields({group_name})
        _config_page_actions(suffix)


def _config_advanced_page() -> None:
    advanced_groups = ("Run Safety", "Model", "Dataset", "Sequence / Packing")
    with lcars.page("Advanced", id="config-advanced", layout="grid", fillers=False):
        for group_name in advanced_groups:
            group_keys = {
                spec.key
                for spec in FIELD_SPECS
                if spec.group == group_name and spec.key not in SETUP_FIELD_KEYS
            }
            if not group_keys:
                continue
            group_id = group_name.lower().replace(" ", "-").replace("/", "")
            with lcars.padd(
                f"{group_name} Advanced",
                color="blue-bell",
                id=f"advanced-{group_id}-panel",
                weight=8,
                aspect="tall",
                group="advanced-fields",
                options=COLLAPSIBLE_PANEL_OPTIONS,
            ):
                _render_config_fields(
                    {group_name},
                    keys=group_keys,
                    id_prefix="advanced",
                )
        _config_page_actions("advanced")


def _run_page() -> None:
    STATE.workflow.sync_active_config(STATE.config_store.active_name)
    with lcars.page("Workflow", id="run", layout="telemetry", fillers=False):
        with lcars.data_panel(
            "Axolotl Workflow",
            color="red",
            id="workflow-graph-panel",
            zone="primary",
            weight=12,
            aspect="wide",
            group="workflow-execution",
        ):
            workflow_state = lcars.node_canvas(
                STATE.workflow.document,
                title="Training Lifecycle",
                execution=STATE.workflow.execution_state(),
                color="red",
                id=WORKFLOW_CANVAS_ID,
                options=_workflow_canvas_options(),
            )
        if _is_active_action(WORKFLOW_CANVAS_ID) and workflow_state is not None:
            _workflow_graph_action(workflow_state)

        with lcars.control_panel(
            "Mission Control",
            color="golden-tanoi",
            id="workflow-control-panel",
            zone="side",
            weight=7,
            aspect="tall",
            group="workflow-execution",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            workflow_status = STATE.workflow.status.upper()
            lcars.metric(
                "Workflow",
                workflow_status,
                status=_workflow_status_severity(),
                color="red",
                id="workflow-status",
                options=lcars.MetricOptions(
                    secondary_value=STATE.config_store.active_name,
                ),
            )
            lcars.text(
                STATE.workflow.current_label,
                size="mono",
                color="anakiwa",
                id="workflow-current-stage",
                options=lcars.TextOptions(wrap="wrap"),
            )
            lcars.progress(
                "Workflow Completion",
                STATE.workflow.progress_percent,
                color="tanoi",
                id="workflow-progress",
                options=lcars.MeterOptions(
                    unit="%",
                    segments=12,
                    ticks=True,
                    warn_threshold=101,
                    crit_threshold=101,
                ),
            )
            lcars.text(
                STATE.workflow.message,
                size="mono",
                color="pale-canary",
                id="workflow-message",
                options=lcars.TextOptions(wrap="wrap"),
            )
            if lcars.button(
                "Start Workflow",
                color="red",
                id="workflow-start",
                disabled=STATE.workflow.is_active or STATE.runner.is_running(),
                options=lcars.ButtonOptions(
                    confirm=("Run every connected Axolotl stage in order with the active config?"),
                    debounce_ms=750,
                    busy_label="Launching",
                ),
            ):
                _start_workflow_action()
            if lcars.button(
                "Cancel Workflow",
                color="red",
                id="workflow-cancel",
                disabled=not STATE.workflow.is_active,
                options=lcars.ButtonOptions(
                    confirm="Stop the active process and cancel every queued workflow stage?",
                    debounce_ms=750,
                    busy_label="Cancelling",
                ),
            ):
                _cancel_workflow_action()
            if lcars.button(
                "Validate Workflow",
                color="anakiwa",
                id="workflow-validate",
                disabled=STATE.workflow.is_active,
            ):
                _validate_workflow_action()
            if lcars.button(
                "Reset Workflow",
                color="lilac",
                id="workflow-reset",
                disabled=STATE.workflow.is_active,
                options=lcars.ButtonOptions(
                    confirm="Replace this graph with the starter preprocess → train → evaluate plan?",
                ),
            ):
                _reset_workflow_action()


def _console_page() -> None:
    with lcars.page("Console", id="console", layout="console", fillers=False):
        with lcars.data_panel(
            "Process Output",
            color="red",
            id="run-process-panel",
            zone="primary",
            weight=12,
            aspect="wide",
            group="process-console",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            lcars.metric(
                "Process",
                STATE.runner.status_label(),
                status=STATE.runner.status_severity(),
                color="red",
                id="run-status",
                options=lcars.MetricOptions(
                    secondary_value=STATE.config_store.active_name,
                ),
            )
            command = " ".join(STATE.runner.state.command) if STATE.runner.state.command else "idle"
            lcars.text(
                command[:500],
                size="mono",
                id="run-command-text",
                options=lcars.TextOptions(copyable=True, wrap="pre"),
            )
            lcars.log(
                LOG_AXOLOTL,
                max_lines=1000,
                title="Axolotl Output",
                id="axolotl-output-log",
                options=LOG_VIEW_OPTIONS,
            )

        with lcars.control_panel(
            "Single Action Override",
            color="blue-bell",
            id="run-controls-panel",
            zone="dock",
            weight=7,
            aspect="wide",
            group="process-console",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            action = lcars.select(
                "Axolotl Action",
                list(AXOLOTL_ACTIONS),
                value="train",
                id="run-action",
                settings=SEARCHABLE_CHOICES,
            )
            launcher = lcars.select(
                "Launcher",
                [
                    lcars.SelectOption(label="Axolotl default", value=""),
                    lcars.SelectOption(label="Python", value="python"),
                    lcars.SelectOption(label="Accelerate", value="accelerate"),
                    lcars.SelectOption(label="Torchrun", value="torchrun"),
                ],
                value="",
                id="run-launcher",
            )
            _seed_text("run-cli-args", "")
            cli_args = lcars.text_input(
                "Axolotl Args",
                placeholder="Action flags or fetch target, shell-style",
                autocomplete=False,
                id="run-cli-args",
                options=COMMAND_INPUT_OPTIONS,
            )
            _seed_text("run-launcher-args", "")
            launcher_args = lcars.text_input(
                "Launcher Args",
                placeholder="Placed after -- for accelerate/torchrun",
                autocomplete=False,
                id="run-launcher-args",
                options=COMMAND_INPUT_OPTIONS,
            )
            if lcars.button(
                "Start Single Action",
                color="red",
                id="run-start",
                disabled=STATE.workflow.is_active or STATE.runner.is_running(),
                options=lcars.ButtonOptions(
                    confirm="Launch only this Axolotl action outside the workflow?",
                    debounce_ms=750,
                    busy_label="Launching",
                ),
            ):
                _start_axolotl_action(
                    action,
                    launcher=launcher,
                    cli_args=cli_args,
                    launcher_args=launcher_args,
                )
            if lcars.button(
                "Stop Process",
                color="red",
                id="run-stop",
                disabled=not STATE.runner.is_running(),
                options=lcars.ButtonOptions(
                    confirm="Stop the active Axolotl process?",
                    debounce_ms=750,
                    busy_label="Stopping",
                ),
            ):
                _stop_axolotl_action()
            if lcars.button("Preflight", color="anakiwa", id="run-preflight-local"):
                _run_preflight_action()


def _resources_page() -> None:
    with lcars.page("Resources", id="resources", layout="console", fillers=False):
        snapshot = STATE.telemetry.latest or STATE.telemetry.sample()
        cfg = _load_config_or_empty()
        primary_disk = _primary_disk(snapshot.disks)
        with lcars.data_panel(
            "System Load",
            color="anakiwa",
            id="resource-load-panel",
            zone="side",
            weight=8,
            aspect="tall",
            span=(2, 2),
            group="resource-overview",
            options=DENSE_PANEL_OPTIONS,
        ):
            meter_options = lcars.MeterOptions(
                unit="%",
                segments=24,
                ticks=True,
                warn_threshold=80,
                crit_threshold=92,
            )
            lcars.gauge(
                "CPU Load",
                snapshot.cpu_percent,
                unit="%",
                warn_threshold=80,
                crit_threshold=92,
                id="cpu-gauge",
                options=meter_options,
            )
            lcars.gauge(
                "Memory Load",
                snapshot.ram_percent,
                unit="%",
                warn_threshold=80,
                crit_threshold=92,
                id="ram-gauge",
                options=meter_options,
            )
        with lcars.data_panel(
            "Resource Trend",
            color="anakiwa",
            id="resource-trend-panel",
            zone="primary",
            weight=12,
            aspect="wide",
            group="resource-overview",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.chart(
                STATE.telemetry.chart_payload(),
                color="anakiwa",
                id="resource-chart",
                options=lcars.ChartOptions(
                    x_axis=lcars.AxisOptions(label="Sample"),
                    y_axis=lcars.AxisOptions(label="Utilization %", min=0, max=100),
                    tooltip=True,
                    zoom=True,
                    reference_lines=[
                        lcars.ReferenceLine(value=80, label="Warning", color="golden-tanoi"),
                        lcars.ReferenceLine(value=92, label="Critical", color="red"),
                    ],
                ),
            )

        with lcars.data_panel(
            "GPU Telemetry",
            color="blue-bell",
            id="resource-gpu-panel",
            zone="side",
            weight=9,
            aspect="wide",
            span=(2, 3),
            group="resource-compute",
            options=DENSE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                gpu_rows(snapshot.gpus),
                title="GPU Telemetry",
                id="gpu-table",
            )

        with lcars.data_panel(
            "CPU / RAM Processes",
            color="lilac",
            id="resource-process-panel",
            zone="primary",
            weight=10,
            aspect="wide",
            group="resource-compute",
            options=DENSE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                process_rows(),
                title="Top RAM / CPU Processes",
                id="process-table",
                filter_columns={"PID", "Process", "State"},
                page_size=5,
            )

        with lcars.data_panel(
            "Active GPU Processes",
            color="blue-bell",
            id="resource-gpu-process-panel",
            zone="side",
            weight=8,
            aspect="wide",
            span=(2, 3),
            group="resource-compute",
            options=DENSE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                gpu_process_rows(),
                title="GPU Processes",
                id="gpu-process-table",
            )

        with lcars.data_panel(
            "Mounted Storage",
            color="golden-tanoi",
            id="resource-storage-panel",
            zone="primary",
            weight=9,
            aspect="wide",
            group="resource-storage",
            options=DENSE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                disk_rows(snapshot.disks),
                title="Mounted Filesystems",
                id="disk-table",
                filter_columns={"Device", "Mount"},
                copy_columns={"Mount"},
            )

        with lcars.data_panel(
            "Training Storage Hotspots",
            color="golden-tanoi",
            id="resource-storage-hotspots-panel",
            zone="primary",
            weight=8,
            aspect="wide",
            group="resource-storage",
            options=DENSE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                _storage_rows(cfg),
                title="Storage Hotspots",
                id="storage-hotspot-table",
                filter_columns={"Location", "Path"},
                copy_columns={"Path"},
            )

        with lcars.data_panel(
            "Storage Pressure",
            color="golden-tanoi",
            id="resource-storage-pressure-panel",
            zone="side",
            weight=7,
            aspect="tall",
            span=(2, 3),
            group="resource-storage",
            options=DENSE_PANEL_OPTIONS,
        ):
            disk_percent = primary_disk.percent if primary_disk is not None else 0.0
            lcars.gauge(
                f"Volume Load · {primary_disk.mountpoint if primary_disk is not None else 'none'}",
                disk_percent,
                unit="%",
                warn_threshold=85,
                crit_threshold=95,
                id="disk-usage-gauge",
                options=lcars.MeterOptions(
                    unit="%",
                    segments=20,
                    ticks=True,
                    warn_threshold=85,
                    crit_threshold=95,
                ),
            )
            lcars.metric(
                "Disk Free",
                format_bytes(primary_disk.free) if primary_disk is not None else "unavailable",
                status=_percent_status(disk_percent, warn=85, crit=95),
                color="golden-tanoi",
                id="disk-free-metric",
                options=lcars.MetricOptions(
                    secondary_value=(
                        format_bytes(primary_disk.total)
                        if primary_disk is not None
                        else "no mounted volume"
                    ),
                ),
            )
            lcars.metric(
                "RAM Used",
                f"{format_bytes(snapshot.ram_used)} / {format_bytes(snapshot.ram_total)}",
                status=_percent_status(snapshot.ram_percent),
                color="blue-bell",
                id="ram-used-metric",
            )


def _hub_page() -> None:
    _handle_hf_table_action()
    with lcars.page("HF Hub", id="hub", layout="telemetry", fillers=False):
        repo_id = _widget_value("hf-repo-id", STATE.hf.last_repo_id).strip()
        target_repo_type = _widget_value(
            "hf-repo-type",
            STATE.hf.last_repo_type,
        )
        selected_result = _hf_result_for(repo_id)
        selected_blocked = bool(selected_result is not None and selected_result.blocked)

        with lcars.data_panel(
            "Repository Browser",
            color="lilac",
            id="hf-results-panel",
            zone="primary",
            weight=12,
            aspect="wide",
            group="hf-browser",
            options=DENSE_PANEL_OPTIONS,
        ):
            lcars.table(
                _hf_result_rows(),
                title="Hugging Face Results",
                color="lilac",
                id=HF_RESULTS_TABLE_ID,
                options=_hf_result_table_options(),
            )

        with lcars.control_panel(
            "Hub Operations",
            color="lilac",
            id="hf-operations-panel",
            zone="primary",
            weight=10,
            aspect="wide",
            group="hf-browser",
            options=DENSE_PANEL_OPTIONS,
        ):
            with lcars.form(
                "Repository Query",
                action_id="hf-search",
                submit_label="Run Query",
                id="hf-search-form",
                color="anakiwa",
                options=lcars.FormOptions(layout="row"),
            ):
                _seed_text("hf-query", "llama instruct")
                query = lcars.text_input(
                    "Query / Repository ID",
                    value="llama instruct",
                    placeholder="keywords or owner/repository",
                    autocomplete=False,
                    id="hf-query",
                    options=SEARCH_INPUT_OPTIONS,
                    hint=(
                        "Search Hub accepts keywords such as “qwen instruct”. Exact Repository "
                        "expects an owner/name id such as Qwen/Qwen3.5-4B."
                    ),
                )
                query_mode = lcars.select(
                    "Mode",
                    HF_QUERY_MODE_OPTIONS,
                    value=HF_QUERY_MODE_OPTIONS[0],
                    id="hf-query-mode",
                    hint=(
                        "Search Hub returns a ranked result set. Inspect Exact Repository loads one "
                        "known owner/name id and its file manifest directly."
                    ),
                )
                search_repo_type = lcars.select(
                    "Repo Type",
                    ["model", "dataset"],
                    value=STATE.hf.last_repo_type,
                    id="hf-search-repo-type",
                    hint=(
                        "Choose model for base checkpoints and adapters, or dataset for training "
                        "corpora. This changes the search without silently retargeting a prior "
                        "selection."
                    ),
                )
                _seed_text("hf-revision", "")
                revision = lcars.text_input(
                    "Revision [optional]",
                    placeholder="branch/tag/commit",
                    autocomplete=False,
                    id="hf-revision",
                    options=COMMAND_INPUT_OPTIONS,
                    hint=(
                        "Leave blank for the repository default. For exact, reproducible work use "
                        "a branch, tag, commit SHA, or refs/pr/... revision."
                    ),
                )
            if _is_active_action("hf-search") or _is_active_action("hf-query"):
                if query_mode == HF_QUERY_MODE_OPTIONS[1]:
                    _hf_inspect_action(query, search_repo_type, revision)
                else:
                    _hf_search_action(
                        query,
                        search_repo_type,
                        sort=_widget_value("hf-sort", HF_SORT_OPTIONS[0]),
                        compatibility=_widget_value(
                            "hf-compatibility",
                            HF_COMPATIBILITY_OPTIONS[0],
                        ),
                        limit=_widget_value("hf-limit", HF_LIMIT_OPTIONS[0]),
                        sift=_widget_value("hf-sift", ""),
                        local_sort=STATE.hf.local_sort,
                        artifact_filter=_widget_value(
                            "hf-artifact-filter",
                            HF_ARTIFACT_FILTER_OPTIONS[0],
                        ),
                        quant_filter=_widget_value(
                            "hf-quant-filter",
                            HF_QUANT_FILTER_OPTIONS[0],
                        ),
                        fit_filter=_widget_value(
                            "hf-fit-filter",
                            HF_FIT_FILTER_OPTIONS[0],
                        ),
                        vram_limit=_widget_value(
                            "hf-vram-limit",
                            str(STATE.hf.vram_limit_gb or 24),
                        ),
                    )

            lcars.header(
                "Selected Repository",
                size="h4",
                color="tanoi",
                id="hf-selected-heading",
                hint=(
                    "Selection is independent from the current search type and filters, so a saved "
                    "target remains available while you explore another result set."
                ),
            )
            selection_value, selection_detail, selection_status = _hf_selection_metric(
                repo_id,
                target_repo_type,
                selected_result,
            )
            lcars.metric(
                "Selection State",
                selection_value,
                status=selection_status,
                color="tanoi",
                id="hf-selection-status",
                options=lcars.MetricOptions(secondary_value=selection_detail),
            )
            with lcars.hint(
                "hf-selection-status",
                title="Selection Workflow",
                trigger=["hover", "press"],
                placement="left",
                max_width=460,
            ):
                lcars.markdown(
                    "1. **Select** a row to make it the command target.\n"
                    "2. **Expand** it to inspect compatibility, lineage, and exact files.\n"
                    "3. **Use** it in the active Axolotl config or **download** only compatible "
                    "files.\n\n"
                    "◆ marks a repository already used by the active config. ● marks a repository "
                    "whose exact manifest has been inspected.",
                    id="hf-selection-guide",
                )
            lcars.text(
                repo_id or "No repository selected.",
                size="mono",
                id="hf-selected-repo-copy",
                options=_hf_selected_text_options(
                    repo_id,
                    target_repo_type,
                ),
                hint=(
                    "This owner/name id is the target for the three actions below. Use its inline "
                    "copy control or open the linked Hugging Face page."
                ),
            )
            if lcars.button(
                "Find Fine-Tunes",
                color="lilac",
                id="hf-related",
                disabled=not bool(repo_id.strip()) or target_repo_type != "model",
                hint=(
                    "Search model lineage and naming metadata for compatible descendants of the "
                    "selected base model, then expose them inside the expanded result."
                ),
            ):
                _hf_related_action(repo_id)
            if lcars.button(
                "Download Compatible Files",
                color="golden-tanoi",
                id="hf-download",
                disabled=not bool(repo_id.strip()) or selected_blocked,
                options=lcars.ButtonOptions(
                    confirm="Queue the compatible files from this repository?",
                    debounce_ms=750,
                    busy_label="Queueing",
                ),
                hint=(
                    "Queue only Axolotl-relevant configs, tokenizer support, and trainable weights "
                    "or dataset files. Track progress and cache use on Content."
                ),
            ):
                _hf_download_action(repo_id, target_repo_type, revision)
            if lcars.button(
                "Use Repo In Config",
                color="tanoi",
                id="hf-use-repo",
                disabled=not bool(repo_id.strip()) or selected_blocked,
                hint=(
                    "Apply a base model, PEFT adapter, or dataset to the correct active YAML field, "
                    "then immediately rerun preflight."
                ),
            ):
                _hf_use_repo_action(repo_id, target_repo_type)

            lcars.header(
                "Result Refinement",
                size="h4",
                color="blue-bell",
                id="hf-filter-heading",
                hint=(
                    "Refinement reruns the saved query and applies Hub ordering, compatibility, "
                    "metadata, artifact, weight-format, and model-fit constraints atomically."
                ),
            )
            lcars.button(
                "Refine Results",
                color="blue-bell",
                id="hf-refine-results",
                hint=(
                    "Open the advanced filter workspace. Its controls live in this pinned popover "
                    "so the main operations rail stays focused on search and repository actions."
                ),
            )
            with lcars.hint(
                "hf-refine-results",
                title="Advanced Result Filters",
                trigger=["click", "press"],
                placement="left",
                max_width=620,
            ):
                with lcars.form(
                    "Refine Current Query",
                    action_id="hf-filter-results",
                    submit_label="Apply / Refresh Results",
                    id="hf-filter-form",
                    color="blue-bell",
                    options=lcars.FormOptions(
                        layout="grid",
                        columns=2,
                        description=(
                            "All values are submitted together, then the complete result snapshot "
                            "is hydrated with exact metadata before local filters are evaluated."
                        ),
                    ),
                ):
                    sort = lcars.select(
                        "Hub Sort",
                        HF_SORT_OPTIONS,
                        value=HF_SORT_OPTIONS[0],
                        id="hf-sort",
                        hint=(
                            "Ordering sent to Hugging Face before exact metadata is hydrated. "
                            "Clicking a visible table column then applies a local sort."
                        ),
                    )
                    compatibility = lcars.select(
                        "Compatibility",
                        HF_COMPATIBILITY_OPTIONS,
                        value=HF_COMPATIBILITY_OPTIONS[0],
                        id="hf-compatibility",
                        hint=(
                            "Compatible-only hides known runtime-only artifacts. Include warnings "
                            "when auditing GGUF, incomplete, or unusual repositories."
                        ),
                    )
                    limit = lcars.select(
                        "Result Limit",
                        HF_LIMIT_OPTIONS,
                        value=HF_LIMIT_OPTIONS[0],
                        id="hf-limit",
                        hint=(
                            "Maximum results requested from the Hub. Exact metadata is loaded for "
                            "the complete result snapshot before it is filtered, sorted, or shown."
                        ),
                    )
                    _seed_text("hf-sift", "")
                    sift = lcars.text_input(
                        "Metadata Contains",
                        placeholder="repo, tag, quant, family",
                        autocomplete=False,
                        id="hf-sift",
                        options=SEARCH_INPUT_OPTIONS,
                        hint=(
                            "Case-insensitive local match across repository id, tags, pipeline, "
                            "library, weights, quantization, and compatibility."
                        ),
                    )
                    artifact_filter = lcars.select(
                        "Artifact",
                        HF_ARTIFACT_FILTER_OPTIONS,
                        value=HF_ARTIFACT_FILTER_OPTIONS[0],
                        id="hf-artifact-filter",
                        hint=(
                            "Narrow models, PEFT adapters, datasets, or runtime-only artifacts "
                            "after repository metadata has been classified."
                        ),
                    )
                    quant_filter = lcars.select(
                        "Weight Format",
                        HF_QUANT_FILTER_OPTIONS,
                        value=HF_QUANT_FILTER_OPTIONS[0],
                        id="hf-quant-filter",
                        hint=(
                            "Filter model weight formats. Dataset searches ignore this model-only "
                            "constraint rather than disappearing unexpectedly."
                        ),
                    )
                    vram_limit = lcars.number_input(
                        "Model VRAM Budget",
                        value=float(STATE.hf.vram_limit_gb or 24),
                        min=1,
                        max=256,
                        step=1,
                        id="hf-vram-limit",
                        options=lcars.NumberInputOptions(
                            precision=0,
                            suffix=" GB",
                            required=True,
                        ),
                        hint=(
                            "A first-pass fit comparison against known model weight bytes. It is "
                            "guidance, not a guarantee of training memory use."
                        ),
                    )
                    fit_filter = lcars.select(
                        "Model VRAM Fit",
                        HF_FIT_FILTER_OPTIONS,
                        value="any",
                        id="hf-fit-filter",
                        hint=(
                            "Keep every model, only models with known size, or only weights within "
                            "the stated VRAM budget. Dataset rows report data size instead."
                        ),
                    )
            lcars.text(
                _current_hf_filter_summary(),
                size="mono",
                color="blue-bell",
                id="hf-filter-summary",
                options=lcars.TextOptions(
                    wrap="wrap",
                    max_lines=4,
                    selectable=False,
                ),
                hint=(
                    "This is the saved atomic refinement state. Open Refine Results to change it; "
                    "table-header sorting then reorders the complete enriched result set locally."
                ),
            )
            if _is_active_action("hf-filter-results"):
                _set_widget_value("hf-query-mode", HF_QUERY_MODE_OPTIONS[0])
                _hf_search_action(
                    _widget_value("hf-query", "llama instruct"),
                    _widget_value("hf-search-repo-type", STATE.hf.last_repo_type),
                    sort=sort,
                    compatibility=compatibility,
                    limit=limit,
                    sift=sift,
                    local_sort=STATE.hf.local_sort,
                    artifact_filter=artifact_filter,
                    quant_filter=quant_filter,
                    fit_filter=fit_filter,
                    vram_limit=vram_limit,
                )


def _content_page() -> None:
    rows, total_text, total_bytes = STATE.hf.cache_rows()
    with lcars.page("Content", id="content", layout="grid", fillers=False):
        with lcars.data_panel(
            "Transfer Queue",
            color="golden-tanoi",
            id="hf-transfers-panel",
            weight=10,
            aspect="wide",
            group="content-transfers",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                _hf_job_rows(),
                title="Download Jobs",
                id="hf-jobs-table",
                filter_columns={"Repo", "Status", "Type"},
                copy_columns={"Repo", "Revision", "Local Path"},
                page_size=25,
            )

        with lcars.data_panel(
            "Hub Activity",
            color="red",
            id="hf-activity-panel",
            weight=7,
            aspect="tall",
            group="content-transfers",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            lcars.log(
                LOG_HF,
                max_lines=300,
                title="HF Activity",
                id="hf-activity-log",
                options=LOG_VIEW_OPTIONS,
            )

        with lcars.data_panel(
            "Downloaded Content",
            color="blue-bell",
            id="content-cache-panel",
            weight=12,
            aspect="wide",
            group="content-cache",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            lcars.metric(
                "HF Cache",
                cache_summary_text(total_bytes, total_text),
                status="ok",
                color="blue-bell",
                id="hf-cache-total",
                options=lcars.MetricOptions(
                    secondary_value=f"{len(rows)} cached repo(s)",
                ),
            )
            _enhanced_table(
                rows
                or [
                    {
                        "Type": "",
                        "Repo": "No cached Hugging Face repos",
                        "Size": "",
                        "Files": "",
                        "Revision": "",
                        "Path": "",
                    }
                ],
                title="HF Cache",
                id="hf-cache-table",
                filter_columns={"Type", "Repo", "Revision", "Path"},
                copy_columns={"Repo", "Revision", "Path"},
                page_size=25,
            )

        with lcars.control_panel(
            "Cache Disposal",
            color="golden-tanoi",
            id="content-disposal-panel",
            weight=6,
            aspect="tall",
            group="content-cache",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _seed_text("delete-repo-id", STATE.hf.last_repo_id)
            repo_id = lcars.text_input(
                "Delete Repo ID",
                placeholder="owner/name",
                autocomplete=False,
                id="delete-repo-id",
                options=SEARCH_INPUT_OPTIONS,
            )
            repo_type = lcars.select(
                "Delete Repo Type",
                ["model", "dataset"],
                value=STATE.hf.last_repo_type,
                id="delete-repo-type",
            )
            if lcars.button("Refresh Cache", color="anakiwa", id="cache-refresh"):
                _update_cache_widgets()
                lcars.notify("HF cache refreshed.")
            if lcars.button(
                "Use Last Snapshot In Config",
                color="blue-bell",
                id="hf-use-local",
                disabled=not bool(STATE.hf.last_local_path),
            ):
                _hf_use_last_local_action(STATE.hf.last_repo_type)
            if lcars.button(
                "Delete Cached Repo",
                color="red",
                id="cache-delete",
                options=lcars.ButtonOptions(
                    confirm="Permanently remove this repository from the local HF cache?",
                    debounce_ms=750,
                    busy_label="Deleting",
                ),
            ):
                _delete_cache_action(repo_id, repo_type)


def _ollama_page() -> None:
    with lcars.page("Ollama", id="ollama", layout="grid", fillers=False):
        with lcars.data_panel(
            "Axolotl Source Gate",
            color="pale-canary",
            id="ollama-rules-panel",
            weight=6,
            aspect="wide",
            group="ollama-overview",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                _ollama_rule_rows(),
                title="Axolotl Source Gate",
                id="ollama-rule-table",
                filter_columns={"Source", "Action"},
            )

        with lcars.data_panel(
            "Local Ollama Models",
            color="pale-canary",
            id="ollama-models-panel",
            weight=12,
            aspect="wide",
            group="ollama-overview",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _enhanced_table(
                STATE.ollama.rows(),
                title="Local Ollama Models",
                id="ollama-table",
                filter_columns={"Model", "Params", "Quant", "Source", "Axolotl"},
                copy_columns={"Model", "Source"},
                page_size=25,
            )

        with lcars.control_panel(
            "Ollama Source Workflow",
            color="tanoi",
            id="ollama-workflow-panel",
            weight=6,
            aspect="wide",
            group="ollama-workflow",
            options=COLLAPSIBLE_PANEL_OPTIONS,
        ):
            _seed_text(
                "ollama-model-name", STATE.ollama.models[0].name if STATE.ollama.models else ""
            )
            model_name = lcars.text_input(
                "Ollama Model Name",
                placeholder="name:tag",
                autocomplete=False,
                id="ollama-model-name",
                options=SEARCH_INPUT_OPTIONS,
            )
            if lcars.button("Refresh Ollama", color="anakiwa", id="ollama-refresh"):
                _ollama_refresh_action()
            if lcars.button("Search HF Source", color="lilac", id="ollama-search-hf"):
                _ollama_search_hf_action(model_name)
            if lcars.button("Use Compatible Source", color="tanoi", id="ollama-use-source"):
                _ollama_use_source_action(model_name)


def _setup_smart_panel() -> None:
    with lcars.control_panel(
        "Smart Setup",
        color="pale-canary",
        id="setup-smart-panel",
        weight=8,
        aspect="tall",
        group="setup-reference",
        options=COLLAPSIBLE_PANEL_OPTIONS,
    ):
        recipe = lcars.select(
            "Recipe",
            list(SETUP_RECIPES),
            value="LoRA SFT starter",
            id="setup-recipe",
            settings=SEARCHABLE_CHOICES,
        )
        model = lcars.select(
            "Model Preset",
            MODEL_PRESETS,
            value=MODEL_PRESETS[0],
            id="setup-model-preset",
            settings=SEARCHABLE_CHOICES,
        )
        dataset = lcars.select(
            "Dataset Preset",
            list(DATASET_PRESETS),
            value=next(iter(DATASET_PRESETS)),
            id="setup-dataset-preset",
            settings=SEARCHABLE_CHOICES,
        )
        if lcars.button("Apply Recipe", color="tanoi", id="setup-apply-recipe"):
            _setup_apply_recipe_action(recipe)
        if lcars.button("Apply Model", color="anakiwa", id="setup-apply-model"):
            _setup_apply_model_action(model)
        if lcars.button("Apply Dataset", color="golden-tanoi", id="setup-apply-dataset"):
            _setup_apply_dataset_action(dataset)
        if lcars.button("Use HF Selection", color="lilac", id="setup-use-hf"):
            _hf_use_repo_action(STATE.hf.last_repo_id, STATE.hf.last_repo_type)
        if lcars.button("Search Model Preset", color="blue-bell", id="setup-search-model"):
            _hf_search_action(
                model,
                "model",
                sort="downloads",
                compatibility=HF_COMPATIBILITY_OPTIONS[0],
                limit="12",
            )


def _setup_default_rows() -> list[dict[str, str]]:
    cfg = _load_config_or_empty()
    specs = [
        (
            "base_model",
            "Required",
            "None",
            "unsloth/Llama-3.2-1B-Instruct",
            "HF model id or local Transformers directory",
        ),
        (
            "datasets.0.path",
            "Required",
            "None",
            "teknium/GPT4-LLM-Cleaned",
            "HF dataset id, local file, or local directory",
        ),
        ("datasets.0.type", "Recommended", "None", "alpaca", "Axolotl formatter strategy"),
        (
            "datasets.0.ds_type",
            "Optional",
            "Infer local file extension",
            "json",
            "Only needed for local files/directories",
        ),
        (
            "sequence_len",
            "Required for most runs",
            "None",
            "2048",
            "Context length used for tokenization/training",
        ),
        (
            "sample_packing",
            "Optional",
            "Unset unless configured",
            "true",
            "Packs multiple samples into one sequence",
        ),
        ("val_set_size", "Optional", "Unset", "0.1", "Validation split fraction or count"),
        ("load_in_8bit", "Optional", "false", "true", "Lower VRAM LoRA starter mode"),
        ("load_in_4bit", "Optional", "false", "false", "QLoRA starter switches this on"),
        ("output_dir", "Optional", "./model-out", "./outputs/lora-out", "Training output path"),
        ("strict", "Optional", "false", "false", "CLI override safety behavior"),
    ]
    return [
        {
            "Field": key,
            "Current": str(_config_path_value(cfg, key) or ""),
            "Need": need,
            "Axolotl Default": axolotl_default,
            "UI Starter": starter,
            "Role": role,
        }
        for key, need, axolotl_default, starter, role in specs
    ]


def _storage_rows(cfg: dict[str, Any]) -> list[dict[str, str]]:
    output_dir = str(cfg.get("output_dir") or "")
    prepared_path = str(cfg.get("dataset_prepared_path") or "")
    return storage_hotspot_rows(PROJECT_ROOT, output_dir=output_dir, prepared_path=prepared_path)


def _primary_disk(disks: list[DiskInfo]) -> DiskInfo | None:
    if not disks:
        return None
    return next((disk for disk in disks if disk.mountpoint == "/"), disks[0])


def _hf_job_rows() -> list[dict[str, str]]:
    rows = STATE.hf.job_rows()
    return rows or [
        {
            "Repo": "",
            "Type": "",
            "Status": "No downloads queued",
            "Revision": "",
            "Estimate": "",
            "Local Path": "",
        }
    ]


def _hf_results_pagination(total_rows: int | None = None) -> tuple[int, int]:
    """Return the current session's validated Hub table page and page size."""

    page_size = _bounded_int(
        _widget_value(HF_RESULTS_PAGE_SIZE_KEY, str(HF_RESULTS_PAGE_SIZE)),
        default=HF_RESULTS_PAGE_SIZE,
        minimum=min(HF_RESULTS_PAGE_SIZES),
        maximum=max(HF_RESULTS_PAGE_SIZES),
    )
    if page_size not in HF_RESULTS_PAGE_SIZES:
        page_size = HF_RESULTS_PAGE_SIZE
    page = _bounded_int(
        _widget_value(HF_RESULTS_PAGE_KEY, "1"),
        default=1,
        minimum=1,
        maximum=1000,
    )
    if total_rows is not None:
        page = min(page, max(1, math.ceil(max(0, total_rows) / page_size)))
    return page, page_size


def _remember_hf_results_pagination(table_state: dict[str, Any]) -> tuple[int, int]:
    """Capture EnhancedTable pagination before a server refresh streams options."""

    current_page, current_page_size = _hf_results_pagination()
    page_size = _bounded_int(
        str(table_state.get("page_size") or current_page_size),
        default=current_page_size,
        minimum=min(HF_RESULTS_PAGE_SIZES),
        maximum=max(HF_RESULTS_PAGE_SIZES),
    )
    if page_size not in HF_RESULTS_PAGE_SIZES:
        page_size = current_page_size
    page = _bounded_int(
        str(table_state.get("page") or current_page),
        default=current_page,
        minimum=1,
        maximum=1000,
    )
    page = min(
        page,
        max(1, math.ceil(len(_hf_visible_results()) / page_size)),
    )
    _set_session_value(HF_RESULTS_PAGE_KEY, str(page))
    _set_session_value(HF_RESULTS_PAGE_SIZE_KEY, str(page_size))
    return page, page_size


def _reset_hf_results_page() -> None:
    """Start a new result set on page one without discarding its page size."""

    _set_session_value(HF_RESULTS_PAGE_KEY, "1")


def _hf_visible_page_results() -> list[Any]:
    """Return the server-owned result slice for the current Hub table page."""

    visible_results = _hf_visible_results()
    page, page_size = _hf_results_pagination(len(visible_results))
    start = (page - 1) * page_size
    return visible_results[start : start + page_size]


def _hf_result_table_options() -> lcars.TableOptions:
    visible_results = _hf_visible_results()
    page, page_size = _hf_results_pagination(len(visible_results))
    exact_count = sum(
        STATE.hf.details_for(result.repo_id, result.repo_type) is not None
        for result in visible_results
    )
    error_count = sum(
        bool(STATE.hf.inspection_error_for(result.repo_id, result.repo_type))
        for result in visible_results
    )
    pending_count = max(0, len(visible_results) - exact_count - error_count)
    if visible_results and pending_count:
        metadata_message = (
            f"Loading exact metadata for {pending_count}/{len(visible_results)} results."
        )
    elif visible_results and error_count:
        metadata_message = (
            f"Exact metadata ready for {exact_count}/{len(visible_results)} results; "
            f"{error_count} unavailable after retry."
        )
    elif visible_results:
        metadata_message = (
            f"Exact metadata ready for all {len(visible_results)} results."
        )
    else:
        metadata_message = "Run a Hub search or inspect an owner/repository id to begin."
    sort_key = (
        STATE.hf.local_sort
        if STATE.hf.local_sort
        in {
            "repo",
            "fit",
            "size",
            "files",
            "downloads",
            "likes",
            "updated",
        }
        else "downloads"
    )
    direction = "desc" if STATE.hf.local_sort_desc else "asc"
    visible_sort_keys = {"repo", "fit", "size", "files", "downloads"}
    visible_ids = {_hf_result_row_id(result) for result in visible_results}
    selected_ids = [
        _hf_result_row_id(result) for result in visible_results if _hf_result_is_current(result)
    ]
    expanded_ids = [row_id for row_id in STATE.hf.expanded_result_ids if row_id in visible_ids]
    return lcars.TableOptions(
        description=(
            "Select a row to target repository commands. Expand it to review compatibility, "
            "lineage, exact files, related models, and inline actions. Repository ids link to "
            "Hugging Face and have dedicated copy controls. ◆ marks the active config and ● "
            "confirms an exact manifest."
        ),
        feedback=lcars.WidgetFeedback(
            state="ready" if visible_results else "empty",
            message=metadata_message,
        ),
        columns=[
            lcars.TableColumn(
                key="repo",
                label="Repository",
                sortable=True,
                first_sort_direction="asc",
            ),
            lcars.TableColumn(
                key="fit",
                label="Fit",
                sortable=True,
                first_sort_direction="asc",
            ),
            lcars.TableColumn(
                key="size",
                label="Artifact",
                sortable=True,
                first_sort_direction="desc",
            ),
            lcars.TableColumn(
                key="files",
                label="Files",
                value_type="number",
                sortable=True,
                first_sort_direction="desc",
                align="end",
            ),
            lcars.TableColumn(
                key="downloads",
                label="Downloads",
                value_type="number",
                sortable=True,
                first_sort_direction="desc",
                align="end",
                value_format=lcars.ValueFormat(compact=True),
            ),
        ],
        sort=(
            [lcars.TableSort(key=sort_key, direction=direction)]
            if sort_key in visible_sort_keys
            else []
        ),
        pagination=lcars.TablePagination(
            page=page,
            page_size=page_size,
            total_rows=len(visible_results),
        ),
        selection=lcars.TableSelection(mode="single", selected_ids=selected_ids),
        expanded_ids=expanded_ids,
        expandable=True,
        sticky_header=True,
        density="compact",
        data_mode="server",
        emit_state_changes=True,
        row_click_select=True,
        interaction=lcars.InteractionOptions(action_id=HF_RESULTS_TABLE_ID),
    )


def _hf_visible_results() -> list[Any]:
    """Search rows plus a directly inspected repository that is outside the search."""

    results = list(STATE.hf.search_results)
    details = STATE.hf.selected_details
    if details is not None and not any(
        result.repo_id == details.result.repo_id and result.repo_type == details.result.repo_type
        for result in results
    ):
        results.append(details.result)
    return sorted_search_results(
        results,
        STATE.hf.local_sort,
        descending=STATE.hf.local_sort_desc,
    )


def _hf_result_row_id(result: Any) -> str:
    return f"{result.repo_type}:{result.repo_id}"


def _hf_parse_result_row_id(row_id: str) -> tuple[str, str] | None:
    repo_type, separator, repo_id = row_id.partition(":")
    if not separator or repo_type not in {"model", "dataset"} or not repo_id:
        return None
    return repo_type, repo_id


def _hf_configured_repositories() -> set[tuple[str, str]]:
    cfg = _load_config_or_empty()
    configured: set[tuple[str, str]] = set()
    model = str(cfg.get("base_model") or "").strip()
    if model:
        configured.add(("model", model))
    datasets = cfg.get("datasets")
    if isinstance(datasets, list):
        for dataset in datasets:
            if isinstance(dataset, dict):
                path = str(dataset.get("path") or "").strip()
            elif isinstance(dataset, str):
                path = dataset.strip()
            else:
                path = ""
            if path:
                configured.add(("dataset", path))
    return configured


def _hf_result_is_current(result: Any) -> bool:
    return result.repo_id == STATE.hf.last_repo_id and result.repo_type == STATE.hf.last_repo_type


def _hf_result_is_inspected(result: Any) -> bool:
    return STATE.hf.details_for(result.repo_id, result.repo_type) is not None


def _hf_result_status(result: Any, *, configured: bool, current: bool) -> str | None:
    if result.blocked:
        return "crit"
    if configured:
        return "ok"
    if result.fit.startswith("fits"):
        return "ok"
    if current:
        return "warn"
    return None


def _hf_result_display(result: Any, *, configured: bool, inspected: bool) -> str:
    markers = f"{'◆' if configured else ''}{'●' if inspected else ''}"
    return f"{markers} {result.repo_id}".strip()


def _hf_result_metadata(result: Any) -> str:
    values = [
        result.role.replace("_", " ") if result.role else "",
        result.pipeline,
        result.library,
        result.params,
        f"{result.file_count:,} files" if result.file_count else "",
        f"{result.downloads:,} downloads" if result.downloads is not None else "",
        f"{result.likes:,} likes" if result.likes is not None else "",
        f"updated {result.updated}" if result.updated else "",
    ]
    return " · ".join(value for value in values if value) or "Inspect to classify this repository"


def _hf_result_lineage(result: Any) -> str:
    values = []
    if result.base_models:
        values.append(f"Base: {result.base_models}")
    if result.children:
        values.append(f"Children: {result.children}")
    if result.tags:
        values.append(f"Tags: {result.tags}")
    text = " · ".join(values) or "No lineage or tags reported"
    return text if len(text) <= 180 else f"{text[:177]}..."


def _hf_result_detail_content(
    result: Any,
    *,
    configured: bool,
    current: bool,
) -> list[Any]:
    status = _hf_result_status(result, configured=configured, current=current)
    payload = {
        "repo_id": result.repo_id,
        "repo_type": result.repo_type,
    }
    details = STATE.hf.details_for(result.repo_id, result.repo_type)
    content: list[Any] = [
        lcars.TableDetailStatus(
            status=(
                "crit"
                if result.blocked
                else ("ok" if result.compatibility.startswith("OK") else (status or "muted"))
            ),
            label=result.compatibility or "Compatibility pending inspection",
        )
    ]
    if result.fit:
        content.append(
            lcars.TableDetailStatus(
                status="ok" if result.fit.startswith("fits") else "muted",
                label=(
                    f"Data · {result.fit}"
                    if result.repo_type == "dataset"
                    else f"VRAM · {result.fit}"
                ),
            )
        )
    if configured:
        content.append(lcars.TableDetailStatus(status="ok", label="ACTIVE CONFIG"))
    content.extend(
        [
            lcars.TableDetailLink(
                href=_hf_repo_url(result.repo_id, result.repo_type),
                label="Open on Hugging Face",
                target="_blank",
                rel="noopener noreferrer",
            ),
            lcars.TableDetailAction(
                label="Inspect / refresh",
                action_id="hf-inspect-row",
                value=payload,
            ),
        ]
    )
    if not result.blocked:
        content.extend(
            [
                lcars.TableDetailAction(
                    label="Use in config",
                    action_id="hf-use-row",
                    value=payload,
                ),
            ]
        )
    if result.repo_type == "model":
        content.append(
            lcars.TableDetailAction(
                label="Find fine-tunes",
                action_id="hf-related-row",
                value=payload,
            )
        )
    content.append(lcars.TableDetailText(text=_hf_result_metadata(result)))
    if result.repo_type == "model":
        content.append(lcars.TableDetailText(text=_hf_result_lineage(result), tone="muted"))
    if details is None:
        content.append(
            lcars.TableDetailText(
                text="Loading the exact repository manifest and file sizes…",
                tone="muted",
            )
        )
    else:
        content.append(
            lcars.TableDetailStatus(
                status="ok" if details.files else "warn",
                label=(
                    f"MANIFEST · {len(details.files):,} FILES"
                    if details.files
                    else "MANIFEST · NO FILES REPORTED"
                ),
            )
        )
        content.extend(_hf_file_detail_content(details, payload))
    content.extend(_hf_related_detail_content(result))
    return content


def _hf_file_detail_content(details: Any, payload: dict[str, str]) -> list[Any]:
    files = sorted(
        details.files,
        key=lambda item: (item.axolotl == "skip", item.kind, item.path.lower()),
    )
    if not files:
        return [
            lcars.TableDetailText(
                text="The Hub metadata did not expose a file manifest.",
                tone="muted",
            )
        ]

    rows: list[lcars.TableRow] = []
    for index, item in enumerate(files[:40]):
        queue_cell: str | lcars.TableCell = "BLOCKED"
        if item.axolotl != "skip":
            queue_cell = lcars.TableCell(
                value="",
                display="",
                action=lcars.ActionSpec(
                    label="Queue",
                    action_id="hf-download-file",
                    value={**payload, "file": item.path},
                ),
                status="ok",
            )
        rows.append(
            lcars.TableRow(
                id=f"{payload['repo_type']}:{payload['repo_id']}:file:{index}",
                cells=[
                    lcars.TableCell(
                        value=item.path,
                        copyable=True,
                        copy_value=item.path,
                        status="muted" if item.axolotl == "skip" else "ok",
                    ),
                    lcars.TableCell(
                        value=item.size,
                        display=format_bytes(item.size) if item.size else "unknown",
                    ),
                    item.kind,
                    lcars.TableCell(
                        value=item.axolotl,
                        status="muted" if item.axolotl == "skip" else "ok",
                    ),
                    queue_cell,
                ],
            )
        )
    content: list[Any] = [
        lcars.TableDetailTable(
            headers=["File", "Size", "Kind", "Axolotl", "Action"],
            rows=rows,
        )
    ]
    if len(files) > 40:
        content.append(
            lcars.TableDetailText(
                text=f"{len(files) - 40:,} additional files are omitted from this expansion.",
                tone="muted",
            )
        )
    return content


def _hf_related_detail_content(result: Any) -> list[Any]:
    if result.repo_type != "model" or STATE.hf.related_repo_id != result.repo_id:
        return []
    if not STATE.hf.related_results:
        return [
            lcars.TableDetailText(
                text="No compatible related fine-tunes were found.",
                tone="muted",
            )
        ]
    rows = []
    for related in STATE.hf.related_results:
        payload = {"repo_id": related.repo_id, "repo_type": related.repo_type}
        rows.append(
            lcars.TableRow(
                id=f"related:{related.repo_id}",
                cells=[
                    lcars.TableCell(
                        value=related.repo_id,
                        link=lcars.LinkSpec(
                            href=_hf_repo_url(related.repo_id, related.repo_type),
                            target="_blank",
                            rel="noopener noreferrer",
                        ),
                        copyable=True,
                        copy_value=related.repo_id,
                    ),
                    related.fit or "unknown",
                    related.quants or related.weights or related.size or "inspect",
                    related.downloads,
                    lcars.TableCell(
                        value="",
                        display="",
                        action=lcars.ActionSpec(
                            label="Inspect",
                            action_id="hf-inspect-row",
                            value=payload,
                        ),
                    ),
                ],
            )
        )
    return [
        lcars.TableDetailText(text="Compatible fine-tunes", tone="muted"),
        lcars.TableDetailTable(
            headers=["Repository", "Fit", "Weights / Quants", "Downloads", "Action"],
            rows=rows,
        ),
    ]


def _hf_repo_url(repo_id: str, repo_type: str) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}"


def _hf_result_rows() -> list[lcars.TableRow]:
    visible_results = _hf_visible_page_results()
    if not visible_results:
        return []

    configured_repositories = _hf_configured_repositories()
    rows: list[lcars.TableRow] = []
    for result in visible_results:
        details = STATE.hf.details_for(result.repo_id, result.repo_type)
        if details is not None:
            result = details.result
        configured = (result.repo_type, result.repo_id) in configured_repositories
        current = _hf_result_is_current(result)
        inspected = _hf_result_is_inspected(result)
        status = _hf_result_status(result, configured=configured, current=current)
        error = STATE.hf.inspection_error_for(result.repo_id, result.repo_type)
        fit_display = result.fit or "unknown"
        artifact_display = result.quants or result.weights or result.size
        if error:
            if fit_display == "unknown":
                fit_display = "metadata unavailable"
            artifact_display = artifact_display or "metadata unavailable"
        elif details is not None:
            if fit_display == "unknown":
                fit_display = "size unavailable"
            artifact_display = artifact_display or "no sized artifact"
        else:
            artifact_display = artifact_display or "loading"
        rows.append(
            lcars.TableRow(
                id=_hf_result_row_id(result),
                cells=[
                    lcars.TableCell(
                        value=result.repo_id,
                        display=_hf_result_display(
                            result,
                            configured=configured,
                            inspected=inspected,
                        ),
                        link=lcars.LinkSpec(
                            href=_hf_repo_url(result.repo_id, result.repo_type),
                            target="_blank",
                            rel="noopener noreferrer",
                        ),
                        copyable=True,
                        copy_value=result.repo_id,
                        status=status,
                    ),
                    fit_display,
                    lcars.TableCell(
                        value=result.weight_bytes or result.size_bytes or 0,
                        display=artifact_display,
                    ),
                    result.file_count,
                    result.downloads,
                ],
                expanded_content=_hf_result_detail_content(
                    result,
                    configured=configured,
                    current=current,
                ),
                loading=details is None and not bool(error),
                error=error or None,
            )
        )
    return rows


def _handle_hf_table_action() -> None:
    if _is_active_action(HF_RESULTS_TABLE_ID):
        payload = _active_action_value()
        kind = str(payload.get("kind") or "")
        table_state = payload.get("state")
        if not isinstance(table_state, dict):
            return
        _remember_hf_results_pagination(table_state)
        expanded_ids = [
            str(row_id)
            for row_id in table_state.get("expanded_ids", [])
            if isinstance(row_id, str) and _hf_parse_result_row_id(row_id) is not None
        ]
        previous_expanded = set(STATE.hf.expanded_result_ids)
        STATE.hf.set_expanded_result_ids(expanded_ids)

        if kind == "selection":
            selected_ids = table_state.get("selected_ids")
            selected_id = (
                str(selected_ids[-1]) if isinstance(selected_ids, list) and selected_ids else ""
            )
            selected = _hf_parse_result_row_id(selected_id)
            if selected is not None:
                repo_type, repo_id = selected
                STATE.hf.select_repository(repo_id, repo_type)  # type: ignore[arg-type]
                _set_session_value("hf-repo-id", repo_id)
                _set_session_value("hf-repo-type", repo_type)
                _update_hf_widgets()
            return

        if kind == "expansion":
            candidates = []
            for row_id in expanded_ids:
                parsed = _hf_parse_result_row_id(row_id)
                if parsed is None:
                    continue
                repo_type, repo_id = parsed
                if STATE.hf.details_for(repo_id, repo_type) is None:  # type: ignore[arg-type]
                    candidates.append((row_id, repo_type, repo_id))
            new_candidates = [
                candidate for candidate in candidates if candidate[0] not in previous_expanded
            ]
            retry_candidates = candidates if set(expanded_ids) == previous_expanded else []
            inspect_candidates = new_candidates or retry_candidates
            if inspect_candidates:
                _, repo_type, repo_id = inspect_candidates[0]
                STATE.hf.select_repository(repo_id, repo_type)  # type: ignore[arg-type]
                _set_session_value("hf-repo-id", repo_id)
                _set_session_value("hf-repo-type", repo_type)
                _hf_inspect_action(repo_id, repo_type, "")
            else:
                _update_hf_widgets()
            return

        if kind == "sort":
            sort_items = table_state.get("sort")
            if isinstance(sort_items, list) and sort_items and isinstance(sort_items[0], dict):
                sort_key = str(sort_items[0].get("key") or "")
                direction = str(sort_items[0].get("direction") or "")
                if direction in {"asc", "desc"}:
                    _hf_sort_action(sort_key, descending=direction == "desc")
            elif isinstance(sort_items, list):
                # TanStack's third click clears its client sort. This table is
                # server-owned, so keep header sorting two-state instead of
                # leaving an arrowless client state over still-sorted rows.
                _hf_sort_action(
                    STATE.hf.local_sort,
                    descending=not STATE.hf.local_sort_desc,
                )
            return

        if kind == "page":
            _update_hf_widgets()
            return
        return

    if _is_active_action("hf-inspect-row"):
        payload = _active_action_value()
        repo_id = str(payload.get("repo_id") or "").strip()
        repo_type = str(payload.get("repo_type") or "")
        if repo_id and repo_type in {"model", "dataset"}:
            _hf_inspect_action(repo_id, repo_type, "")
        return
    if _is_active_action("hf-download-file"):
        payload = _active_action_value()
        repo_id = str(payload.get("repo_id") or "").strip()
        repo_type = str(payload.get("repo_type") or "")
        file_path = str(payload.get("file") or "").strip()
        if not repo_id or repo_type not in {"model", "dataset"} or not file_path:
            lcars.notify("The selected Hub file could not be queued.", level="error")
            return
        try:
            STATE.hf.start_file_download(repo_id, repo_type, file_path)  # type: ignore[arg-type]
            lcars.notify(f"Queued {file_path} from {repo_id}.")
            _update_hf_widgets()
            _append_hf_logs()
        except Exception as exc:
            lcars.notify(f"File download not queued: {exc}", level="error")
        return
    if _is_active_action("hf-use-row"):
        payload = _active_action_value()
        repo_id = str(payload.get("repo_id") or "").strip()
        repo_type = str(payload.get("repo_type") or "")
        if repo_id and repo_type in {"model", "dataset"}:
            _hf_use_repo_action(repo_id, repo_type)
        else:
            lcars.notify("The selected repository could not be applied.", level="error")
        return
    if _is_active_action("hf-related-row"):
        payload = _active_action_value()
        repo_id = str(payload.get("repo_id") or "").strip()
        if repo_id:
            _hf_related_action(repo_id)
        else:
            lcars.notify("The selected model has no repository id.", level="error")


def _hf_selected_text_options(repo_id: str, repo_type: str) -> lcars.TextOptions:
    repo_id = repo_id.strip()
    return lcars.TextOptions(
        description="The table selection and repository commands target this owner/name id.",
        wrap="wrap",
        selectable=True,
        copyable=bool(repo_id),
        link=lcars.LinkSpec(
            href=_hf_repo_url(repo_id, repo_type),
            target="_blank",
            rel="noopener noreferrer",
        )
        if repo_id
        else None,
    )


def _hf_selection_metric(
    repo_id: str,
    repo_type: str,
    result: Any,
) -> tuple[str, str, Literal["ok", "warn", "crit"]]:
    """Summarize the command target without conflating it with the active search."""

    repo_id = repo_id.strip()
    type_label = "DATASET" if repo_type == "dataset" else "MODEL"
    if not repo_id:
        return (
            "NO TARGET",
            "Select a result row or inspect an exact owner/repository id.",
            "warn",
        )
    if result is None:
        return (
            f"{type_label} · SAVED TARGET",
            "Expand or inspect this repository to classify its files and compatibility.",
            "warn",
        )

    role_labels = {
        "base_model": "BASE MODEL",
        "peft_adapter": "PEFT ADAPTER",
        "runtime_quant": "RUNTIME ONLY",
        "dataset": "DATASET",
    }
    parts = [type_label]
    role = role_labels.get(str(result.role), "")
    if role and role != type_label:
        parts.append(role)
    if result.fit:
        parts.append(str(result.fit).upper())
    if result.file_count:
        parts.append(f"{result.file_count:,} FILES")

    detail = str(result.compatibility or "Repository metadata loaded.")
    if result.blocked:
        status: Literal["ok", "warn", "crit"] = "crit"
    elif detail.startswith("OK"):
        status = "ok"
    else:
        status = "warn"
    return " · ".join(parts), detail, status


def _effective_hf_artifact_filter(repo_type: str, artifact_filter: str) -> str:
    """Prevent a saved model/dataset-only filter from blanking the opposite search type."""

    value = artifact_filter or HF_ARTIFACT_FILTER_OPTIONS[0]
    if repo_type == "dataset" and value not in {"any", "any artifact", "datasets"}:
        return HF_ARTIFACT_FILTER_OPTIONS[0]
    if repo_type == "model" and value == "datasets":
        return HF_ARTIFACT_FILTER_OPTIONS[0]
    return value


def _current_hf_filter_summary() -> str:
    repo_type = _widget_value("hf-search-repo-type", STATE.hf.last_repo_type)
    sort = _widget_value("hf-sort", HF_SORT_OPTIONS[0]).replace("_", " ")
    limit = _widget_value("hf-limit", HF_LIMIT_OPTIONS[0])
    compatibility = _widget_value(
        "hf-compatibility",
        HF_COMPATIBILITY_OPTIONS[0],
    )
    artifact = _effective_hf_artifact_filter(
        repo_type,
        _widget_value("hf-artifact-filter", HF_ARTIFACT_FILTER_OPTIONS[0]),
    )
    quant = _widget_value("hf-quant-filter", HF_QUANT_FILTER_OPTIONS[0])
    fit = _widget_value("hf-fit-filter", HF_FIT_FILTER_OPTIONS[0])
    sift = _widget_value("hf-sift", "").strip()
    vram = _widget_value(
        "hf-vram-limit",
        str(STATE.hf.vram_limit_gb or 24),
    )

    parts = [
        f"{limit} {repo_type} results",
        f"Hub sort {sort}",
        (
            "compatible only"
            if compatibility == HF_COMPATIBILITY_OPTIONS[0]
            else "warnings + blocked included"
        ),
    ]
    if sift:
        parts.append(f"metadata contains {sift}")
    if artifact not in {"any", "any artifact"}:
        parts.append(artifact)
    if repo_type == "model":
        if quant != HF_QUANT_FILTER_OPTIONS[0]:
            parts.append(quant)
        if fit != HF_FIT_FILTER_OPTIONS[0]:
            parts.append(f"{fit} @ {vram}GB")
        else:
            parts.append(f"{vram}GB fit reference")
    return "ACTIVE FILTERS · " + " · ".join(parts)


def _ollama_rule_rows() -> list[dict[str, str]]:
    return [
        {
            "Source": "Local Transformers dir",
            "Action": "Apply",
            "Reason": "config/tokenizer/weights can be read by Axolotl",
        },
        {
            "Source": "hf.co / model name",
            "Action": "Search HF",
            "Reason": "find original safetensors repo or compatible fine-tune",
        },
        {
            "Source": "GGUF/internal blob",
            "Action": "Block",
            "Reason": "runtime artifact, not an Axolotl base_model",
        },
    ]


def _lora_preset_options(recommended_key: str) -> list[lcars.SelectOption]:
    return [
        lcars.SelectOption(
            label=(
                f"{preset.label} · RECOMMENDED FOR THIS GPU"
                if preset.key == recommended_key
                else preset.label
            ),
            value=preset.key,
            description=(
                f"{preset.summary} Best for: {preset.best_for}. "
                f"{preset.method}; {preset.speed.lower()}."
            ),
        )
        for preset in LORA_PRESETS
    ]


def _lora_preset_rows(recommended_key: str) -> list[dict[str, str]]:
    return [
        {
            "Fit": "RECOMMENDED" if preset.key == recommended_key else "OPTION",
            "Preset": preset.label,
            "Method": preset.method,
            "Run shape": preset.speed,
            "Best for": preset.best_for,
            "Hardware": preset.hardware,
        }
        for preset in LORA_PRESETS
    ]


def _lora_model_template_rows() -> list[dict[str, str]]:
    return [
        {
            "Family": template.family,
            "Model": template.model_id,
            "Architecture": template.architecture,
            "First recipe": get_lora_preset(template.default_preset).label,
            "Adapter scope": (
                "Attention + shared text paths; routed experts frozen"
                if template.moe
                else "Text decoder only"
            ),
            "Hardware": template.hardware,
            "Requires": f"Axolotl {template.min_axolotl}+",
        }
        for template in LORA_MODEL_TEMPLATES
    ]


def _lora_dataset_cache_rows() -> list[dict[str, str]]:
    return STATE.hf.dataset_cache_rows()


def _lora_downloaded_dataset_cache_rows(
    rows: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    rows = _lora_dataset_cache_rows() if rows is None else rows
    return [
        row
        for row in rows
        if str(row.get("Type") or "").strip().casefold() in {"dataset", "datasets"}
        and str(row.get("Repo") or "").strip()
        and str(row.get("Status") or "READY").strip().upper() == "READY"
    ]


def _lora_downloaded_dataset_options(
    rows: list[dict[str, str]],
) -> list[lcars.SelectOption]:
    if not rows:
        return [
            lcars.SelectOption(
                label="No completed dataset downloads found",
                value="",
                description="Download a dataset from HF Hub, then return to this page.",
            )
        ]
    return [
        lcars.SelectOption(
            label=f"{row['Repo']} · {row.get('Size') or 'size unknown'}",
            value=row["Repo"],
            description=(
                f"Cached revision {row.get('Revision') or 'unknown'} at "
                f"{row.get('Path') or 'the Hugging Face cache'}."
            ),
        )
        for row in rows
    ]


def _lora_hf_dataset_format_options() -> list[lcars.SelectOption]:
    return [
        lcars.SelectOption(
            label=dataset_format.label,
            value=dataset_format.key,
            description=f"{dataset_format.summary} Shape: {dataset_format.record_shape}",
        )
        for dataset_format in LORA_HF_DATASET_FORMATS
    ]


def _infer_lora_hf_dataset_format(cfg: dict[str, Any]) -> str:
    dataset_type = str(_config_path_value(cfg, "datasets.0.type") or "").casefold()
    messages_field = str(
        _config_path_value(cfg, "datasets.0.field_messages") or ""
    ).casefold()
    if dataset_type == "completion":
        return "plain-text"
    if dataset_type == "alpaca":
        return "alpaca"
    if dataset_type == "sharegpt" or messages_field == "conversations":
        return "sharegpt"
    return "openai-messages"


def _lora_data_source_label(
    report: DatasetReport,
    cached_dataset_ids: tuple[str, ...],
) -> str:
    if not report.source:
        return "NOT CHOSEN"
    if report.source_kind == "local":
        return "LOCAL JSONL"
    if report.source in cached_dataset_ids:
        return "DOWNLOADED HF"
    return "HF REPOSITORY"


def _lora_active_dataset_summary(
    cfg: dict[str, Any],
    report: DatasetReport,
    cached_dataset_ids: tuple[str, ...],
) -> str:
    source_label = _lora_data_source_label(report, cached_dataset_ids)
    dataset_type = str(_config_path_value(cfg, "datasets.0.type") or "not set")
    format_key = _infer_lora_hf_dataset_format(cfg)
    format_label = get_lora_dataset_format(format_key).label
    split = str(_config_path_value(cfg, "datasets.0.split") or "train")
    subset = str(_config_path_value(cfg, "datasets.0.name") or "none")
    contents = (
        "rows checked when preprocessing starts"
        if report.example_count is None
        else (
            f"{report.example_count} examples / {report.message_count} messages / "
            f"{report.placeholder_count} placeholders"
        )
    )
    return (
        f"TRAINING WILL READ: {report.source or 'not chosen'} · {source_label} · "
        f"{dataset_type} / {format_label} · split {split} · subset {subset} · {contents}"
    )


def _lora_dataset_download_rows(
    cache_rows: list[dict[str, str]],
    configured_dataset: str,
) -> list[dict[str, str]]:
    rows = [
        {
            "Dataset": row["Repo"],
            "Status": (
                "IN USE"
                if row["Repo"] == configured_dataset
                and str(row.get("Status") or "READY").upper() == "READY"
                else str(row.get("Status") or "READY").upper()
            ),
            "Size": row.get("Size") or "",
            "Revision": row.get("Revision") or "",
            "Cache path": row.get("Path") or "",
            "What to do": row.get("Problem") or "Ready to select below.",
        }
        for row in cache_rows
    ]
    cached_ids = {row["Repo"] for row in cache_rows}
    for job in STATE.hf.job_rows():
        repo_id = str(job.get("Repo") or "")
        if (
            str(job.get("Type") or "").casefold() != "dataset"
            or not repo_id
            or repo_id in cached_ids
        ):
            continue
        rows.append(
            {
                "Dataset": repo_id,
                "Status": str(job.get("Status") or "queued").upper(),
                "Size": str(job.get("Estimate") or ""),
                "Revision": str(job.get("Revision") or ""),
                "Cache path": str(job.get("Local Path") or ""),
                "What to do": _lora_dataset_job_guidance(
                    str(job.get("Status") or "queued")
                ),
            }
        )
    if not rows:
        return [
            {
                "Dataset": "No dataset downloads found",
                "Status": "OPEN HF HUB",
                "Size": "",
                "Revision": "",
                "Cache path": "",
                "What to do": "Find and download a dataset from HF Hub.",
            }
        ]
    return sorted(rows, key=lambda row: (row["Status"] != "IN USE", row["Dataset"].casefold()))


def _lora_dataset_job_guidance(status: str) -> str:
    normalized = status.strip().casefold()
    if normalized == "failed":
        return "Open Content for the error, then download again."
    if normalized == "complete":
        return "Refresh the cache if this does not become READY."
    return "Wait for the transfer to finish; this page updates automatically."


def _lora_dataset_cache_notice(
    cache_rows: list[dict[str, str]],
    ready_rows: list[dict[str, str]],
) -> str:
    incomplete = [
        row
        for row in cache_rows
        if str(row.get("Status") or "").strip().upper() == "INCOMPLETE"
    ]
    if incomplete:
        names = ", ".join(str(row.get("Repo") or "unknown") for row in incomplete[:3])
        extra = f" and {len(incomplete) - 3} more" if len(incomplete) > 3 else ""
        ready_note = (
            " Other completed datasets remain selectable below."
            if ready_rows
            else " No dataset can be selected until a download reaches READY."
        )
        return (
            f"Incomplete Hugging Face dataset cache found: {names}{extra}. "
            "Its snapshot is missing or unreadable, so training cannot safely use it. "
            "Return to HF Hub and download that dataset again."
            f"{ready_note}"
        )
    return (
        "No completed dataset downloads were found in the Hugging Face cache. "
        "Open HF Hub, search with Repo Type = dataset, inspect and download one, "
        "then return here. The transfer appears automatically when complete."
    )


def _lora_tuning_rows(cfg: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "Setting": hint.label,
            "Current": _lora_setting_text(hint.key, cfg.get(hint.key)),
            "Starter range": hint.starter_range,
            "Change it when": hint.tune_when,
            "Tradeoff": hint.tradeoff,
        }
        for hint in LORA_TUNING_HINTS
    ]


def _lora_setting_text(key: str, value: Any) -> str:
    if value is None or value == "":
        return "not set"
    if isinstance(value, bool):
        return "on" if value else "off"
    if key == "learning_rate":
        try:
            return f"{float(value):.6f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            pass
    return str(value)


def _lora_journey_rows(
    cfg: dict[str, Any],
    dataset: DatasetReport,
    has_artifact: bool,
) -> list[dict[str, str]]:
    setup_ready = bool(
        str(cfg.get("base_model") or "").strip()
        and str(cfg.get("adapter") or "") in {"lora", "qlora"}
        and str(cfg.get("output_dir") or "").strip()
    )
    training_ready = (
        _lora_dataset_trainable(cfg, dataset)
        and STATE.runner.axolotl_path is not None
        and not any(issue.severity == "error" for issue in STATE.preflight_issues)
    )
    test_ready = has_artifact and bool(STATE.ollama.models)
    return [
        {
            "Step": "1 · Setup",
            "Status": "READY" if setup_ready else "START HERE",
            "What happens": "Choose a goal, base model, and memory profile",
            "Page": "?page=lora-setup",
        },
        {
            "Step": "2 · Data",
            "Status": (
                "READY" if _lora_dataset_trainable(cfg, dataset) else "NEEDS EXAMPLES"
            ),
            "What happens": "Write and validate the ideal conversations to imitate",
            "Page": "?page=lora-data",
        },
        {
            "Step": "3 · Train",
            "Status": "READY" if training_ready else "WAITING",
            "What happens": "Run preflight, preprocess, train, and watch resources",
            "Page": "?page=lora-train",
        },
        {
            "Step": "4 · Test",
            "Status": "READY" if test_ready else "WAITING",
            "What happens": "Build an Ollama adapter model and compare responses",
            "Page": "?page=lora-test",
        },
    ]


def _lora_training_plan_rows(cfg: dict[str, Any]) -> list[dict[str, str]]:
    micro_batch = _bounded_int(
        str(cfg.get("micro_batch_size") or 1),
        default=1,
        minimum=1,
        maximum=1_000_000,
    )
    accumulation = _bounded_int(
        str(cfg.get("gradient_accumulation_steps") or 1),
        default=1,
        minimum=1,
        maximum=1_000_000,
    )
    adapter = str(cfg.get("adapter") or "unset")
    memory = (
        "4-bit base (QLoRA)"
        if cfg.get("load_in_4bit")
        else ("8-bit base (LoRA)" if cfg.get("load_in_8bit") else "full-precision base")
    )
    model_template = get_lora_model_template(str(cfg.get("base_model") or ""))
    rows = [
        {
            "Choice": "Smart preset",
            "Value": get_lora_preset(infer_lora_preset(cfg)).label,
            "Why": "Closest guided recipe to the active YAML values",
        },
        {
            "Choice": "Base model",
            "Value": str(cfg.get("base_model") or "not chosen"),
            "Why": "The general model whose behavior the adapter changes",
        },
        {
            "Choice": "Goal layer",
            "Value": adapter.upper(),
            "Why": "Saves only a small adapter instead of another full model",
        },
        {
            "Choice": "Training data",
            "Value": str(_config_path_value(cfg, "datasets.0.path") or "not chosen"),
            "Why": "Examples of the exact behavior to imitate",
        },
        {
            "Choice": "Memory mode",
            "Value": memory,
            "Why": "Controls how the base model is loaded on the GPU",
        },
        {
            "Choice": "Context",
            "Value": f"{cfg.get('sequence_len') or 0} tokens",
            "Why": "Maximum training-example length",
        },
        {
            "Choice": "Effective batch",
            "Value": f"{micro_batch * accumulation} examples",
            "Why": f"{micro_batch} at once × {accumulation} accumulation steps per GPU",
        },
        {
            "Choice": "Duration",
            "Value": f"{cfg.get('num_epochs') or 0} pass(es) through the data",
            "Why": "A small first run helps reveal data and memory problems",
        },
        {
            "Choice": "Output",
            "Value": str(cfg.get("output_dir") or "not chosen"),
            "Why": "Where checkpoints and the final adapter are written",
        },
    ]
    if model_template is not None:
        rows.insert(
            2,
            {
                "Choice": "Architecture template",
                "Value": f"{model_template.family} · {model_template.architecture}",
                "Why": "Sets the correct chat format and text-only LoRA targets",
            },
        )
    return rows


def _lora_training_brief_rows(
    cfg: dict[str, Any],
    dataset: DatasetReport,
) -> list[dict[str, str]]:
    examples = (
        "Hub dataset"
        if dataset.example_count is None
        else f"{dataset.example_count} local example(s)"
    )
    micro_batch = _bounded_int(
        str(cfg.get("micro_batch_size") or 1),
        default=1,
        minimum=1,
        maximum=1_000_000,
    )
    accumulation = _bounded_int(
        str(cfg.get("gradient_accumulation_steps") or 1),
        default=1,
        minimum=1,
        maximum=1_000_000,
    )
    return [
        {
            "Check": "Preset",
            "Value": get_lora_preset(infer_lora_preset(cfg)).label,
        },
        {
            "Check": "Method",
            "Value": str(cfg.get("adapter") or "unset").upper(),
        },
        {
            "Check": "Data",
            "Value": examples,
        },
        {
            "Check": "Shape",
            "Value": (
                f"rank {cfg.get('lora_r') or 'unset'} · "
                f"{cfg.get('sequence_len') or 'unset'} tokens · "
                f"batch {micro_batch * accumulation}"
            ),
        },
        {
            "Check": "Duration",
            "Value": f"{cfg.get('num_epochs') or 'unset'} epoch(s)",
        },
        {
            "Check": "Output",
            "Value": str(cfg.get("output_dir") or "unset"),
        },
    ]


def _lora_dataset_issue_rows(report: DatasetReport) -> list[dict[str, str]]:
    rows = [
        {"Level": "ERROR", "Detail": detail}
        for detail in report.errors
    ]
    rows.extend({"Level": "ADVICE", "Detail": detail} for detail in report.warnings)
    if not rows:
        rows.append(
            {
                "Level": "OK",
                "Detail": "The configured dataset has valid structure and no draft placeholders.",
            }
        )
    return rows


def _lora_dataset_trainable(cfg: dict[str, Any], dataset: DatasetReport) -> bool:
    if not dataset.ready:
        return False
    if dataset.source_kind != "local" or dataset.example_count is None:
        return True
    try:
        validation_size = float(cfg.get("val_set_size") or 0)
    except (TypeError, ValueError):
        validation_size = 0.0
    return validation_size <= 0 or dataset.example_count >= 2


def _lora_dataset_status(cfg: dict[str, Any], dataset: DatasetReport) -> str:
    if dataset.ready and not _lora_dataset_trainable(cfg, dataset):
        return "ADD ONE MORE"
    return dataset.status


def _lora_gate_detail(
    errors: list[PreflightIssue],
    dataset: DatasetReport,
    cfg: dict[str, Any] | None = None,
) -> str:
    if not dataset.ready:
        if dataset.errors:
            return dataset.errors[0]
        if dataset.placeholder_count:
            return "Replace every EDIT ME placeholder first."
        return "Finish the training dataset first."
    if cfg is not None and not _lora_dataset_trainable(cfg, dataset):
        return "Add at least one more example so training and validation both have data."
    if errors:
        return errors[0].detail
    if STATE.workflow.is_active:
        return "Another workflow currently owns the active config."
    return "Dataset and Axolotl preflight are ready."


def _lora_elapsed_text() -> str:
    started = STATE.runner.state.started_at
    if started is None:
        return "—"
    ended = STATE.runner.state.ended_at
    seconds = max(0, int((ended if ended is not None else time.time()) - started))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _lora_process_progress() -> float:
    if STATE.runner.is_running():
        return 50.0
    if STATE.runner.state.status == "complete":
        return 100.0
    return 0.0


def _lora_artifact_rows(artifacts: list[Any]) -> list[dict[str, str]]:
    if not artifacts:
        return [
            {
                "State": "No adapter yet",
                "Path": "",
                "Size": "",
                "Modified": "",
                "Base Model": "",
            }
        ]
    return [
        {
            "State": "Ready for testing",
            "Path": str(artifact.path),
            "Size": format_bytes(artifact.size_bytes),
            "Modified": artifact.modified_text,
            "Base Model": artifact.base_model or "not recorded",
        }
        for artifact in artifacts
    ]


def _lora_ollama_select_options(names: list[str]) -> list[lcars.SelectOption]:
    if not names:
        return [
            lcars.SelectOption(
                label="No local Ollama models detected",
                value="",
                description="Start Ollama and install a matching base model.",
            )
        ]
    return [
        lcars.SelectOption(
            label="Choose model explicitly",
            value="",
        ),
        *[lcars.SelectOption(label=name, value=name) for name in names],
    ]


def _lora_architecture_hint(base_model: str) -> str:
    lowered = base_model.lower()
    if "llama" in lowered:
        return "Llama family"
    if "mistral" in lowered or "mixtral" in lowered:
        return "Mistral family"
    if "gemma" in lowered:
        return "Gemma family"
    return "Check Ollama support before importing"


def _render_config_fields(
    groups: set[str] | None = None,
    *,
    keys: set[str] | None = None,
    include_headers: bool = True,
    id_prefix: str = "",
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    current_group = ""
    cfg = _load_config_or_empty()
    group_filter = set(groups or CONFIG_GROUP_ORDER)
    ordered_specs = sorted(
        enumerate(FIELD_SPECS),
        key=lambda item: (
            CONFIG_GROUP_ORDER.index(item[1].group) if item[1].group in CONFIG_GROUP_ORDER else 999,
            item[0],
        ),
    )
    for _, spec in ordered_specs:
        if spec.group not in group_filter:
            continue
        if keys is not None and spec.key not in keys:
            continue
        if spec.group != current_group:
            current_group = spec.group
            if include_headers:
                safe_group_id = current_group.lower().replace(" ", "-").replace("/", "")
                prefix = f"{id_prefix}-" if id_prefix else ""
                lcars.header(
                    current_group, size="h3", color="pale-canary", id=f"hdr-{prefix}{safe_group_id}"
                )
                note = CONFIG_GROUP_NOTES.get(current_group)
                if note:
                    lcars.text(note, id=f"note-{prefix}{safe_group_id}")
        values[spec.widget_id] = _render_field(spec, cfg)
    return values


def _render_field(spec: FieldSpec, cfg: dict[str, Any]) -> Any:
    value = STATE.config_store.control_value(spec, cfg)
    label = _field_label(spec)
    help_text = _field_help(spec)
    hint_text = (
        help_text
        if spec.key in CONFIG_FIELD_HINTS or lora_tuning_hint(spec.key)
        else None
    )
    if spec.kind in {"text", "csv_list", "json"}:
        # force: the active YAML owns config values, so a rebuilt manifest must not
        # keep showing what the previous build seeded.
        _seed_text(spec.widget_id, str(value or ""), force=True)
        if spec.kind == "json":
            return lcars.text_input(
                label,
                value=str(value),
                placeholder=spec.placeholder or "{key: value}",
                autocomplete=False,
                id=spec.widget_id,
                options=lcars.TextInputOptions(
                    multiline=True,
                    rows=4,
                    description=help_text,
                    validation=lcars.ValidationOptions(
                        required=spec.key in SETUP_REQUIRED_KEYS,
                    ),
                ),
                hint=hint_text,
            )
        return lcars.text_input(
            label,
            value=str(value),
            placeholder=spec.placeholder,
            autocomplete=False,
            id=spec.widget_id,
            options=lcars.TextInputOptions(
                description=help_text,
                validation=lcars.ValidationOptions(
                    required=spec.key in SETUP_REQUIRED_KEYS,
                ),
            ),
            hint=hint_text,
        )
    if spec.kind == "number":
        if spec.optional:
            _seed_text(spec.widget_id, "" if value in (None, "") else str(value), force=True)
            return lcars.text_input(
                label,
                value=str(value),
                placeholder=spec.placeholder or "unset",
                autocomplete=False,
                id=spec.widget_id,
                options=lcars.TextInputOptions(
                    input_type="text",
                    description=(
                        f"Optional numeric value; leave blank for the Axolotl default. "
                        f"Step: {spec.step:g}. {help_text}"
                    ),
                    validation=lcars.ValidationOptions(
                        pattern=r"^-?(?:\d+(?:\.\d*)?|\.\d+)?$",
                        message="Enter a number or leave the field empty.",
                    ),
                ),
                hint=hint_text,
            )
        return lcars.number_input(
            label,
            value=float(value),
            min=spec.minimum,
            max=spec.maximum,
            step=spec.step,
            id=spec.widget_id,
            options=lcars.NumberInputOptions(
                precision=_step_precision(spec.step),
                required=True,
                description=help_text,
            ),
            hint=hint_text,
        )
    if spec.kind == "bool":
        return lcars.toggle(
            label,
            value=bool(value),
            id=spec.widget_id,
            options=lcars.ToggleOptions(
                on_label="Enabled",
                off_label="Disabled",
                description=help_text,
            ),
            hint=hint_text,
        )
    if spec.kind == "tri_bool":
        selected = str(value if value not in (None, "") else "unset")
        if selected not in {"unset", "true", "false"}:
            selected = "true" if selected.lower() in {"1", "yes", "on"} else selected
        return lcars.select(
            label,
            [
                lcars.SelectOption(
                    label="Unset / Axolotl default",
                    value="unset",
                    description=CONFIG_VALUE_HINTS.get(spec.key, {}).get("unset"),
                ),
                lcars.SelectOption(
                    label="Enabled",
                    value="true",
                    description=CONFIG_VALUE_HINTS.get(spec.key, {}).get("true"),
                ),
                lcars.SelectOption(
                    label="Disabled",
                    value="false",
                    description=CONFIG_VALUE_HINTS.get(spec.key, {}).get("false"),
                ),
            ],
            value=selected,
            id=spec.widget_id,
            settings=lcars.ChoiceOptions(description=help_text),
            hint=hint_text,
        )
    selected = str(value if value not in (None, "") else (spec.default or ""))
    return lcars.select(
        label,
        _config_select_options(spec, selected),
        value=selected,
        id=spec.widget_id,
        settings=lcars.ChoiceOptions(
            searchable=len(spec.options) > 6,
            description=help_text,
        ),
        hint=hint_text,
    )


def _config_select_options(
    spec: FieldSpec,
    selected: str,
) -> list[lcars.SelectOption]:
    """Label unset choices and preserve custom YAML values instead of blanking."""

    options = [
        lcars.SelectOption(
            label="Unset / Axolotl default" if value == "" else value,
            value=value,
            description=CONFIG_VALUE_HINTS.get(spec.key, {}).get(value),
        )
        for value in spec.options
    ]
    if selected and selected not in spec.options:
        options.insert(
            0,
            lcars.SelectOption(
                label=f"{selected} · custom YAML value",
                value=selected,
                description="Preserved from the active config.",
            ),
        )
    return options


def _field_help(spec: FieldSpec) -> str:
    specific = CONFIG_FIELD_HINTS.get(spec.key) or lora_tuning_hint(spec.key)
    if specific:
        return f"{specific} Axolotl key: {spec.key}."
    expected = f" Expected value: {spec.placeholder}." if spec.placeholder else ""
    optional = (
        " Leave it unset unless this run specifically needs it."
        if spec.optional
        else ""
    )
    return (
        f"Axolotl key: {spec.key}. Part of {spec.group}.{expected}{optional}"
    )


def _field_label(spec: FieldSpec) -> str:
    if spec.key in SETUP_REQUIRED_KEYS:
        return f"{spec.label} [required]"
    if spec.optional:
        return f"{spec.label} [optional]"
    if spec.default is not None:
        return f"{spec.label} [ui {spec.default}]"
    return spec.label


def _config_page_actions(suffix: str) -> None:
    with lcars.control_panel(
        "Page Actions",
        color="golden-tanoi",
        id=f"config-actions-{suffix}",
        weight=3,
        aspect="wide",
        group=f"{suffix}-fields",
        options=COLLAPSIBLE_PANEL_OPTIONS,
    ):
        if lcars.button(
            "Save Config",
            color="tanoi",
            id=f"config-save-{suffix}",
            hint=(
                "Write every structured control on this page into the active YAML while preserving "
                "fields owned by the other config pages."
            ),
        ):
            _save_config_action()
        if lcars.button(
            "Run Preflight",
            color="anakiwa",
            id=f"config-preflight-{suffix}",
            hint=(
                "Validate model, dataset, precision, attention, distributed, resume, auth, and "
                "integration settings without starting Axolotl."
            ),
        ):
            _run_preflight_action()
        lcars.markdown(
            "[Raw YAML editor](/raw)",
            id=f"raw-link-{suffix}",
            options=lcars.MarkdownOptions(link_target="_self"),
        )


def _switch_config_action(selected: str) -> None:
    if _workflow_blocks_config_change():
        return
    try:
        STATE.config_store.set_active(selected)
        lcars.notify(f"Active config switched to {selected}.")
        _update_config_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Could not switch config: {exc}", level="error")


def _create_config_action(new_name: str) -> None:
    if _workflow_blocks_config_change():
        return
    try:
        created = STATE.config_store.create_named(new_name.strip())
        lcars.notify(f"Created config {created}.")
        _update_config_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Could not create config: {exc}", level="error")


def _coverage_rows() -> list[dict[str, str]]:
    counts = {group: 0 for group in CONFIG_GROUP_ORDER}
    for spec in FIELD_SPECS:
        counts[spec.group] = counts.get(spec.group, 0) + 1
    return [
        {
            "Page": _page_for_group(group),
            "Group": group,
            "Fields": str(counts[group]),
            "Role": CONFIG_GROUP_NOTES.get(group, ""),
        }
        for group in CONFIG_GROUP_ORDER
        if counts.get(group)
    ]


def _page_for_group(group: str) -> str:
    if group in {"Run Safety", "Model", "Dataset", "Sequence / Packing"}:
        return "Setup"
    if group in {"Training", "Adapter / PEFT", "Optimizer"}:
        return "Train"
    if group in {"Precision / Memory", "Attention / Kernels", "Distributed"}:
        return "Hardware"
    return "Tracking"


def _save_config_action(values: dict[str, Any] | None = None) -> None:
    if _workflow_blocks_config_change():
        return
    try:
        payload = values if values is not None else _collect_editor_values()
        STATE.config_store.save_editor_values(payload)
        lcars.notify("Structured config saved.")
        _update_config_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Config save failed: {exc}", level="error")


def _duplicate_config_action() -> None:
    if _workflow_blocks_config_change():
        return
    try:
        new_name = STATE.config_store.create_copy("copy-of-" + STATE.config_store.active_name)
        lcars.notify(f"Created config {new_name}.")
        _update_config_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Could not duplicate config: {exc}", level="error")


def _run_preflight_action() -> None:
    issues = STATE.refresh_preflight()
    _update_preflight_widgets(issues)
    _update_lora_widgets()
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warn")
    if errors:
        lcars.notify(
            f"Preflight blocked launch: {errors} error(s), {warnings} warning(s).", level="error"
        )
    else:
        lcars.notify(f"Preflight passed with {warnings} warning(s).")


def _workflow_canvas_options() -> lcars.NodeCanvasOptions:
    return lcars.NodeCanvasOptions(
        editable=not STATE.workflow.is_active,
        interaction=lcars.InteractionOptions(
            mode="server",
            action_id=WORKFLOW_CANVAS_ID,
        ),
        min_zoom=0.35,
        max_zoom=1.8,
        snap_to_grid=True,
        grid_size=16,
        minimap=False,
        allow_import_export=True,
        history_limit=75,
        show_palette=True,
        show_run=False,
        show_queue=False,
        show_cancel=False,
    )


def _workflow_status_severity() -> str:
    if STATE.workflow.status == "error":
        return "crit"
    if STATE.workflow.status in {"queued", "cancelled"}:
        return "warn"
    return "ok"


def _workflow_blocks_config_change() -> bool:
    if not STATE.workflow.is_active:
        return False
    lcars.notify(
        "The active config is locked until the workflow completes or is cancelled.",
        level="error",
    )
    return True


def _persist_workflow_document() -> None:
    changed = UI_STATE.set(
        "workflow_document",
        STATE.workflow.document.model_dump(mode="json"),
    )
    if changed:
        UI_STATE.save()


def _workflow_graph_action(state: lcars.NodeCanvasState) -> None:
    try:
        STATE.workflow.replace_document(
            state.document,
            STATE.config_store.active_name,
        )
        _persist_workflow_document()
        if state.last_event == "import":
            lcars.notify("Workflow graph imported and normalized to Axolotl stages.")
        _update_workflow_widgets()
    except Exception as exc:
        lcars.notify(f"Workflow edit rejected: {exc}", level="error")
        _update_workflow_widgets(include_document=True)


def _validate_workflow_action() -> None:
    try:
        plan = STATE.workflow.plan()
        issues = STATE.refresh_preflight()
        _update_preflight_widgets(issues)
        errors = [issue for issue in issues if issue.severity == "error"]
        sequence = " → ".join(step.label for step in plan)
        STATE.workflow.status = "idle"
        STATE.workflow.message = f"Valid plan: {sequence}."
        STATE.workflow.node_execution = {}
        if errors:
            lcars.notify(
                f"Workflow graph is valid, but preflight blocks launch: {errors[0].detail}",
                level="error",
            )
        else:
            lcars.notify(f"Workflow valid: {sequence}.")
        _update_workflow_widgets()
    except Exception as exc:
        STATE.workflow.status = "error"
        STATE.workflow.message = str(exc)
        lcars.notify(f"Workflow invalid: {exc}", level="error")
        _update_workflow_widgets()


def _start_workflow_action() -> None:
    try:
        issues = STATE.refresh_preflight()
        _update_preflight_widgets(issues)
        errors = [issue for issue in issues if issue.severity == "error"]
        if errors:
            raise WorkflowError(f"Preflight blocked launch: {errors[0].detail}")
        plan = STATE.workflow.plan()
        STATE.workflow.start(STATE.runner, STATE.config_store.active_path)
        lcars.notify(f"Workflow started: {' → '.join(step.label for step in plan)}.")
        _update_workflow_widgets()
        lcars.update(
            "run-status",
            value=STATE.runner.status_label(),
            status=STATE.runner.status_severity(),
        )
        lcars.update(
            "run-command-text",
            content=" ".join(STATE.runner.state.command),
        )
    except Exception as exc:
        lcars.notify(f"Unable to start workflow: {exc}", level="error")
        _update_workflow_widgets()


def _cancel_workflow_action() -> None:
    try:
        STATE.workflow.cancel(STATE.runner)
        lcars.notify("Workflow cancelled.")
        _update_workflow_widgets()
        lcars.update(
            "run-status",
            value=STATE.runner.status_label(),
            status=STATE.runner.status_severity(),
        )
    except Exception as exc:
        lcars.notify(f"Unable to cancel workflow: {exc}", level="error")


def _reset_workflow_action() -> None:
    try:
        STATE.workflow.reset(STATE.config_store.active_name)
        _persist_workflow_document()
        lcars.notify("Starter workflow restored.")
        _update_workflow_widgets(include_document=True)
    except Exception as exc:
        lcars.notify(f"Unable to reset workflow: {exc}", level="error")


def _update_workflow_widgets(*, include_document: bool = False) -> None:
    canvas_payload: dict[str, Any] = {
        "execution": STATE.workflow.execution_state().model_dump(mode="json"),
        "options": _workflow_canvas_options().model_dump(mode="json"),
    }
    if include_document:
        canvas_payload["document"] = STATE.workflow.document.model_dump(mode="json")
    lcars.update(WORKFLOW_CANVAS_ID, **canvas_payload)
    severity = _workflow_status_severity()
    lcars.update(
        "workflow-status",
        value=STATE.workflow.status.upper(),
        status=severity,
    )
    lcars.update(
        "workflow-current-stage",
        content=STATE.workflow.current_label,
    )
    lcars.update("workflow-progress", value=STATE.workflow.progress_percent)
    lcars.update("workflow-message", content=STATE.workflow.message)
    lcars.update(
        "workflow-start",
        disabled=STATE.workflow.is_active or STATE.runner.is_running(),
    )
    lcars.update("workflow-cancel", disabled=not STATE.workflow.is_active)
    lcars.update(
        "workflow-validate",
        disabled=STATE.workflow.is_active,
    )
    lcars.update(
        "workflow-reset",
        disabled=STATE.workflow.is_active,
    )
    lcars.update(
        "run-start",
        disabled=STATE.workflow.is_active or STATE.runner.is_running(),
    )
    lcars.update("run-stop", disabled=not STATE.runner.is_running())


def _start_axolotl_action(
    action: str,
    *,
    launcher: str = "",
    cli_args: str = "",
    launcher_args: str = "",
) -> None:
    try:
        if STATE.workflow.is_active:
            lcars.notify(
                "A workflow is active. Cancel it before launching a single action.",
                level="error",
            )
            return
        if launcher and action not in LAUNCHER_ACTIONS:
            lcars.notify(
                f"{action} does not accept launcher mode. Clear Launcher and retry.", level="error"
            )
            return
        if launcher_args.strip() and not launcher:
            lcars.notify(
                "Launcher Args require python, accelerate, or torchrun launcher mode.",
                level="error",
            )
            return
        if action in CONFIG_ACTIONS:
            issues = STATE.refresh_preflight()
            _update_preflight_widgets(issues)
            errors = [issue for issue in issues if issue.severity == "error"]
            if errors:
                lcars.notify(
                    f"Axolotl launch blocked by preflight: {errors[0].detail}", level="error"
                )
                return
        STATE.runner.start(
            action,
            STATE.config_store.active_path,
            launcher=launcher,
            cli_args=cli_args,
            launcher_args=launcher_args,
        )
        lcars.notify("Axolotl process started.")
        lcars.update("run-status", value=STATE.runner.status_label(), status="ok")
        lcars.update("run-command-text", content=" ".join(STATE.runner.state.command))
        _update_workflow_widgets()
    except Exception as exc:
        lcars.notify(f"Unable to start Axolotl: {exc}", level="error")


def _stop_axolotl_action() -> None:
    if STATE.workflow.is_active:
        _cancel_workflow_action()
        return
    STATE.runner.stop()
    lcars.notify("Axolotl stop requested.")
    lcars.update(
        "run-status", value=STATE.runner.status_label(), status=STATE.runner.status_severity()
    )
    _update_workflow_widgets()


def _lora_use_downloaded_dataset_action(
    repo_id: str,
    format_key: str,
    split: str,
    subset: str,
) -> None:
    if _workflow_blocks_config_change():
        return
    try:
        cached_rows = _lora_downloaded_dataset_cache_rows()
        cached_ids = {row["Repo"] for row in cached_rows}
        if repo_id not in cached_ids:
            raise LoraStudioError(
                "That dataset is not a completed download. Wait for the transfer or refresh "
                "the HF cache list."
            )
        cfg = STATE.config_store.load()
        model_template = get_lora_model_template(str(cfg.get("base_model") or ""))
        updates = downloaded_dataset_config_updates(
            repo_id,
            format_key,
            split=split,
            subset=subset,
            use_top_level_chat_template=model_template is not None,
        )
        STATE.config_store.apply_updates(updates)
        _set_session_value("lora-hf-dataset", repo_id)
        _set_session_value("lora-hf-dataset-format", format_key)
        _set_session_value("lora-hf-dataset-split", split.strip() or "train")
        _set_session_value("lora-hf-dataset-subset", subset.strip())
        _update_config_widgets()
        issues = STATE.refresh_preflight()
        _update_preflight_widgets(issues)
        _update_lora_widgets()
        _update_lora_downloaded_dataset_widgets()
        selected_format = get_lora_dataset_format(format_key)
        lcars.notify(
            f"Using downloaded dataset {repo_id} as {selected_format.label}. "
            "Axolotl will reuse the Hugging Face cache. Next, review preflight on Train."
        )
    except Exception as exc:
        lcars.notify(f"Could not use downloaded dataset: {exc}", level="error")


def _lora_setup_action(
    project_name: str,
    goal: str,
    base_model: str,
    preset_key: str,
) -> None:
    if _workflow_blocks_config_change():
        return
    try:
        slug = normalize_project_name(project_name)
        if goal not in LORA_GOALS:
            raise LoraStudioError("Choose a guided LoRA goal.")
        if not base_model.strip():
            raise LoraStudioError("Choose a base model.")
        current_cfg = STATE.config_store.load()
        current_base_model = str(current_cfg.get("base_model") or "").strip()
        effective_preset_key = preset_key
        if (
            base_model.strip() != current_base_model
            and preset_key == infer_lora_preset(current_cfg)
        ):
            # A model change should inherit that model's safe default unless the
            # operator also made an explicit recipe change in the same form.
            effective_preset_key = recommend_lora_preset(
                _detected_gpu_vram_gb(),
                base_model,
            )
        config_name = (
            STATE.config_store.active_name
            if Path(STATE.config_store.active_name).stem == slug
            else f"{slug}.yml"
        )
        created = False
        if STATE.config_store.active_name != config_name:
            if config_name in STATE.config_store.list_configs():
                raise LoraStudioError(
                    f"{config_name} already exists. Switch to it on the Config page or "
                    "choose a new project name."
                )
            STATE.config_store.create_named(config_name)
            created = True
        STATE.config_store.apply_updates(
            beginner_config_updates(
                slug,
                base_model=base_model,
                preset=effective_preset_key,
            )
        )
        _set_widget_value("lora-project-name", slug)
        _set_session_value("lora-goal", goal)
        _set_session_value("lora-base-model", base_model)
        _set_session_value("lora-preset", effective_preset_key)
        _set_widget_value("lora-data-filename", f"{slug}.jsonl")
        _set_widget_value("lora-test-model-name", f"{slug}-lora")
        current_editor = _widget_value("lora-data-editor")
        if created or not current_editor.strip() or _lora_editor_is_generated_template(
            current_editor,
            project_name,
        ):
            _set_widget_value(
                "lora-data-editor",
                starter_dataset_template(goal, slug.replace("-", " ").title()),
            )
        _update_config_widgets()
        issues = STATE.refresh_preflight()
        _update_preflight_widgets(issues)
        _update_lora_widgets()
        model_template = get_lora_model_template(base_model)
        template_text = (
            f"{model_template.family} architecture defaults + "
            if model_template is not None
            else ""
        )
        lcars.notify(
            f"{template_text}{get_lora_preset(effective_preset_key).label} project {slug} saved. "
            "Next, replace the EDIT ME examples on the Data page."
        )
    except Exception as exc:
        lcars.notify(f"Could not save guided LoRA setup: {exc}", level="error")


def _lora_save_dataset_action(
    filename: str,
    editor_text: str,
    *,
    notify: bool = True,
) -> bool:
    if _workflow_blocks_config_change():
        return False
    try:
        model_template = get_lora_model_template(
            str(_load_config_or_empty().get("base_model") or "")
        )
        target, draft_report = save_chat_jsonl(
            PROJECT_ROOT,
            filename.strip(),
            editor_text,
        )
        STATE.config_store.apply_updates(
            {
                "datasets.0.path": f"./data/{target.name}",
                "datasets.0.type": "chat_template",
                "datasets.0.ds_type": "json",
                "datasets.0.field_messages": "messages",
                # Known model templates own message formatting at the top level.
                "datasets.0.chat_template": (
                    None if model_template is not None else "tokenizer_default"
                ),
                "datasets.0.roles_to_train": "assistant",
                "datasets.0.train_on_eos": "turn",
            }
        )
        _set_session_value("lora-data-filename", target.name)
        _set_session_value("lora-data-editor", editor_text)
        _update_config_widgets()
        issues = STATE.refresh_preflight()
        _update_preflight_widgets(issues)
        _update_lora_widgets()
        if notify and draft_report.placeholder_count:
            lcars.notify(
                f"Saved {draft_report.example_count} example(s) as a draft. Replace "
                f"{draft_report.placeholder_count} placeholder(s) before training."
            )
        elif notify:
            lcars.notify(
                f"Dataset saved and configured: {draft_report.example_count} example(s)."
            )
        return True
    except Exception as exc:
        lcars.notify(f"Dataset was not saved: {exc}", level="error")
        return False


def _lora_add_example_action(
    *,
    project_name: str,
    filename: str,
    user_prompt: str,
    ideal_response: str,
    system_prompt: str,
) -> None:
    try:
        line = chat_example_line(
            user_prompt,
            ideal_response,
            system_prompt=system_prompt,
        )
        current = _widget_value("lora-data-editor").strip()
        replace_template = not current or _lora_editor_is_generated_template(
            current,
            project_name,
        )
        if current and not replace_template:
            report = inspect_jsonl_text(current)
            if report.errors:
                raise LoraStudioError(
                    "The advanced JSONL editor contains an error. Fix it or load a fresh "
                    "template before adding form examples."
                )
        updated = line if replace_template else f"{current}\n{line}"
        _set_widget_value("lora-data-editor", updated)
        if _lora_save_dataset_action(filename, updated, notify=False):
            _set_widget_value("lora-example-user", "")
            _set_widget_value("lora-example-answer", "")
            lcars.notify(
                "Example added and saved. Add another with different wording or a harder case."
            )
    except Exception as exc:
        lcars.notify(f"Could not add the training example: {exc}", level="error")


def _lora_reset_dataset_template_action(goal: str, project_name: str) -> None:
    chosen_goal = goal if goal in LORA_GOALS else LORA_GOALS[0]
    try:
        display_name = normalize_project_name(project_name).replace("-", " ").title()
    except LoraStudioError:
        display_name = "The Assistant"
    template = starter_dataset_template(chosen_goal, display_name)
    _set_widget_value("lora-data-editor", template)
    report = inspect_jsonl_text(template)
    lcars.update(
        "lora-data-checks-table",
        **_table_payload(_lora_dataset_issue_rows(report)),
    )
    lcars.notify("Loaded a fresh draft template. Replace every EDIT ME response before saving.")


def _lora_validate_dataset_action() -> None:
    report = inspect_configured_dataset(PROJECT_ROOT, _load_config_or_empty())
    _update_lora_widgets()
    if report.ready:
        lcars.notify(
            "Dataset structure is ready. Quality still depends on the examples you wrote."
        )
    else:
        detail = (
            report.errors[0]
            if report.errors
            else f"{report.placeholder_count} placeholder(s) remain."
        )
        lcars.notify(f"Dataset needs attention: {detail}", level="error")


def _lora_start_action(action: str) -> None:
    cfg = _load_config_or_empty()
    dataset = inspect_configured_dataset(PROJECT_ROOT, cfg)
    if not _lora_dataset_trainable(cfg, dataset):
        lcars.notify(
            f"Training is blocked until the dataset is ready: "
            f"{_lora_gate_detail([], dataset, cfg)}",
            level="error",
        )
        return
    _start_axolotl_action(action)
    _update_lora_widgets()


def _lora_build_ollama_action(
    base_model: str,
    adapter_path: str,
    model_name: str,
) -> None:
    cfg = _load_config_or_empty()
    try:
        created = STATE.ollama.create_adapter_model(
            project_root=PROJECT_ROOT,
            model_name=model_name,
            base_model=base_model,
            adapter_path=adapter_path,
        )
        _set_session_value("lora-test-base-model", base_model)
        _set_widget_value("lora-test-chat-model", created.name)
        _set_widget_value("lora-test-compare-base", base_model)
        _set_widget_value("lora-test-model-name", model_name.strip())
        _update_lora_ollama_selects(
            preferred_base=base_model,
            preferred_tuned=created.name,
        )
        _update_lora_widgets()
        if str(cfg.get("adapter") or "") == "qlora":
            lcars.notify(
                f"Built {created.name}. This came from a QLoRA run; compare carefully because "
                "Ollama recommends non-quantized adapters for the smoothest import."
            )
        else:
            lcars.notify(f"Ollama test model ready: {created.name}.")
    except Exception as exc:
        lcars.notify(f"Could not build the Ollama adapter model: {exc}", level="error")
    finally:
        _append_ollama_logs()


def _lora_refresh_ollama_action() -> None:
    STATE.ollama.refresh()
    base = suggested_ollama_model(
        str(_load_config_or_empty().get("base_model") or ""),
        [model.name for model in STATE.ollama.models],
    )
    _update_lora_ollama_selects(preferred_base=base)
    _update_lora_widgets()
    if STATE.ollama.last_error:
        lcars.notify(STATE.ollama.last_error, level="error")
    else:
        lcars.notify(f"Detected {len(STATE.ollama.models)} local Ollama model(s).")


def _lora_compare_action(
    base_model: str,
    tuned_model: str,
    prompt: str,
    system_prompt: str,
) -> None:
    if not base_model.strip() or not tuned_model.strip():
        lcars.notify("Choose both a base model and a tuned model.", level="error")
        return
    if base_model.strip() == tuned_model.strip():
        lcars.notify(
            "Choose two different models so the comparison can reveal the adapter's effect.",
            level="error",
        )
        return
    try:
        base_result = STATE.ollama.chat(
            base_model,
            prompt,
            system_prompt=system_prompt,
            temperature=0.2,
        )
        STATE.lora_base_response = base_result.content
        STATE.lora_base_response_meta = _lora_chat_meta(base_result)
        tuned_result = STATE.ollama.chat(
            tuned_model,
            prompt,
            system_prompt=system_prompt,
            temperature=0.2,
        )
        STATE.lora_tuned_response = tuned_result.content
        STATE.lora_tuned_response_meta = _lora_chat_meta(tuned_result)
        lcars.update(
            "lora-base-response",
            content=STATE.lora_base_response,
            options=lcars.TextOptions(
                description=STATE.lora_base_response_meta,
                selectable=True,
                copyable=True,
                wrap="pre",
            ).model_dump(mode="json"),
        )
        lcars.update(
            "lora-tuned-response",
            content=STATE.lora_tuned_response,
            options=lcars.TextOptions(
                description=STATE.lora_tuned_response_meta,
                selectable=True,
                copyable=True,
                wrap="pre",
            ).model_dump(mode="json"),
        )
        lcars.notify(
            "Comparison complete. Look for the target behavior on prompts that were not trained."
        )
    except Exception as exc:
        lcars.notify(f"Ollama comparison failed: {exc}", level="error")
    finally:
        _append_ollama_logs()


def _lora_chat_meta(result: Any) -> str:
    speed = result.tokens_per_second
    speed_text = f" · {speed:.1f} token/s" if speed is not None else ""
    return f"{result.model} · {result.eval_count} generated token(s){speed_text}"


def _lora_editor_is_generated_template(text: str, project_name: str) -> bool:
    try:
        display_name = normalize_project_name(project_name).replace("-", " ").title()
    except LoraStudioError:
        display_name = project_name.strip() or "The Assistant"
    candidate = text.strip()
    return any(
        candidate == starter_dataset_template(goal, display_name).strip()
        for goal in LORA_GOALS
    )


def _setup_apply_recipe_action(recipe: str) -> None:
    if _workflow_blocks_config_change():
        return
    updates = SETUP_RECIPES.get(recipe)
    if not updates:
        lcars.notify("Unknown setup recipe.", level="error")
        return
    try:
        STATE.config_store.apply_updates(updates)
        lcars.notify(f"Applied setup recipe: {recipe}.")
        _update_config_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Could not apply setup recipe: {exc}", level="error")


def _setup_apply_model_action(model: str) -> None:
    if _workflow_blocks_config_change():
        return
    try:
        STATE.config_store.apply_model(model)
        lcars.notify(f"Applied model preset: {model}.")
        _update_config_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Could not apply model preset: {exc}", level="error")


def _setup_apply_dataset_action(dataset: str) -> None:
    if _workflow_blocks_config_change():
        return
    preset = DATASET_PRESETS.get(dataset)
    if preset is None:
        lcars.notify("Unknown dataset preset.", level="error")
        return
    try:
        path, dataset_type = preset
        STATE.config_store.apply_dataset(path, dataset_type)
        lcars.notify(f"Applied dataset preset: {path}.")
        _update_config_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Could not apply dataset preset: {exc}", level="error")


def _hf_search_action(
    query: str,
    repo_type: str,
    *,
    sort: str = "downloads",
    compatibility: str = HF_COMPATIBILITY_OPTIONS[0],
    limit: str = "12",
    sift: str = "",
    local_sort: str = "downloads",
    artifact_filter: str = "any",
    quant_filter: str = "any",
    fit_filter: str = "any",
    vram_limit: float | int | str = 0,
) -> None:
    if repo_type not in {"model", "dataset"}:
        lcars.notify("Repo type must be model or dataset.", level="error")
        return
    _reset_hf_results_page()
    _set_widget_value("hf-query", query)
    _set_widget_value("hf-search-repo-type", repo_type)
    effective_artifact_filter = _effective_hf_artifact_filter(
        repo_type,
        artifact_filter,
    )
    if effective_artifact_filter != artifact_filter:
        _set_widget_value("hf-artifact-filter", effective_artifact_filter)
        lcars.update("hf-artifact-filter", value=effective_artifact_filter)
    vram = _optional_float(vram_limit)
    STATE.hf.search(
        query,
        repo_type,  # type: ignore[arg-type]
        sort=sort,
        compatible_only=False,
        limit=_bounded_int(limit, default=12, minimum=1, maximum=50),
    )
    STATE.hf.vram_limit_gb = vram
    STATE.hf.hydrate_search_results()
    STATE.hf.sift_results(
        text=sift,
        sort=local_sort,
        descending=_kept_sort_direction(local_sort),
        compatible_only=compatibility == HF_COMPATIBILITY_OPTIONS[0],
        artifact_filter=effective_artifact_filter,
        quant_filter=quant_filter,
        fit_filter=fit_filter,
        vram_limit_gb=vram,
    )
    _set_session_value("hf-repo-type", repo_type)
    _set_session_value("hf-repo-id", STATE.hf.last_repo_id)
    _update_hf_widgets()
    _append_hf_logs()


def _hf_inspect_action(repo_id: str, repo_type: str, revision: str = "") -> None:
    repo_id = repo_id.strip()
    if not repo_id:
        lcars.notify("Repository ID is required.", level="error")
        return
    if repo_type not in {"model", "dataset"}:
        lcars.notify("Repo type must be model or dataset.", level="error")
        return
    STATE.hf.select_repository(repo_id, repo_type)  # type: ignore[arg-type]
    _set_session_value("hf-repo-id", repo_id)
    _set_session_value("hf-repo-type", repo_type)
    details = STATE.hf.inspect_repo(  # type: ignore[arg-type]
        repo_id,
        repo_type,
        revision=revision.strip() or None,
    )
    if details is not None:
        _set_session_value("hf-repo-id", details.result.repo_id)
        _set_session_value("hf-repo-type", details.result.repo_type)
        STATE.hf.set_expanded_result_ids(
            [
                *STATE.hf.expanded_result_ids,
                _hf_result_row_id(details.result),
            ]
        )
    else:
        error = STATE.hf.inspection_error_for(repo_id, repo_type)  # type: ignore[arg-type]
        lcars.notify(f"Repository inspection failed: {error or repo_id}", level="error")
    _update_hf_widgets()
    _append_hf_logs()


def _hf_related_action(repo_id: str) -> None:
    result = _hf_result_for(repo_id)
    if result is not None:
        STATE.hf.select_repository(result.repo_id, result.repo_type)
    STATE.hf.find_related_models(repo_id)
    if result is not None:
        STATE.hf.set_expanded_result_ids(
            [
                *STATE.hf.expanded_result_ids,
                _hf_result_row_id(result),
            ]
        )
    _update_hf_widgets()
    _append_hf_logs()


def _kept_sort_direction(sort_key: str) -> bool | None:
    """Keep a direction chosen from a column header when the sort key is unchanged."""

    if sort_key == STATE.hf.local_sort:
        return STATE.hf.local_sort_desc
    return None


def _hf_sort_action(sort_key: str, descending: bool | None = None) -> None:
    STATE.hf.sort_current_results(sort_key, descending=descending)
    _update_hf_widgets()
    _append_hf_logs()


def _hf_download_action(repo_id: str, repo_type: str, revision: str) -> None:
    if repo_type not in {"model", "dataset"}:
        lcars.notify("Repo type must be model or dataset.", level="error")
        return
    result = _hf_result_for(repo_id)
    if result is not None and result.blocked:
        lcars.notify(f"Download blocked: {result.compatibility}", level="error")
        return
    try:
        STATE.hf.start_download(repo_id, repo_type, revision=revision.strip() or None)  # type: ignore[arg-type]
        lcars.notify(f"Download queued for {repo_id}.")
        _update_hf_widgets()
    except Exception as exc:
        lcars.notify(f"Download failed to queue: {exc}", level="error")


def _hf_use_repo_action(repo_id: str, repo_type: str) -> None:
    if _workflow_blocks_config_change():
        return
    try:
        if repo_type == "model":
            result = _hf_result_for(repo_id)
            if result is not None and result.role == "peft_adapter":
                STATE.config_store.apply_updates({"lora_model_dir": repo_id, "adapter": "lora"})
            elif _looks_gguf(repo_id) or (result is not None and result.blocked):
                detail = (
                    result.compatibility if result is not None else "likely GGUF/runtime artifact"
                )
                lcars.notify(
                    f"Refusing to set incompatible model repo as Axolotl base_model: {detail}",
                    level="error",
                )
                return
            else:
                STATE.config_store.apply_model(repo_id)
        elif repo_type == "dataset":
            STATE.config_store.apply_dataset(repo_id)
        else:
            lcars.notify("Repo type must be model or dataset.", level="error")
            return
        lcars.notify(f"Applied {repo_type} {repo_id} to config.")
        _update_config_widgets()
        _update_hf_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Could not apply repo: {exc}", level="error")


def _hf_use_last_local_action(repo_type: str) -> None:
    if _workflow_blocks_config_change():
        return
    path = STATE.hf.last_local_path
    if not path:
        lcars.notify("No completed local HF snapshot is available yet.", level="error")
        return
    try:
        result = STATE.hf.selected_details.result if STATE.hf.selected_details else None
        if repo_type == "model" and result is not None and result.role == "peft_adapter":
            STATE.config_store.apply_updates({"lora_model_dir": path, "adapter": "lora"})
        elif repo_type == "model":
            STATE.config_store.apply_model(path)
        else:
            STATE.config_store.apply_dataset(path)
        lcars.notify(f"Applied local snapshot to config: {path}")
        _update_config_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Could not apply local snapshot: {exc}", level="error")


def _delete_cache_action(repo_id: str, repo_type: str) -> None:
    if repo_type not in {"model", "dataset"}:
        lcars.notify("Repo type must be model or dataset.", level="error")
        return
    try:
        freed = STATE.hf.delete_repo(repo_id, repo_type)  # type: ignore[arg-type]
        lcars.notify(f"Deleted cached repo; expected freed space {freed}.")
        _update_cache_widgets()
    except Exception as exc:
        lcars.notify(f"Cache delete failed: {exc}", level="error")


def _ollama_refresh_action() -> None:
    STATE.ollama.refresh()
    lcars.update(
        "ollama-table",
        **_table_payload(STATE.ollama.rows(), copy_columns={"Model", "Source"}),
    )
    _update_lora_ollama_selects(
        preferred_base=suggested_ollama_model(
            str(_load_config_or_empty().get("base_model") or ""),
            [model.name for model in STATE.ollama.models],
        )
    )
    _update_lora_widgets()
    if STATE.ollama.last_error:
        lcars.notify(STATE.ollama.last_error, level="error")
    else:
        lcars.notify(f"Detected {len(STATE.ollama.models)} Ollama model(s).")


def _ollama_search_hf_action(model_name: str) -> None:
    model = STATE.ollama.select(model_name.strip())
    if model is None:
        lcars.notify(
            "Ollama model was not found. Refresh and enter the exact name:tag.", level="error"
        )
        return
    _reset_hf_results_page()
    query = model.hf_query or model.hf_hint or model.name.split(":", 1)[0]
    results = STATE.hf.search(
        query,
        "model",
        limit=12,
        sort="downloads",
        compatible_only=False,
    )
    STATE.hf.hydrate_search_results()
    results = STATE.hf.sift_results(
        sort="downloads",
        compatible_only=True,
        vram_limit_gb=STATE.hf.vram_limit_gb,
    )
    if model.hf_hint and not _looks_gguf(model.hf_hint):
        STATE.hf.inspect_repo(model.hf_hint, "model")
    _update_hf_widgets()
    _set_widget_value("hf-query", query)
    _set_widget_value("hf-search-repo-type", "model")
    if results:
        _set_session_value("hf-repo-id", STATE.hf.last_repo_id)
        _set_session_value("hf-repo-type", STATE.hf.last_repo_type)
    lcars.notify(f"HF model search loaded for Ollama source: {query}.")
    _append_hf_logs()


def _ollama_use_source_action(model_name: str) -> None:
    if _workflow_blocks_config_change():
        return
    model = STATE.ollama.select(model_name.strip())
    if model is None:
        lcars.notify(
            "Ollama model was not found. Refresh and enter the exact name:tag.", level="error"
        )
        return
    if not model.compatible:
        lcars.notify(
            f"Blocked: {model.name} is not Axolotl-readable. {model.reason}", level="error"
        )
        return
    STATE.config_store.apply_model(model.compatible_path)
    lcars.notify(f"Applied Ollama source path to base_model: {model.compatible_path}")
    _update_config_widgets()
    _run_preflight_action()


def live_tick() -> None:
    snapshot = STATE.telemetry.sample()
    primary_disk = _primary_disk(snapshot.disks)
    STATE.resource_tick += 1
    STATE.workflow.tick(STATE.runner)
    lcars.update("cpu-gauge", value=snapshot.cpu_percent)
    lcars.update("ram-gauge", value=snapshot.ram_percent)
    lcars.update(
        "ram-used-metric",
        value=f"{format_bytes(snapshot.ram_used)} / {format_bytes(snapshot.ram_total)}",
        status=_percent_status(snapshot.ram_percent),
    )
    lcars.update("gpu-table", **_table_payload(gpu_rows(snapshot.gpus)))
    lcars.update("process-table", **_table_payload(process_rows()))
    lcars.update("gpu-process-table", **_table_payload(gpu_process_rows()))
    lcars.update(
        "disk-usage-gauge",
        value=primary_disk.percent if primary_disk is not None else 0.0,
    )
    lcars.update(
        "disk-free-metric",
        value=format_bytes(primary_disk.free) if primary_disk is not None else "unavailable",
        status=(
            _percent_status(primary_disk.percent, warn=85, crit=95)
            if primary_disk is not None
            else "warn"
        ),
        options=lcars.MetricOptions(
            secondary_value=(
                format_bytes(primary_disk.total)
                if primary_disk is not None
                else "no mounted volume"
            ),
        ).model_dump(mode="json"),
    )
    lcars.update(
        "disk-table",
        **_table_payload(
            disk_rows(snapshot.disks),
            copy_columns={"Device", "Mount"},
        ),
    )
    if STATE.resource_tick % 5 == 0:
        lcars.update(
            "storage-hotspot-table",
            **_table_payload(
                _storage_rows(_load_config_or_empty()),
                copy_columns={"Path"},
            ),
        )
    lcars.update("resource-chart", series=_series_payload(STATE.telemetry.chart_payload()))
    lcars.update(
        "run-status", value=STATE.runner.status_label(), status=STATE.runner.status_severity()
    )
    command = " ".join(STATE.runner.state.command) if STATE.runner.state.command else "idle"
    lcars.update("run-command-text", content=command[:500])
    _update_lora_live_widgets(snapshot)
    _update_workflow_widgets()
    _append_runner_logs()
    _append_hf_logs()
    _append_ollama_logs()
    _update_cache_widgets(live=True)


def _append_runner_logs() -> None:
    lines = STATE.runner.drain_logs()
    if lines:
        lcars.append_log(LOG_AXOLOTL, *lines)


def _append_hf_logs() -> None:
    lines = STATE.hf.drain_logs()
    if lines:
        lcars.append_log(LOG_HF, *lines)


def _append_ollama_logs() -> None:
    lines = STATE.ollama.drain_logs()
    if lines:
        lcars.append_log(LOG_OLLAMA, *lines)


def _update_preflight_widgets(issues: list[PreflightIssue]) -> None:
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warn")
    lcars.update("preflight-table", **_table_payload(issue_rows(issues)))
    lcars.update(
        "run-gate-metric", value="BLOCKED" if errors else "READY", status="crit" if errors else "ok"
    )
    lcars.update("warning-count-metric", value=str(warnings), status="warn" if warnings else "ok")


def _update_config_widgets() -> None:
    if STATE.workflow.sync_active_config(STATE.config_store.active_name):
        _persist_workflow_document()
        lcars.update(
            WORKFLOW_CANVAS_ID,
            document=STATE.workflow.document.model_dump(mode="json"),
        )
    summary = STATE.config_store.summary_rows()
    payload = _table_payload(summary, copy_columns={"Key", "Value"})
    lcars.update("config-summary-table", **payload)
    lcars.update("config-page-summary-table", **payload)
    lcars.update("config-coverage-table", **_table_payload(_coverage_rows()))
    configs = STATE.config_store.list_configs()
    lcars.update(
        "active-config-select",
        value=STATE.config_store.active_name,
        options=[
            lcars.SelectOption(label=name, value=name).model_dump(mode="json") for name in configs
        ],
    )
    active_cfg = _load_config_or_empty()
    active_base = str(active_cfg.get("base_model") or LORA_BASE_MODEL_VALUES[0])
    base_options = [
        lcars.SelectOption(
            label=label,
            value=value,
            description=LORA_BASE_MODEL_HINTS.get(value),
        )
        for label, value in LORA_BASE_MODELS
    ]
    if active_base not in LORA_BASE_MODEL_VALUES:
        base_options.insert(
            0,
            lcars.SelectOption(
                label=f"{active_base} · current custom model",
                value=active_base,
                description="Preserved from the active YAML.",
            ),
        )
    project_name = Path(STATE.config_store.active_name).stem
    active_preset = infer_lora_preset(active_cfg)
    recommended_preset = recommend_lora_preset(
        _detected_gpu_vram_gb(),
        active_base,
    )
    store = get_session_state(get_ctx().session_id)
    store["lora-project-name"] = project_name
    store["lora-base-model"] = active_base
    store["lora-preset"] = active_preset
    lcars.update("lora-project-name", value=project_name)
    lcars.update(
        "lora-base-model",
        value=active_base,
        options=[option.model_dump(mode="json") for option in base_options],
    )
    lcars.update(
        "lora-preset",
        value=active_preset,
        options=[
            option.model_dump(mode="json")
            for option in _lora_preset_options(recommended_preset)
        ],
    )
    try:
        values = STATE.config_store.editor_values()
    except Exception:
        return
    store["active-config-select"] = STATE.config_store.active_name
    for spec in FIELD_SPECS:
        widget_id = spec.widget_id
        value = values[widget_id]
        store[widget_id] = value
        if spec.kind == "bool":
            lcars.update(widget_id, checked=value)
        elif spec.kind == "select":
            lcars.update(
                widget_id,
                value=value,
                options=[
                    option.model_dump(mode="json")
                    for option in _config_select_options(spec, str(value))
                ],
            )
        else:
            lcars.update(widget_id, value=value)
    lcars.update(
        "setup-defaults-table",
        **_table_payload(_setup_default_rows(), copy_columns={"Field"}),
    )


def _update_lora_widgets() -> None:
    cfg = _load_config_or_empty()
    dataset = inspect_configured_dataset(PROJECT_ROOT, cfg)
    dataset_cache_rows = _lora_dataset_cache_rows()
    cached_datasets = _lora_downloaded_dataset_cache_rows(dataset_cache_rows)
    cached_dataset_ids = tuple(row["Repo"] for row in cached_datasets)
    artifacts = discover_adapter_artifacts(PROJECT_ROOT, cfg)
    errors = [issue for issue in STATE.preflight_issues if issue.severity == "error"]
    steps = _lora_journey_rows(cfg, dataset, bool(artifacts))
    completed = sum(1 for row in steps if row["Status"] == "READY")
    can_train = (
        not errors
        and _lora_dataset_trainable(cfg, dataset)
        and not STATE.workflow.is_active
        and not STATE.runner.is_running()
    )

    lcars.update("lora-home-progress", value=completed / len(steps) * 100)
    lcars.update("lora-journey-table", **_table_payload(steps))
    plan_payload = _table_payload(
        _lora_training_plan_rows(cfg),
        copy_columns={"Value"},
    )
    lcars.update("lora-setup-plan-table", **plan_payload)
    recommended_preset = recommend_lora_preset(
        _detected_gpu_vram_gb(),
        str(cfg.get("base_model") or ""),
    )
    lcars.update(
        "lora-preset-recommendation",
        value=get_lora_preset(recommended_preset).label,
        options=lcars.MetricOptions(
            secondary_value=_detected_gpu_label(),
        ).model_dump(mode="json"),
    )
    lcars.update(
        "lora-preset-table",
        **_table_payload(_lora_preset_rows(recommended_preset)),
    )
    lcars.update(
        "lora-tuning-table",
        **_table_payload(_lora_tuning_rows(cfg)),
    )
    lcars.update(
        "lora-train-plan-table",
        **_table_payload(_lora_training_brief_rows(cfg, dataset)),
    )
    lcars.update(
        "lora-data-status",
        value=_lora_dataset_status(cfg, dataset),
        status=(
            "ok"
            if _lora_dataset_trainable(cfg, dataset)
            else ("crit" if dataset.errors else "warn")
        ),
        options=lcars.MetricOptions(
            secondary_value=dataset.source or "No source"
        ).model_dump(mode="json"),
    )
    lcars.update(
        "lora-data-active-source",
        content=_lora_active_dataset_summary(cfg, dataset, cached_dataset_ids),
    )
    lcars.update(
        "lora-data-checks-table",
        **_table_payload(_lora_dataset_issue_rows(dataset)),
    )
    lcars.update(
        "lora-train-gate",
        value="READY" if can_train else "BLOCKED",
        status="ok" if can_train else "crit",
        options=lcars.MetricOptions(
            secondary_value=_lora_gate_detail(errors, dataset, cfg)
        ).model_dump(mode="json"),
    )
    lcars.update(
        "lora-train-data-status",
        value=_lora_dataset_status(cfg, dataset),
        status="ok" if _lora_dataset_trainable(cfg, dataset) else "warn",
        options=lcars.MetricOptions(
            secondary_value=(
                "remote"
                if dataset.example_count is None
                else f"{dataset.example_count} examples"
            )
        ).model_dump(mode="json"),
    )
    lcars.update(
        "lora-artifacts-table",
        **_table_payload(
            _lora_artifact_rows(artifacts),
            copy_columns={"Path", "Base Model"},
        ),
    )
    lcars.update(
        "lora-train-preprocess",
        disabled=not _lora_dataset_trainable(cfg, dataset) or STATE.runner.is_running(),
    )
    lcars.update("lora-train-start", disabled=not can_train)
    lcars.update("lora-train-stop", disabled=not STATE.runner.is_running())

    trained_base = str(cfg.get("base_model") or "")
    adapter_default = str(artifacts[0].path) if artifacts else str(cfg.get("output_dir") or "")
    lcars.update(
        "lora-test-adapter-status",
        value="FOUND" if artifacts else "NOT FOUND",
        status="ok" if artifacts else "warn",
        options=lcars.MetricOptions(
            secondary_value=adapter_default or "Finish training first"
        ).model_dump(mode="json"),
    )
    lcars.update(
        "lora-test-compatibility-table",
        **_table_payload(
            [
                {
                    "Requirement": "Exact base",
                    "Your training value": trained_base,
                    "Why": "A different Ollama base can produce erratic behavior",
                },
                {
                    "Requirement": "Adapter format",
                    "Your training value": (
                        "Safetensors detected" if artifacts else "Not detected"
                    ),
                    "Why": "Ollama imports the PEFT adapter directory",
                },
                {
                    "Requirement": "Architecture",
                    "Your training value": _lora_architecture_hint(trained_base),
                    "Why": "Ollama documents adapter import for Llama, Mistral, and Gemma",
                },
            ],
            copy_columns={"Your training value"},
        ),
    )
    lcars.update(
        "lora-test-ollama-status",
        value="READY" if STATE.ollama.models else "NOT CONNECTED",
        status="ok" if STATE.ollama.models else "crit",
        options=lcars.MetricOptions(
            secondary_value=(
                f"{len(STATE.ollama.models)} local model(s)"
                if STATE.ollama.models
                else STATE.ollama.last_error or "No local models detected"
            )
        ).model_dump(mode="json"),
    )


def _update_lora_ollama_selects(
    *,
    preferred_base: str = "",
    preferred_tuned: str = "",
) -> None:
    names = [model.name for model in STATE.ollama.models]
    serialized = [
        option.model_dump(mode="json") for option in _lora_ollama_select_options(names)
    ]
    preferences = {
        "lora-test-base-model": preferred_base,
        "lora-test-compare-base": preferred_base,
        "lora-test-chat-model": preferred_tuned,
    }
    for widget_id, preferred in preferences.items():
        current = _widget_value(widget_id)
        selected = current if current in names else ""
        if preferred in names:
            selected = preferred
        _set_session_value(widget_id, selected)
        lcars.update(widget_id, value=selected, options=serialized, disabled=not names)
    lcars.update(
        "ollama-table",
        **_table_payload(
            STATE.ollama.rows(),
            copy_columns={"Model", "Source"},
        ),
    )


def _update_lora_live_widgets(snapshot: Any) -> None:
    gpu = snapshot.gpus[0] if snapshot.gpus else None
    lcars.update(
        "lora-train-status",
        value=STATE.runner.status_label(),
        status=STATE.runner.status_severity(),
    )
    lcars.update("lora-train-elapsed", value=_lora_elapsed_text())
    lcars.update(
        "lora-train-gpu",
        value=f"{gpu.utilization:.0f}%" if gpu is not None else "NOT DETECTED",
        status=_percent_status(gpu.utilization) if gpu is not None else "warn",
        options=lcars.MetricOptions(
            secondary_value=(
                f"{format_bytes(gpu.memory_used)} / {format_bytes(gpu.memory_total)}"
                if gpu is not None
                else "training may be unavailable or CPU-only"
            )
        ).model_dump(mode="json"),
    )
    lcars.update(
        "lora-train-ram",
        value=f"{snapshot.ram_percent:.0f}%",
        status=_percent_status(snapshot.ram_percent),
        options=lcars.MetricOptions(
            secondary_value=(
                f"{format_bytes(snapshot.ram_used)} / {format_bytes(snapshot.ram_total)}"
            )
        ).model_dump(mode="json"),
    )
    lcars.update(
        "lora-train-progress",
        value=_lora_process_progress(),
        options=lcars.MeterOptions(
            unit="%",
            indeterminate=STATE.runner.is_running(),
            description="Axolotl's detailed step and loss output appears in the log below.",
        ).model_dump(mode="json"),
    )
    lcars.update("lora-train-stop", disabled=not STATE.runner.is_running())
    if STATE.resource_tick % 5 == 0:
        artifacts = discover_adapter_artifacts(PROJECT_ROOT, _load_config_or_empty())
        lcars.update(
            "lora-artifacts-table",
            **_table_payload(
                _lora_artifact_rows(artifacts),
                copy_columns={"Path", "Base Model"},
            ),
        )
        lcars.update(
            "lora-test-adapter-status",
            value="FOUND" if artifacts else "NOT FOUND",
            status="ok" if artifacts else "warn",
        )


def _update_hf_widgets() -> None:
    result_options_model = _hf_result_table_options()
    repo_id = STATE.hf.last_repo_id.strip()
    result = _hf_result_for(repo_id)
    blocked = bool(result is not None and result.blocked)
    selection_value, selection_detail, selection_status = _hf_selection_metric(
        repo_id,
        STATE.hf.last_repo_type,
        result,
    )
    lcars.update(
        HF_RESULTS_TABLE_ID,
        headers=[column.label or column.key for column in result_options_model.columns or []],
        rows=[row.model_dump(mode="json") for row in _hf_result_rows()],
        options=result_options_model.model_dump(mode="json"),
    )
    lcars.update(
        "hf-jobs-table",
        **_table_payload(
            _hf_job_rows(),
            copy_columns={"Repo", "Revision", "Local Path"},
        ),
    )
    lcars.update(
        "hf-related",
        disabled=not bool(repo_id) or STATE.hf.last_repo_type != "model",
    )
    lcars.update("hf-download", disabled=not bool(repo_id) or blocked)
    lcars.update("hf-use-repo", disabled=not bool(repo_id) or blocked)
    lcars.update("hf-use-local", disabled=not bool(STATE.hf.last_local_path))
    lcars.update(
        "hf-selection-status",
        value=selection_value,
        status=selection_status,
        options=lcars.MetricOptions(
            secondary_value=selection_detail,
        ).model_dump(mode="json"),
    )
    lcars.update(
        "hf-filter-summary",
        content=_current_hf_filter_summary(),
    )
    lcars.update(
        "hf-selected-repo-copy",
        content=STATE.hf.last_repo_id.strip() or "No repository selected.",
        options=_hf_selected_text_options(
            STATE.hf.last_repo_id,
            STATE.hf.last_repo_type,
        ).model_dump(mode="json"),
    )


def _update_lora_downloaded_dataset_widgets(
    *,
    cache_rows: list[dict[str, str]] | None = None,
) -> None:
    rows = cache_rows if cache_rows is not None else _lora_dataset_cache_rows()
    ready_rows = _lora_downloaded_dataset_cache_rows(rows)
    configured_dataset = str(
        _config_path_value(_load_config_or_empty(), "datasets.0.path") or ""
    ).strip()
    options = [
        option.model_dump(mode="json")
        for option in _lora_downloaded_dataset_options(ready_rows)
    ]
    lcars.update(
        "lora-downloaded-dataset-table",
        **_table_payload(
            _lora_dataset_download_rows(rows, configured_dataset),
            copy_columns={"Dataset", "Cache path"},
        ),
    )
    lcars.update(
        "lora-hf-dataset",
        options=options,
        disabled=not bool(ready_rows),
    )
    lcars.update("lora-downloaded-dataset-form", disabled=not bool(ready_rows))
    lcars.update(
        "lora-no-downloaded-datasets",
        content=_lora_dataset_cache_notice(rows, ready_rows),
        visible=(
            not bool(ready_rows)
            or any(row.get("Status") == "INCOMPLETE" for row in rows)
        ),
    )


def _update_cache_widgets(*, live: bool = False) -> None:
    try:
        rows, total_text, total_bytes = STATE.hf.cache_rows()
    except Exception:
        return
    lcars.update("hf-cache-total", value=cache_summary_text(total_bytes, total_text))
    lcars.update(
        "hf-cache-table",
        **_table_payload(
            rows
            or [
                {
                    "Type": "",
                    "Repo": "No cached Hugging Face repos",
                    "Size": "",
                    "Files": "",
                    "Revision": "",
                    "Path": "",
                }
            ],
            copy_columns={"Repo", "Revision", "Path"},
        ),
    )
    lcars.update(
        "hf-jobs-table",
        **_table_payload(
            _hf_job_rows(),
            copy_columns={"Repo", "Revision", "Local Path"},
        ),
    )
    lcars.update("hf-use-local", disabled=not bool(STATE.hf.last_local_path))
    _update_lora_downloaded_dataset_widgets()
    if not live:
        _append_hf_logs()


def _hf_result_for(repo_id: str) -> Any:
    repo_id = repo_id.strip()
    if STATE.hf.selected_details and STATE.hf.selected_details.result.repo_id == repo_id:
        return STATE.hf.selected_details.result
    for result in [*STATE.hf.search_results, *STATE.hf.related_results]:
        if result.repo_id == repo_id:
            return result
    return None


def create_lcars_app(
    ui_fn: Callable[[], None], *, live_fn: Callable[[], None] | None = None
) -> FastAPI:
    pre_run_config = get_ctx().config
    build_ctx = _LCARSContext(
        mode=Mode.BUILD,
        session_id="build",
        builder=_ManifestBuilder(),
        config=pre_run_config,
    )
    set_ctx(build_ctx)
    ui_fn()
    assert build_ctx.builder is not None
    manifest = build_ctx.builder.build(build_ctx.config)
    form_children_by_action = _index_form_children(manifest)

    app = create_app(manifest=manifest)
    _install_internal_navigation(app)
    _install_manifest_refresh(app, ui_fn, build_ctx.config)
    event_bus = app.state.event_bus

    async def _dsl_action_handler(
        action_id: str, value: Any, session_id: str = "http_fallback"
    ) -> None:
        handle_ctx = _LCARSContext(
            mode=Mode.HANDLE,
            session_id=session_id,
            active_action_id=action_id,
            active_action_value=value,
            config=build_ctx.config,
            builder=_ManifestBuilder(),
        )
        set_ctx(handle_ctx)
        if isinstance(value, dict):
            session_state = get_session_state(session_id)
            child_ids = form_children_by_action.get(action_id)
            if child_ids is None:
                for key, item_value in value.items():
                    if isinstance(key, str):
                        session_state[key] = item_value
            else:
                for child_id in child_ids:
                    if child_id in value:
                        session_state[child_id] = value[child_id]
        ui_fn()
        _persist_widget_state(session_id)
        _mark_manifest_stale(app)
        for envelope in handle_ctx.pending_events:
            await event_bus.publish(envelope)

    app.state.plugin_action_handlers["*"] = _dsl_action_handler

    if live_fn is not None:

        async def _live_loop() -> None:
            while True:
                await asyncio.sleep(2.0)
                live_ctx = _LCARSContext(
                    mode=Mode.LIVE,
                    session_id="live",
                    config=build_ctx.config,
                    builder=_ManifestBuilder(),
                )
                set_ctx(live_ctx)
                try:
                    live_fn()
                except Exception:
                    continue
                for envelope in live_ctx.pending_events:
                    await event_bus.publish(envelope)

        app.state._live_coro_factory = _live_loop

    _install_raw_editor(app)
    return app


def _build_manifest(ui_fn: Callable[[], None], config: Any) -> Any:
    build_ctx = _LCARSContext(
        mode=Mode.BUILD,
        session_id="build",
        builder=_ManifestBuilder(),
        config=config,
    )
    set_ctx(build_ctx)
    ui_fn()
    assert build_ctx.builder is not None
    return build_ctx.builder.build(build_ctx.config)


class _LiveManifestState(State):
    """App state that rebuilds the manifest on demand once it goes stale.

    The manifest is what a reconnecting browser receives, so it has to reflect the
    current config and control selections rather than whatever was true at startup.
    Rebuilding costs a few hundred milliseconds, so instead of paying that on every
    action we mark it stale and rebuild only when someone actually reads it.
    """

    def __getattr__(self, key: str) -> Any:
        if key == "manifest" and self._state.get("_manifest_stale"):
            self._state["_manifest_stale"] = False
            rebuild = self._state.get("_manifest_rebuild")
            if rebuild is not None:
                try:
                    self._state["manifest"] = rebuild()
                except Exception:
                    # Serving a slightly stale manifest beats failing the connection.
                    pass
        return super().__getattr__(key)


def _install_manifest_refresh(app: FastAPI, ui_fn: Callable[[], None], config: Any) -> None:
    """Make `app.state.manifest` self-refreshing when marked stale."""

    app.state = _LiveManifestState(app.state._state)
    app.state._manifest_rebuild = lambda: _build_manifest(ui_fn, config)
    app.state._manifest_stale = False


def _install_internal_navigation(app: FastAPI) -> None:
    """Turn same-app ``?page=...`` links into client-side LCARS tab changes."""

    root_index = next(
        (
            index
            for index, route in enumerate(app.router.routes)
            if getattr(route, "path", "") == "/"
            and "GET" in (getattr(route, "methods", set()) or set())
        ),
        None,
    )
    if root_index is None:
        return

    original_route = app.router.routes.pop(root_index)
    original_root = original_route.endpoint

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root_with_internal_navigation() -> Any:
        page = original_root()
        if not isinstance(page, str) or "</body>" not in page:
            return page
        script = (
            '<script type="module" src="/lcars/internal-navigation.js"></script>'
        )
        return page.replace("</body>", f"{script}\n</body>", 1)

    enhanced_root = app.router.routes.pop()
    app.router.routes.insert(root_index, enhanced_root)

    @app.get("/lcars/internal-navigation.js", include_in_schema=False)
    def internal_navigation_script() -> FileResponse:
        return FileResponse(
            INTERNAL_NAVIGATION_SCRIPT,
            media_type="text/javascript",
        )

    _move_last_route_before_spa(app)


def _mark_manifest_stale(app: FastAPI) -> None:
    """Flag the manifest for rebuild before the next client reads it."""

    app.state._manifest_stale = True


def _install_raw_editor(app: FastAPI) -> None:
    @app.get("/raw", response_class=HTMLResponse, include_in_schema=False)
    def raw_get(request: Request) -> str:
        _ = request
        return _raw_html()

    _move_last_route_before_spa(app)

    @app.post("/raw", response_class=HTMLResponse, include_in_schema=False)
    def raw_post(content: str = Form(...)) -> str:
        status = ""
        if STATE.workflow.is_active:
            return _raw_html(
                status="Not saved: the active config is locked while its workflow is running."
            )
        try:
            STATE.config_store.save_raw_text(content)
            STATE.refresh_preflight()
            _mark_manifest_stale(app)
            status = "Saved. Return to LCARS and run preflight."
        except Exception as exc:
            status = f"Not saved: {exc}"
        return _raw_html(status=status)

    _move_last_route_before_spa(app)

    @app.get("/raw/return", include_in_schema=False)
    def raw_return() -> RedirectResponse:
        return RedirectResponse("/")

    _move_last_route_before_spa(app)


def _move_last_route_before_spa(app: FastAPI) -> None:
    if not app.router.routes:
        return
    route = app.router.routes.pop()
    insert_at = len(app.router.routes)
    for index, existing in enumerate(app.router.routes):
        if getattr(existing, "path", "") == "/{full_path:path}":
            insert_at = index
            break
    app.router.routes.insert(insert_at, route)


def _raw_html(status: str = "") -> str:
    try:
        content = STATE.config_store.active_path.read_text(encoding="utf-8")
    except OSError as exc:
        content = f"# Could not read config: {exc}\n"
    status_html = f"<p class='status'>{html.escape(status)}</p>" if status else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Axolotl Raw Config Editor</title>
  <style>
    :root {{ color-scheme: dark; --bg:#05070d; --panel:#111827; --line:#ff9b28; --blue:#78c7ff; --text:#f7f1da; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    header {{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:16px 20px; border-bottom:4px solid var(--line); }}
    h1 {{ margin:0; color:var(--line); font-size:24px; letter-spacing:0; }}
    a {{ color:var(--blue); text-decoration:none; }}
    main {{ padding:18px; }}
    textarea {{ box-sizing:border-box; width:100%; min-height:72vh; resize:vertical; background:#03050a; color:var(--text); border:2px solid var(--line); border-radius:6px; padding:14px; font:14px/1.42 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    button {{ margin-top:14px; border:0; border-radius:4px; background:var(--line); color:#120800; padding:11px 18px; font-weight:800; cursor:pointer; }}
    .status {{ color:var(--blue); }}
    .path {{ color:#d8bcff; overflow-wrap:anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>AXOLOTL RAW YAML</h1>
    <a href="/">Return to LCARS</a>
  </header>
  <main>
    <p class="path">{html.escape(str(STATE.config_store.active_path))}</p>
    {status_html}
    <form method="post" action="/raw">
      <textarea name="content" spellcheck="false">{html.escape(content)}</textarea>
      <br>
      <button type="submit">SAVE YAML</button>
    </form>
  </main>
</body>
</html>"""


def _collect_editor_values() -> dict[str, Any]:
    ctx = get_ctx()
    store = get_session_state(ctx.session_id)
    values = STATE.config_store.editor_values()
    for spec in FIELD_SPECS:
        if spec.widget_id in store:
            values[spec.widget_id] = store[spec.widget_id]
    return values


def _seed_text(widget_id: str, value: str, *, force: bool = False) -> None:
    ctx = get_ctx()
    store = get_session_state(ctx.session_id)
    if force and ctx.mode == Mode.BUILD:
        store[widget_id] = value
    else:
        store.setdefault(widget_id, value)


def _hydrate_widget_state() -> None:
    """Replay saved control values into the session store before widgets read it.

    A browser reload lands on a brand new session whose store is empty, so without
    this every control would fall back to its hardcoded build-time default. In BUILD
    mode saved values are authoritative (the manifest must show what was last chosen);
    elsewhere they only fill gaps so live edits in the session keep winning.
    """

    ctx = get_ctx()
    store = get_session_state(ctx.session_id)
    saved = UI_STATE.widget_values()
    defaults = _persisted_widget_defaults()
    choices = _persisted_widget_choices()
    if "hf-search-repo-type" not in saved:
        # Before the v4.2 Hub redesign, hf-repo-type was the search selector
        # even though app.hf_repo_type tracked the selected repository. Preserve
        # both meanings while migrating the next persisted snapshot.
        saved["hf-search-repo-type"] = saved.get(
            "hf-repo-type",
            defaults["hf-search-repo-type"],
        )
        saved["hf-repo-type"] = UI_STATE.get(
            "hf_repo_type",
            defaults["hf-repo-type"],
        )
    for widget_id in PERSISTED_WIDGET_IDS:
        value = _normalized_persisted_widget_value(
            widget_id,
            saved.get(widget_id, defaults[widget_id]),
            defaults=defaults,
            choices=choices,
        )
        if ctx.mode == Mode.BUILD:
            store[widget_id] = value
        else:
            store.setdefault(widget_id, value)


def _persist_widget_state(session_id: str) -> None:
    """Snapshot a session's control values and cross-page selections to disk."""

    store = get_session_state(session_id)
    defaults = _persisted_widget_defaults()
    choices = _persisted_widget_choices()
    normalized = {
        widget_id: _normalized_persisted_widget_value(
            widget_id,
            store.get(widget_id, defaults[widget_id]),
            defaults=defaults,
            choices=choices,
        )
        for widget_id in PERSISTED_WIDGET_IDS
    }
    store.update(normalized)
    changed = UI_STATE.remember_widgets(normalized)
    changed = (
        UI_STATE.set_many(
            {
                "active_config": STATE.config_store.active_name,
                "hf_repo_id": normalized["hf-repo-id"],
                "hf_repo_type": normalized["hf-repo-type"],
                "hf_local_sort": STATE.hf.local_sort,
                "hf_local_sort_desc": STATE.hf.local_sort_desc,
                "hf_vram_limit": STATE.hf.vram_limit_gb,
                "hf_expanded_result_ids": list(STATE.hf.expanded_result_ids),
            }
        )
        or changed
    )
    if changed:
        UI_STATE.save()


def _widget_value(widget_id: str, default: str = "") -> str:
    """Current value of a widget in this session, for controls rendered later."""

    store = get_session_state(get_ctx().session_id)
    value = store.get(widget_id, default)
    return default if value is None else str(value)


def _load_config_or_empty() -> dict[str, Any]:
    try:
        return STATE.config_store.load()
    except ConfigError:
        return {}


def _config_path_value(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def _table_payload(
    rows: list[dict[str, Any]],
    *,
    copy_columns: set[str] | None = None,
    numeric_columns: set[str] | None = None,
) -> dict[str, Any]:
    if not rows:
        return {"headers": [], "rows": []}
    copy_columns = copy_columns or set()
    numeric_columns = numeric_columns or set()
    headers = list(rows[0].keys())
    return {
        "headers": headers,
        "rows": [
            {
                "id": f"row-{index}",
                "cells": [
                    _serialized_table_value(
                        row.get(header, ""),
                        numeric=header in numeric_columns,
                        copyable=header in copy_columns,
                    )
                    for header in headers
                ],
            }
            for index, row in enumerate(rows)
        ],
    }


def _enhanced_table(
    rows: list[dict[str, Any]],
    *,
    title: str,
    id: str,
    filter_columns: set[str] | None = None,
    numeric_columns: set[str] | None = None,
    date_columns: set[str] | None = None,
    copy_columns: set[str] | None = None,
    page_size: int | None = None,
) -> None:
    """Render a compact native table with data controls and copy affordances."""

    filter_columns = filter_columns or set()
    numeric_columns = numeric_columns or set()
    date_columns = date_columns or set()
    copy_columns = copy_columns or set()
    headers = list(rows[0].keys()) if rows else []
    columns = []
    for header in headers:
        value_type = (
            "number"
            if header in numeric_columns
            else ("date" if header in date_columns else "text")
        )
        columns.append(
            lcars.TableColumn(
                key=header,
                label=header,
                value_type=value_type,
                sortable=True,
                first_sort_direction="desc" if value_type in {"number", "date"} else "asc",
                filter=(
                    "number"
                    if header in numeric_columns and header in filter_columns
                    else ("text" if header in filter_columns else "none")
                ),
                align="end" if header in numeric_columns else "start",
            )
        )

    typed_rows = [
        {
            key: _typed_table_value(
                value,
                numeric=key in numeric_columns,
                copyable=key in copy_columns,
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    lcars.table(
        typed_rows,
        title=title,
        id=id,
        options=lcars.TableOptions(
            columns=columns or None,
            pagination=(
                lcars.TablePagination(page_size=page_size) if page_size is not None else None
            ),
            sticky_header=True,
            density="compact",
        ),
    )


def _typed_table_value(value: Any, *, numeric: bool, copyable: bool = False) -> Any:
    typed = value
    if numeric and value not in ("", None) and not isinstance(value, (int, float)):
        display = str(value)
        candidate = display.strip().replace(",", "").removesuffix("%")
        try:
            parsed = float(candidate)
        except ValueError:
            pass
        else:
            raw: int | float = int(parsed) if parsed.is_integer() else parsed
            typed = lcars.TableCell(value=raw, display=display)
    if not copyable or value in ("", None):
        return typed
    if isinstance(typed, lcars.TableCell):
        return typed.model_copy(
            update={"copyable": True, "copy_value": str(value)},
        )
    return lcars.TableCell(
        value=typed,
        copyable=True,
        copy_value=str(value),
    )


def _serialized_table_value(value: Any, *, numeric: bool, copyable: bool) -> Any:
    typed = _typed_table_value(value, numeric=numeric, copyable=copyable)
    if isinstance(typed, lcars.TableCell):
        return typed.model_dump(mode="json")
    return typed


def _set_widget_value(widget_id: str, value: str) -> None:
    _set_session_value(widget_id, value)
    lcars.update(widget_id, value=value)


def _set_session_value(widget_id: str, value: str) -> None:
    """Store non-rendered workflow state without emitting a dead widget update."""

    store = get_session_state(get_ctx().session_id)
    store[widget_id] = value


def _bounded_int(value: str, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _step_precision(step: float) -> int:
    rendered = f"{step:.12f}".rstrip("0").rstrip(".")
    return len(rendered.split(".", 1)[1]) if "." in rendered else 0


def _optional_float(value: float | int | str) -> float | None:
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _is_active_action(widget_id: str) -> bool:
    ctx = get_ctx()
    return ctx.mode == Mode.HANDLE and ctx.active_action_id == widget_id


def _active_action_value() -> dict[str, Any]:
    value = get_ctx().active_action_value
    return value if isinstance(value, dict) else {}


def _series_payload(data: dict[str, list[float]]) -> list[dict[str, Any]]:
    return [{"name": name, "data": values} for name, values in data.items()]


def _percent_status(value: float, *, warn: float = 80, crit: float = 92) -> str:
    if value >= crit:
        return "crit"
    if value >= warn:
        return "warn"
    return "ok"


def _looks_gguf(value: str) -> bool:
    lowered = value.lower()
    return lowered.endswith(".gguf") or "-gguf" in lowered or "/gguf" in lowered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Axolotl LCARS UI.")
    parser.add_argument("--host", "--ip", dest="host", default="127.0.0.1")
    parser.add_argument("--port", dest="port", default=8000, type=int)
    parser.add_argument("--open", dest="open_browser", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_lcars_app(build_ui, live_fn=live_tick)
    if args.open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}/")).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

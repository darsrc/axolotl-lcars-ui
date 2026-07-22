"""LCARS application entry point for Axolotl management."""

from __future__ import annotations

import argparse
import asyncio
import html
import threading
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

import lcars_ui as lcars
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from axolotl_lcars_ui.config_store import FIELD_SPECS, ConfigError, ConfigStore, FieldSpec
from axolotl_lcars_ui.hf_manager import (
    HuggingFaceManager,
    cache_summary_text,
    detail_file_rows,
    detail_summary_rows,
    related_rows,
    result_link_markdown,
    result_options,
    search_rows,
)
from axolotl_lcars_ui.ollama import OllamaManager
from axolotl_lcars_ui.resources import (
    TelemetrySampler,
    disk_rows,
    format_bytes,
    gpu_process_rows,
    gpu_rows,
    process_rows,
    storage_hotspot_rows,
)
from axolotl_lcars_ui.runner import AXOLOTL_ACTIONS, CONFIG_ACTIONS, LAUNCHER_ACTIONS, AxolotlRunner
from axolotl_lcars_ui.validator import AxolotlPreflight, PreflightIssue, issue_rows

from lcars_ui.app import create_app
from lcars_ui.dsl._builder import _ManifestBuilder
from lcars_ui.dsl._state import Mode, _LCARSContext, get_ctx, get_session_state, set_ctx
from lcars_ui.dsl.api import _index_form_children


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_AXOLOTL = "axolotl-run-log"
LOG_HF = "hf-log"
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
HF_LOCAL_SORT_OPTIONS = ["downloads", "likes", "fit", "size", "updated", "repo"]
HF_COMPATIBILITY_OPTIONS = ["compatible files only", "include warnings and blocked"]
HF_ARTIFACT_FILTER_OPTIONS = ["any artifact", "base/trainable models", "PEFT adapters", "datasets", "runtime only"]
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
    "NousResearch/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b",
    "mistralai/Mistral-7B-v0.1",
]

DATASET_PRESETS = {
    "teknium/GPT4-LLM-Cleaned | alpaca": ("teknium/GPT4-LLM-Cleaned", "alpaca"),
    "tatsu-lab/alpaca | alpaca": ("tatsu-lab/alpaca", "alpaca"),
    "HuggingFaceH4/ultrachat_200k | chat_template": ("HuggingFaceH4/ultrachat_200k", "chat_template"),
    "./data/train.jsonl | completion": ("./data/train.jsonl", "completion"),
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
    preflight: AxolotlPreflight
    preflight_issues: list[PreflightIssue] = field(default_factory=list)
    resource_tick: int = 0

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


STATE = AppState(
    config_store=ConfigStore(PROJECT_ROOT),
    telemetry=TelemetrySampler(),
    hf=HuggingFaceManager(),
    ollama=OllamaManager(),
    runner=AxolotlRunner(PROJECT_ROOT),
    preflight=AxolotlPreflight(PROJECT_ROOT, OllamaManager()),
)
STATE.preflight = AxolotlPreflight(PROJECT_ROOT, STATE.ollama)
STATE.telemetry.sample()
STATE.ollama.refresh()
STATE.refresh_preflight()


def build_ui() -> None:
    """Build the static LCARS manifest and handle rerun actions."""

    lcars.config(
        "AXOLOTL LCARS",
        theme="galaxy",
        subtitle="CONFIGURATION / TELEMETRY / CONTENT OPS",
        header_color="tanoi",
        sound_enabled=True,
    )

    lcars.nav("Command", page="command", color="tanoi")
    lcars.nav("Config", page="config", color="golden-tanoi")
    lcars.nav("Setup", page="config-setup", color="pale-canary")
    lcars.nav("Train", page="config-train", color="tanoi")
    lcars.nav("Hardware", page="config-hardware", color="anakiwa")
    lcars.nav("Tracking", page="config-tracking", color="lilac")
    lcars.nav("Advanced", page="config-advanced", color="blue-bell")
    lcars.nav("Run", page="run", color="red")
    lcars.nav("Resources", page="resources", color="anakiwa")
    lcars.nav("HF Hub", page="hub", color="lilac")
    lcars.nav("Content", page="content", color="blue-bell")
    lcars.nav("Ollama", page="ollama", color="pale-canary")

    _command_page()
    _config_page()
    _config_setup_page()
    _config_train_page()
    _config_hardware_page()
    _config_tracking_page()
    _config_advanced_page()
    _run_page()
    _resources_page()
    _hub_page()
    _content_page()
    _ollama_page()


def _command_page() -> None:
    with lcars.page("Command", id="command", layout="console"):
        with lcars.console("Axolotl Operations", color="tanoi"):
            with lcars.data_panel("Launch Readiness", color="tanoi"):
                issues = STATE.preflight_issues
                errors = sum(1 for issue in issues if issue.severity == "error")
                warnings = sum(1 for issue in issues if issue.severity == "warn")
                lcars.metric("Run Gate", "BLOCKED" if errors else "READY", status="crit" if errors else "ok", color="red" if errors else "tanoi", id="run-gate-metric")
                lcars.metric("Warnings", str(warnings), status="warn" if warnings else "ok", color="golden-tanoi", id="warning-count-metric")
                lcars.metric("Axolotl CLI", "FOUND" if STATE.runner.axolotl_path else "MISSING", status="ok" if STATE.runner.axolotl_path else "crit", color="anakiwa", id="axolotl-cli-metric")
                lcars.table(issue_rows(issues), title="Preflight Matrix", id="preflight-table")

            with lcars.control_panel("Primary Actions", color="golden-tanoi"):
                if lcars.button("Run Preflight", color="anakiwa", id="run-preflight"):
                    _run_preflight_action()
                if lcars.button("Save Structured Config", color="tanoi", id="save-structured-config"):
                    _save_config_action()
                if lcars.button("Start Training", color="red", id="quick-start-training"):
                    _start_axolotl_action("train")
                if lcars.button("Stop Axolotl", color="red", id="quick-stop-axolotl"):
                    _stop_axolotl_action()
                lcars.markdown(
                    f"Active config: `{STATE.config_store.active_path}`\n\n"
                    "[Open raw YAML editor](/raw)",
                    id="command-raw-link",
                )

            with lcars.data_panel("Current Config Summary", color="blue-bell"):
                lcars.table(STATE.config_store.summary_rows(), title="Active YAML", id="config-summary-table")


def _config_page() -> None:
    with lcars.page("Config", id="config", layout="console"):
        with lcars.console("Config Manager", color="golden-tanoi"):
            lcars.markdown(
                "Structured pages cover the high-impact Axolotl surface. The raw YAML editor remains "
                "the complete escape hatch for deeply nested or experimental options.",
                id="config-note",
            )
            with lcars.control_panel("Config Files", color="golden-tanoi"):
                configs = STATE.config_store.list_configs()
                selected = lcars.select(
                    "Active Config",
                    configs,
                    value=STATE.config_store.active_name,
                    id="active-config-select",
                )
                _seed_text("new-config-name", "experiment.yml")
                new_name = lcars.text_input(
                    "New Config Name",
                    placeholder="experiment.yml",
                    autocomplete=False,
                    id="new-config-name",
                )
                if lcars.button("Switch Config", color="anakiwa", id="config-switch"):
                    _switch_config_action(selected)
                if lcars.button("Create Starter", color="tanoi", id="config-create"):
                    _create_config_action(new_name)
                if lcars.button("Duplicate Active", color="lilac", id="config-duplicate"):
                    _duplicate_config_action()
                if lcars.button("Save All Structured", color="tanoi", id="config-save-all"):
                    _save_config_action()
                if lcars.button("Validate Config", color="anakiwa", id="config-validate"):
                    _run_preflight_action()

            with lcars.data_panel("Coverage Map", color="blue-bell"):
                lcars.table(_coverage_rows(), title="Structured Surface", id="config-coverage-table")
                lcars.table(STATE.config_store.summary_rows(), title="Summary", id="config-page-summary-table")


def _config_setup_page() -> None:
    with lcars.page("Setup", id="config-setup", layout="console"):
        with lcars.console("Setup Workflow", color="pale-canary"):
            _setup_smart_panel()
            with lcars.data_panel("Defaults / Examples", color="blue-bell"):
                lcars.table(_setup_default_rows(), title="Axolotl Defaults And Starters", id="setup-defaults-table")
            for group, color in (
                ("Model", "pale-canary"),
                ("Dataset", "golden-tanoi"),
                ("Sequence / Packing", "anakiwa"),
                ("Run Safety", "lilac"),
            ):
                with lcars.data_panel(f"{group} Essentials", color=color):
                    _render_config_fields({group}, keys=SETUP_FIELD_KEYS, include_headers=False)
            _config_page_actions("setup")


def _config_train_page() -> None:
    with lcars.page("Train", id="config-train", layout="grid"):
        with lcars.padd("Training / Adapter / Optimizer", color="tanoi"):
            _render_config_fields({"Training", "Adapter / PEFT", "Optimizer"})
            _config_page_actions("train")


def _config_hardware_page() -> None:
    with lcars.page("Hardware", id="config-hardware", layout="grid"):
        with lcars.padd("Precision / Kernels / Distributed", color="anakiwa"):
            _render_config_fields({"Precision / Memory", "Attention / Kernels", "Distributed"})
            _config_page_actions("hardware")


def _config_tracking_page() -> None:
    with lcars.page("Tracking", id="config-tracking", layout="grid"):
        with lcars.padd("Tracking / Integrations / RL", color="lilac"):
            _render_config_fields({"Tracking", "Integrations", "RL / Evaluation"})
            _config_page_actions("tracking")


def _config_advanced_page() -> None:
    with lcars.page("Advanced", id="config-advanced", layout="console"):
        with lcars.console("Advanced Structured Axolotl Surface", color="blue-bell"):
            with lcars.padd("Advanced Setup Fields", color="blue-bell"):
                advanced_groups = {"Run Safety", "Model", "Dataset", "Sequence / Packing"}
                advanced_keys = {
                    spec.key
                    for spec in FIELD_SPECS
                    if spec.group in advanced_groups and spec.key not in SETUP_FIELD_KEYS
                }
                _render_config_fields(keys=advanced_keys, id_prefix="advanced")
                _config_page_actions("advanced")


def _run_page() -> None:
    with lcars.page("Run", id="run", layout="console"):
        with lcars.console("Axolotl Run Monitor", color="red"):
            with lcars.data_panel("Process State", color="red"):
                lcars.metric("Status", STATE.runner.status_label(), status=STATE.runner.status_severity(), color="red", id="run-status")
                command = " ".join(STATE.runner.state.command) if STATE.runner.state.command else "idle"
                lcars.text(command[:260], size="mono", id="run-command-text")
                lcars.log(LOG_AXOLOTL, max_lines=1000, title="Axolotl Output")

            with lcars.control_panel("Launch Controls", color="golden-tanoi"):
                action = lcars.select("Axolotl Action", list(AXOLOTL_ACTIONS), value="train", id="run-action")
                launcher = lcars.select("Launcher", ["", "python", "accelerate", "torchrun"], value="", id="run-launcher")
                _seed_text("run-cli-args", "")
                cli_args = lcars.text_input("Axolotl Args", placeholder="Action flags or fetch target, shell-style", autocomplete=False, id="run-cli-args")
                _seed_text("run-launcher-args", "")
                launcher_args = lcars.text_input("Launcher Args", placeholder="Placed after -- for accelerate/torchrun", autocomplete=False, id="run-launcher-args")
                if lcars.button("Start", color="red", id="run-start"):
                    _start_axolotl_action(action, launcher=launcher, cli_args=cli_args, launcher_args=launcher_args)
                if lcars.button("Stop", color="red", id="run-stop"):
                    _stop_axolotl_action()
                if lcars.button("Preflight", color="anakiwa", id="run-preflight-local"):
                    _run_preflight_action()

            with lcars.data_panel("Live Hardware", color="anakiwa"):
                snapshot = STATE.telemetry.latest or STATE.telemetry.sample()
                lcars.metric("CPU", f"{snapshot.cpu_percent:.0f}%", status=_percent_status(snapshot.cpu_percent), color="anakiwa", id="system-cpu")
                lcars.metric("RAM", f"{snapshot.ram_percent:.0f}%", status=_percent_status(snapshot.ram_percent), color="blue-bell", id="system-ram")
                lcars.table(gpu_rows(snapshot.gpus), title="GPU", id="run-gpu-table")


def _resources_page() -> None:
    with lcars.page("Resources", id="resources", layout="telemetry"):
        snapshot = STATE.telemetry.latest or STATE.telemetry.sample()
        cfg = _load_config_or_empty()
        with lcars.diagnostic("System Telemetry", color="anakiwa"):
            lcars.gauge("CPU Load", snapshot.cpu_percent, unit="%", warn_threshold=80, crit_threshold=92, id="cpu-gauge")
            lcars.gauge("Memory Load", snapshot.ram_percent, unit="%", warn_threshold=80, crit_threshold=92, id="ram-gauge")
            lcars.metric(
                "RAM Used",
                f"{format_bytes(snapshot.ram_used)} / {format_bytes(snapshot.ram_total)}",
                status=_percent_status(snapshot.ram_percent),
                color="blue-bell",
                id="ram-used-metric",
            )
            lcars.chart(STATE.telemetry.chart_payload(), title="Resource Trend", color="anakiwa", id="resource-chart")
            lcars.table(gpu_rows(snapshot.gpus), title="GPU Telemetry", id="gpu-table")
            lcars.table(process_rows(), title="Top RAM / CPU Processes", id="process-table")
            lcars.table(gpu_process_rows(), title="GPU Processes", id="gpu-process-table")
            lcars.table(disk_rows(snapshot.disks), title="Mounted Disks", id="disk-table")
            lcars.table(_storage_rows(cfg), title="Storage Hotspots", id="storage-hotspot-table")


def _hub_page() -> None:
    with lcars.page("HF Hub", id="hub", layout="console"):
        with lcars.data_panel("Search Results", color="lilac", zone="primary"):
            lcars.markdown(result_link_markdown(STATE.hf.search_results), id="hf-results-links")
            lcars.table(search_rows(STATE.hf.search_results), title="HF Results", id="hf-results-table")

        with lcars.control_panel("Search Controls", color="lilac", zone="side", id="search-command"):
            _seed_text("hf-query", "llama instruct")
            query = lcars.text_input("Search", placeholder="model or dataset query", autocomplete=False, id="hf-query")
            repo_type = lcars.select("Repo Type", ["model", "dataset"], value=STATE.hf.last_repo_type, id="hf-repo-type")
            sort = lcars.select("HF Sort", HF_SORT_OPTIONS, value="downloads", id="hf-sort")
            compatibility = lcars.select("Browse Filter", HF_COMPATIBILITY_OPTIONS, value=HF_COMPATIBILITY_OPTIONS[0], id="hf-compatibility")
            limit = lcars.select("Limit", HF_LIMIT_OPTIONS, value=HF_LIMIT_OPTIONS[0], id="hf-limit")
            vram_limit = lcars.number_input(
                "VRAM Limit GB [filter]",
                value=float(STATE.hf.vram_limit_gb or 24),
                min=1,
                max=256,
                step=1,
                id="hf-vram-limit",
            )
            if lcars.button("Run Search", color="anakiwa", id="hf-search"):
                _hf_search_action(
                    query,
                    repo_type,
                    sort=sort,
                    compatibility=compatibility,
                    limit=limit,
                    sift="",
                    local_sort="downloads",
                    artifact_filter=HF_ARTIFACT_FILTER_OPTIONS[0],
                    quant_filter=HF_QUANT_FILTER_OPTIONS[0],
                    fit_filter="any",
                    vram_limit=vram_limit,
                )
            if _is_active_action("hf-query"):
                _hf_search_action(
                    query,
                    repo_type,
                    sort=sort,
                    compatibility=compatibility,
                    limit=limit,
                    sift="",
                    local_sort="downloads",
                    artifact_filter=HF_ARTIFACT_FILTER_OPTIONS[0],
                    quant_filter=HF_QUANT_FILTER_OPTIONS[0],
                    fit_filter="any",
                    vram_limit=vram_limit,
                )
            _seed_text("hf-sift", "")
            sift = lcars.text_input("Contains [optional]", placeholder="repo, tag, quant, family", autocomplete=False, id="hf-sift")
            local_sort = lcars.select("Result Sort", HF_LOCAL_SORT_OPTIONS, value="downloads", id="hf-local-sort")
            artifact_filter = lcars.select("Artifact", HF_ARTIFACT_FILTER_OPTIONS, value=HF_ARTIFACT_FILTER_OPTIONS[0], id="hf-artifact-filter")
            quant_filter = lcars.select("Weight Format", HF_QUANT_FILTER_OPTIONS, value=HF_QUANT_FILTER_OPTIONS[0], id="hf-quant-filter")
            fit_filter = lcars.select("VRAM Fit", HF_FIT_FILTER_OPTIONS, value="any", id="hf-fit-filter")
            if lcars.button("Apply Sift", color="blue-bell", id="hf-apply-sift"):
                _hf_sift_action(
                    sift=sift,
                    local_sort=local_sort,
                    artifact_filter=artifact_filter,
                    quant_filter=quant_filter,
                    fit_filter=fit_filter,
                    vram_limit=vram_limit,
                )
            if _is_active_action("hf-sift"):
                _hf_sift_action(
                    sift=sift,
                    local_sort=local_sort,
                    artifact_filter=artifact_filter,
                    quant_filter=quant_filter,
                    fit_filter=fit_filter,
                    vram_limit=vram_limit,
                )
            hf_options = result_options(STATE.hf.search_results)
            if STATE.hf.last_repo_id and STATE.hf.last_repo_id not in hf_options:
                hf_options = [STATE.hf.last_repo_id, *[option for option in hf_options if option]]
            result_choice = lcars.select(
                "Search Result",
                hf_options,
                value=STATE.hf.last_repo_id if STATE.hf.last_repo_id in hf_options else "",
                id="hf-result-select",
            )
            _seed_text("hf-repo-id", STATE.hf.last_repo_id, force=True)
            repo_id = lcars.text_input("Copy Repo ID", placeholder="owner/name", autocomplete=False, id="hf-repo-id")
            _seed_text("hf-revision", "")
            revision = lcars.text_input("Revision [optional]", placeholder="branch/tag/commit", autocomplete=False, id="hf-revision")
            if lcars.button("Select Result", color="blue-bell", id="hf-select-result"):
                _hf_select_result_action(result_choice, revision)
            if lcars.button("Inspect Repo", color="anakiwa", id="hf-inspect"):
                _hf_inspect_action(repo_id, repo_type, revision)
            if lcars.button("Find Fine-Tunes", color="lilac", id="hf-related"):
                _hf_related_action(repo_id)
            if lcars.button("Download Selected", color="golden-tanoi", id="hf-download"):
                _hf_download_action(repo_id, repo_type, revision)
            if lcars.button("Use Repo In Config", color="tanoi", id="hf-use-repo"):
                _hf_use_repo_action(repo_id, repo_type)
            if lcars.button("Use Last Local Snapshot", color="blue-bell", id="hf-use-local"):
                _hf_use_last_local_action(repo_type)

        with lcars.data_panel("Selection", color="anakiwa", zone="dock", id="selection-panel"):
            lcars.markdown(_hf_selected_markdown(repo_id, repo_type), id="hf-selected-link")
            lcars.table(detail_summary_rows(STATE.hf.selected_details), title="Selected Repo", id="hf-detail-table")
            lcars.table(detail_file_rows(STATE.hf.selected_details), title="Compatible / Quant Files", id="hf-files-table")
            lcars.table(related_rows(STATE.hf.related_results), title="Related Models", id="hf-related-table")

        with lcars.data_panel("Transfers", color="golden-tanoi", zone="dock"):
            lcars.table(STATE.hf.job_rows(), title="Download Jobs", id="hf-jobs-table")
            lcars.log(LOG_HF, max_lines=300, title="HF Activity")


def _content_page() -> None:
    rows, total_text, total_bytes = STATE.hf.cache_rows()
    with lcars.page("Content", id="content", layout="telemetry"):
        with lcars.diagnostic("Downloaded Content Manager", color="blue-bell"):
            lcars.metric("HF Cache", cache_summary_text(total_bytes, total_text), status="ok", color="blue-bell", id="hf-cache-total")
            lcars.table(rows or [{"Type": "", "Repo": "No cached Hugging Face repos", "Size": "", "Files": "", "Revision": "", "Path": ""}], title="HF Cache", id="hf-cache-table")
            with lcars.control_panel("Cache Disposal", color="golden-tanoi"):
                _seed_text("delete-repo-id", STATE.hf.last_repo_id)
                repo_id = lcars.text_input("Delete Repo ID", placeholder="owner/name", autocomplete=False, id="delete-repo-id")
                repo_type = lcars.select("Delete Repo Type", ["model", "dataset"], value=STATE.hf.last_repo_type, id="delete-repo-type")
                if lcars.button("Refresh Cache", color="anakiwa", id="cache-refresh"):
                    _update_cache_widgets()
                    lcars.notify("HF cache refreshed.")
                if lcars.button("Delete Cached Repo", color="red", id="cache-delete"):
                    _delete_cache_action(repo_id, repo_type)


def _ollama_page() -> None:
    with lcars.page("Ollama", id="ollama", layout="grid"):
        with lcars.padd("Ollama Detection", color="pale-canary"):
            lcars.table(_ollama_rule_rows(), title="Axolotl Source Gate", id="ollama-rule-table")
            lcars.table(STATE.ollama.rows(), title="Local Ollama Models", id="ollama-table")
            _seed_text("ollama-model-name", STATE.ollama.models[0].name if STATE.ollama.models else "")
            model_name = lcars.text_input("Ollama Model Name", placeholder="name:tag", autocomplete=False, id="ollama-model-name")
            if lcars.button("Refresh Ollama", color="anakiwa", id="ollama-refresh"):
                _ollama_refresh_action()
            if lcars.button("Search HF Source", color="lilac", id="ollama-search-hf"):
                _ollama_search_hf_action(model_name)
            if lcars.button("Use Compatible Source", color="tanoi", id="ollama-use-source"):
                _ollama_use_source_action(model_name)


def _setup_smart_panel() -> None:
    with lcars.control_panel("Smart Setup", color="pale-canary", zone="primary"):
        recipe = lcars.select("Recipe", list(SETUP_RECIPES), value="LoRA SFT starter", id="setup-recipe")
        model = lcars.select("Model Preset", MODEL_PRESETS, value=MODEL_PRESETS[0], id="setup-model-preset")
        dataset = lcars.select("Dataset Preset", list(DATASET_PRESETS), value=next(iter(DATASET_PRESETS)), id="setup-dataset-preset")
        if lcars.button("Apply Recipe", color="tanoi", id="setup-apply-recipe"):
            _setup_apply_recipe_action(recipe)
        if lcars.button("Apply Model", color="anakiwa", id="setup-apply-model"):
            _setup_apply_model_action(model)
        if lcars.button("Apply Dataset", color="golden-tanoi", id="setup-apply-dataset"):
            _setup_apply_dataset_action(dataset)
        if lcars.button("Use HF Selection", color="lilac", id="setup-use-hf"):
            _hf_use_repo_action(STATE.hf.last_repo_id, STATE.hf.last_repo_type)
        if lcars.button("Search Model Preset", color="blue-bell", id="setup-search-model"):
            _hf_search_action(model, "model", sort="downloads", compatibility=HF_COMPATIBILITY_OPTIONS[0], limit="12")


def _setup_default_rows() -> list[dict[str, str]]:
    cfg = _load_config_or_empty()
    specs = [
        ("base_model", "Required", "None", "NousResearch/Llama-3.2-1B", "HF model id or local Transformers directory"),
        ("datasets.0.path", "Required", "None", "teknium/GPT4-LLM-Cleaned", "HF dataset id, local file, or local directory"),
        ("datasets.0.type", "Recommended", "None", "alpaca", "Axolotl formatter strategy"),
        ("datasets.0.ds_type", "Optional", "Infer local file extension", "json", "Only needed for local files/directories"),
        ("sequence_len", "Required for most runs", "None", "2048", "Context length used for tokenization/training"),
        ("sample_packing", "Optional", "Unset unless configured", "true", "Packs multiple samples into one sequence"),
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


def _hf_selected_markdown(repo_id: str, repo_type: str) -> str:
    repo_id = repo_id.strip()
    if not repo_id:
        return "No repository selected."
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"HF path: `huggingface.co/{prefix}{repo_id}`"


def _ollama_rule_rows() -> list[dict[str, str]]:
    return [
        {"Source": "Local Transformers dir", "Action": "Apply", "Reason": "config/tokenizer/weights can be read by Axolotl"},
        {"Source": "hf.co / model name", "Action": "Search HF", "Reason": "find original safetensors repo or compatible fine-tune"},
        {"Source": "GGUF/internal blob", "Action": "Block", "Reason": "runtime artifact, not an Axolotl base_model"},
    ]


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
                lcars.header(current_group, size="h3", color="pale-canary", id=f"hdr-{prefix}{safe_group_id}")
                note = CONFIG_GROUP_NOTES.get(current_group)
                if note:
                    lcars.text(note, id=f"note-{prefix}{safe_group_id}")
        values[spec.widget_id] = _render_field(spec, cfg)
    return values


def _render_field(spec: FieldSpec, cfg: dict[str, Any]) -> Any:
    value = STATE.config_store.field_value(spec, cfg)
    label = _field_label(spec)
    if spec.kind in {"text", "csv_list", "json"}:
        _seed_text(spec.widget_id, str(value or ""))
        if spec.kind == "json":
            return lcars.text_input(
                label,
                placeholder=spec.placeholder or "{key: value}",
                autocomplete=False,
                id=spec.widget_id,
            )
        return lcars.text_input(label, placeholder=spec.placeholder, autocomplete=False, id=spec.widget_id)
    if spec.kind == "number":
        if spec.optional:
            _seed_text(spec.widget_id, "" if value in (None, "") else str(value))
            return lcars.text_input(
                label,
                placeholder=spec.placeholder or "unset",
                autocomplete=False,
                id=spec.widget_id,
            )
        return lcars.number_input(
            label,
            value=float(value if value not in ("", None) else spec.default or 0),
            min=spec.minimum,
            max=spec.maximum,
            step=spec.step,
            id=spec.widget_id,
        )
    if spec.kind == "bool":
        return lcars.toggle(label, value=bool(value), id=spec.widget_id)
    if spec.kind == "tri_bool":
        selected = str(value if value not in (None, "") else "unset")
        if selected not in {"unset", "true", "false"}:
            selected = "true" if selected.lower() in {"1", "yes", "on"} else selected
        return lcars.select(label, ["unset", "true", "false"], value=selected, id=spec.widget_id)
    selected = str(value if value not in (None, "") else (spec.default or ""))
    if spec.options and selected not in spec.options:
        selected = ""
    return lcars.select(label, list(spec.options), value=selected, id=spec.widget_id)


def _field_label(spec: FieldSpec) -> str:
    if spec.key in SETUP_REQUIRED_KEYS:
        return f"{spec.label} [required]"
    if spec.optional:
        return f"{spec.label} [optional]"
    if spec.default is not None:
        return f"{spec.label} [ui {spec.default}]"
    return spec.label


def _config_page_actions(suffix: str) -> None:
    with lcars.control_panel("Page Actions", color="golden-tanoi"):
        if lcars.button("Save Config", color="tanoi", id=f"config-save-{suffix}"):
            _save_config_action()
        if lcars.button("Run Preflight", color="anakiwa", id=f"config-preflight-{suffix}"):
            _run_preflight_action()
        lcars.markdown("[Raw YAML editor](/raw)", id=f"raw-link-{suffix}")


def _switch_config_action(selected: str) -> None:
    try:
        STATE.config_store.set_active(selected)
        lcars.notify(f"Active config switched to {selected}.")
        _update_config_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Could not switch config: {exc}", level="error")


def _create_config_action(new_name: str) -> None:
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
    try:
        payload = values if values is not None else _collect_editor_values()
        STATE.config_store.save_editor_values(payload)
        lcars.notify("Structured config saved.")
        _update_config_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Config save failed: {exc}", level="error")


def _duplicate_config_action() -> None:
    try:
        new_name = STATE.config_store.create_copy("copy-of-" + STATE.config_store.active_name)
        lcars.notify(f"Created config {new_name}.")
    except Exception as exc:
        lcars.notify(f"Could not duplicate config: {exc}", level="error")


def _run_preflight_action() -> None:
    issues = STATE.refresh_preflight()
    _update_preflight_widgets(issues)
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warn")
    if errors:
        lcars.notify(f"Preflight blocked launch: {errors} error(s), {warnings} warning(s).", level="error")
    else:
        lcars.notify(f"Preflight passed with {warnings} warning(s).")


def _start_axolotl_action(
    action: str,
    *,
    launcher: str = "",
    cli_args: str = "",
    launcher_args: str = "",
) -> None:
    try:
        if launcher and action not in LAUNCHER_ACTIONS:
            lcars.notify(f"{action} does not accept launcher mode. Clear Launcher and retry.", level="error")
            return
        if launcher_args.strip() and not launcher:
            lcars.notify("Launcher Args require python, accelerate, or torchrun launcher mode.", level="error")
            return
        if action in CONFIG_ACTIONS:
            issues = STATE.refresh_preflight()
            _update_preflight_widgets(issues)
            errors = [issue for issue in issues if issue.severity == "error"]
            if errors:
                lcars.notify(f"Axolotl launch blocked by preflight: {errors[0].detail}", level="error")
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
    except Exception as exc:
        lcars.notify(f"Unable to start Axolotl: {exc}", level="error")


def _stop_axolotl_action() -> None:
    STATE.runner.stop()
    lcars.notify("Axolotl stop requested.")
    lcars.update("run-status", value=STATE.runner.status_label(), status=STATE.runner.status_severity())


def _setup_apply_recipe_action(recipe: str) -> None:
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
    try:
        STATE.config_store.apply_model(model)
        lcars.notify(f"Applied model preset: {model}.")
        _update_config_widgets()
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Could not apply model preset: {exc}", level="error")


def _setup_apply_dataset_action(dataset: str) -> None:
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
    vram = _optional_float(vram_limit)
    results = STATE.hf.search(
        query,
        repo_type,  # type: ignore[arg-type]
        sort=sort,
        compatible_only=compatibility == HF_COMPATIBILITY_OPTIONS[0],
        limit=_bounded_int(limit, default=12, minimum=1, maximum=50),
    )
    results = STATE.hf.sift_results(
        text=sift,
        sort=local_sort,
        artifact_filter=artifact_filter,
        quant_filter=quant_filter,
        fit_filter=fit_filter,
        vram_limit_gb=vram,
    )
    _update_hf_widgets()
    if results:
        _set_widget_value("hf-repo-id", results[0].repo_id)
    _append_hf_logs()


def _hf_sift_action(
    *,
    sift: str,
    local_sort: str,
    artifact_filter: str,
    quant_filter: str,
    fit_filter: str,
    vram_limit: float | int | str,
) -> None:
    results = STATE.hf.sift_results(
        text=sift,
        sort=local_sort,
        artifact_filter=artifact_filter,
        quant_filter=quant_filter,
        fit_filter=fit_filter,
        vram_limit_gb=_optional_float(vram_limit),
    )
    _update_hf_widgets()
    if results:
        _set_widget_value("hf-repo-id", results[0].repo_id)
    _append_hf_logs()


def _hf_select_result_action(repo_id: str, revision: str = "") -> None:
    result = STATE.hf.select_result(repo_id)
    if result is None:
        lcars.notify("Select a Hugging Face result first.", level="error")
        return
    _set_widget_value("hf-repo-id", result.repo_id)
    _set_widget_value("hf-repo-type", result.repo_type)
    STATE.hf.inspect_repo(result.repo_id, result.repo_type, revision=revision.strip() or None)
    _update_hf_widgets()
    _append_hf_logs()


def _hf_inspect_action(repo_id: str, repo_type: str, revision: str = "") -> None:
    if repo_type not in {"model", "dataset"}:
        lcars.notify("Repo type must be model or dataset.", level="error")
        return
    details = STATE.hf.inspect_repo(repo_id, repo_type, revision=revision.strip() or None)  # type: ignore[arg-type]
    if details is not None:
        _set_widget_value("hf-repo-id", details.result.repo_id)
        _set_widget_value("hf-repo-type", details.result.repo_type)
    _update_hf_widgets()
    _append_hf_logs()


def _hf_related_action(repo_id: str) -> None:
    STATE.hf.find_related_models(repo_id)
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
    try:
        if repo_type == "model":
            result = _hf_result_for(repo_id)
            if result is not None and result.role == "peft_adapter":
                STATE.config_store.apply_updates({"lora_model_dir": repo_id, "adapter": "lora"})
            elif _looks_gguf(repo_id) or (result is not None and result.blocked):
                detail = result.compatibility if result is not None else "likely GGUF/runtime artifact"
                lcars.notify(f"Refusing to set incompatible model repo as Axolotl base_model: {detail}", level="error")
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
        _run_preflight_action()
    except Exception as exc:
        lcars.notify(f"Could not apply repo: {exc}", level="error")


def _hf_use_last_local_action(repo_type: str) -> None:
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
    lcars.update("ollama-table", **_table_payload(STATE.ollama.rows()))
    if STATE.ollama.last_error:
        lcars.notify(STATE.ollama.last_error, level="error")
    else:
        lcars.notify(f"Detected {len(STATE.ollama.models)} Ollama model(s).")


def _ollama_search_hf_action(model_name: str) -> None:
    model = STATE.ollama.select(model_name.strip())
    if model is None:
        lcars.notify("Ollama model was not found. Refresh and enter the exact name:tag.", level="error")
        return
    query = model.hf_query or model.hf_hint or model.name.split(":", 1)[0]
    results = STATE.hf.search(query, "model", limit=12, sort="downloads", compatible_only=True)
    if model.hf_hint and not _looks_gguf(model.hf_hint):
        STATE.hf.inspect_repo(model.hf_hint, "model")
    _update_hf_widgets()
    if results:
        _set_widget_value("hf-query", query)
        _set_widget_value("hf-repo-id", results[0].repo_id)
    lcars.notify(f"HF model search loaded for Ollama source: {query}.")
    _append_hf_logs()


def _ollama_use_source_action(model_name: str) -> None:
    model = STATE.ollama.select(model_name.strip())
    if model is None:
        lcars.notify("Ollama model was not found. Refresh and enter the exact name:tag.", level="error")
        return
    if not model.compatible:
        lcars.notify(f"Blocked: {model.name} is not Axolotl-readable. {model.reason}", level="error")
        return
    STATE.config_store.apply_model(model.compatible_path)
    lcars.notify(f"Applied Ollama source path to base_model: {model.compatible_path}")
    _update_config_widgets()
    _run_preflight_action()


def live_tick() -> None:
    snapshot = STATE.telemetry.sample()
    STATE.resource_tick += 1
    lcars.update("system-cpu", value=f"{snapshot.cpu_percent:.0f}%", status=_percent_status(snapshot.cpu_percent))
    lcars.update("system-ram", value=f"{snapshot.ram_percent:.0f}%", status=_percent_status(snapshot.ram_percent))
    lcars.update("cpu-gauge", value=snapshot.cpu_percent)
    lcars.update("ram-gauge", value=snapshot.ram_percent)
    lcars.update(
        "ram-used-metric",
        value=f"{format_bytes(snapshot.ram_used)} / {format_bytes(snapshot.ram_total)}",
        status=_percent_status(snapshot.ram_percent),
    )
    lcars.update("gpu-table", **_table_payload(gpu_rows(snapshot.gpus)))
    lcars.update("run-gpu-table", **_table_payload(gpu_rows(snapshot.gpus)))
    lcars.update("process-table", **_table_payload(process_rows()))
    lcars.update("gpu-process-table", **_table_payload(gpu_process_rows()))
    lcars.update("disk-table", **_table_payload(disk_rows(snapshot.disks)))
    if STATE.resource_tick % 5 == 0:
        lcars.update("storage-hotspot-table", **_table_payload(_storage_rows(_load_config_or_empty())))
    lcars.update("resource-chart", series=_series_payload(STATE.telemetry.chart_payload()))
    lcars.update("run-status", value=STATE.runner.status_label(), status=STATE.runner.status_severity())
    _append_runner_logs()
    _append_hf_logs()
    _update_cache_widgets(live=True)


def _append_runner_logs() -> None:
    lines = STATE.runner.drain_logs()
    if lines:
        lcars.append_log(LOG_AXOLOTL, *lines)


def _append_hf_logs() -> None:
    lines = STATE.hf.drain_logs()
    if lines:
        lcars.append_log(LOG_HF, *lines)


def _update_preflight_widgets(issues: list[PreflightIssue]) -> None:
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warn")
    lcars.update("preflight-table", **_table_payload(issue_rows(issues)))
    lcars.update("run-gate-metric", value="BLOCKED" if errors else "READY", status="crit" if errors else "ok")
    lcars.update("warning-count-metric", value=str(warnings), status="warn" if warnings else "ok")


def _update_config_widgets() -> None:
    summary = STATE.config_store.summary_rows()
    payload = _table_payload(summary)
    lcars.update("config-summary-table", **payload)
    lcars.update("config-page-summary-table", **payload)
    lcars.update("config-coverage-table", **_table_payload(_coverage_rows()))
    lcars.update("active-config-select", value=STATE.config_store.active_name)
    try:
        values = STATE.config_store.editor_values()
    except Exception:
        return
    store = get_session_state(get_ctx().session_id)
    for widget_id, value in values.items():
        store[widget_id] = value
        lcars.update(widget_id, value=value)
    lcars.update("setup-defaults-table", **_table_payload(_setup_default_rows()))


def _update_hf_widgets() -> None:
    selected_options = result_options(STATE.hf.search_results)
    selected_value = STATE.hf.last_repo_id if STATE.hf.last_repo_id in selected_options else selected_options[0]
    lcars.update("hf-results-table", **_table_payload(search_rows(STATE.hf.search_results)))
    lcars.update("hf-results-links", content=result_link_markdown(STATE.hf.search_results))
    lcars.update("hf-detail-table", **_table_payload(detail_summary_rows(STATE.hf.selected_details)))
    lcars.update("hf-files-table", **_table_payload(detail_file_rows(STATE.hf.selected_details)))
    lcars.update("hf-related-table", **_table_payload(related_rows(STATE.hf.related_results)))
    lcars.update("hf-jobs-table", **_table_payload(STATE.hf.job_rows()))
    lcars.update("hf-result-select", options=_select_options(selected_options), value=selected_value)
    lcars.update("hf-selected-link", content=_hf_selected_markdown(STATE.hf.last_repo_id, STATE.hf.last_repo_type))


def _update_cache_widgets(*, live: bool = False) -> None:
    try:
        rows, total_text, total_bytes = STATE.hf.cache_rows()
    except Exception:
        return
    lcars.update("hf-cache-total", value=cache_summary_text(total_bytes, total_text))
    lcars.update("hf-cache-table", **_table_payload(rows or [{"Type": "", "Repo": "No cached Hugging Face repos", "Size": "", "Files": "", "Revision": "", "Path": ""}]))
    lcars.update("hf-jobs-table", **_table_payload(STATE.hf.job_rows()))
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


def create_lcars_app(ui_fn: Callable[[], None], *, live_fn: Callable[[], None] | None = None) -> FastAPI:
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
    event_bus = app.state.event_bus

    async def _dsl_action_handler(action_id: str, value: Any, session_id: str = "http_fallback") -> None:
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

    _install_hf_routes(app, ui_fn, build_ctx.config)
    _install_raw_editor(app)
    return app


def _install_hf_routes(app: FastAPI, ui_fn: Callable[[], None], config: Any) -> None:
    @app.get("/hf/select/{repo_type}/{repo_id:path}", include_in_schema=False)
    def hf_select(repo_type: str, repo_id: str) -> RedirectResponse:
        repo_id = unquote(repo_id).strip()
        if repo_type in {"model", "dataset"} and repo_id:
            result = STATE.hf.select_result(repo_id)
            selected_type = result.repo_type if result is not None else repo_type
            STATE.hf.inspect_repo(repo_id, selected_type)  # type: ignore[arg-type]
            _refresh_manifest(app, ui_fn, config)
        return RedirectResponse("/?page=hub")

    _move_last_route_before_spa(app)


def _refresh_manifest(app: FastAPI, ui_fn: Callable[[], None], config: Any) -> None:
    build_ctx = _LCARSContext(
        mode=Mode.BUILD,
        session_id="build",
        builder=_ManifestBuilder(),
        config=config,
    )
    set_ctx(build_ctx)
    ui_fn()
    assert build_ctx.builder is not None
    app.state.manifest = build_ctx.builder.build(build_ctx.config)


def _install_raw_editor(app: FastAPI) -> None:
    @app.get("/raw", response_class=HTMLResponse, include_in_schema=False)
    def raw_get(request: Request) -> str:
        _ = request
        return _raw_html()

    _move_last_route_before_spa(app)

    @app.post("/raw", response_class=HTMLResponse, include_in_schema=False)
    def raw_post(content: str = Form(...)) -> str:
        status = ""
        try:
            STATE.config_store.save_raw_text(content)
            STATE.refresh_preflight()
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


def _table_payload(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"headers": [], "rows": []}
    headers = list(rows[0].keys())
    return {
        "headers": headers,
        "rows": [
            {"id": f"row-{index}", "cells": [str(row.get(header, "")) for header in headers]}
            for index, row in enumerate(rows)
        ],
    }


def _select_options(options: list[str]) -> list[dict[str, str]]:
    return [{"label": option or "(none)", "value": option} for option in options]


def _set_widget_value(widget_id: str, value: str) -> None:
    ctx = get_ctx()
    store = get_session_state(ctx.session_id)
    store[widget_id] = value
    lcars.update(widget_id, value=value)


def _bounded_int(value: str, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _optional_float(value: float | int | str) -> float | None:
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _is_active_action(widget_id: str) -> bool:
    ctx = get_ctx()
    return ctx.mode == Mode.HANDLE and ctx.active_action_id == widget_id


def _series_payload(data: dict[str, list[float]]) -> list[dict[str, Any]]:
    return [{"name": name, "data": values} for name, values in data.items()]


def _percent_status(value: float) -> str:
    if value >= 92:
        return "crit"
    if value >= 80:
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

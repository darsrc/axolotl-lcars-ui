"""Axolotl-focused preflight validation and compatibility checks."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import scan_cache_dir

from axolotl_lcars_ui.ollama import OllamaManager
from axolotl_lcars_ui.resources import format_bytes


Severity = Literal["error", "warn", "ok"]


@dataclass(frozen=True)
class PreflightIssue:
    severity: Severity
    check: str
    detail: str


class AxolotlPreflight:
    """Checks the config against the formats and workflows Axolotl actually consumes."""

    def __init__(self, project_root: Path, ollama: OllamaManager) -> None:
        self.project_root = project_root
        self.ollama = ollama

    def validate(self, cfg: dict[str, Any]) -> list[PreflightIssue]:
        issues: list[PreflightIssue] = []
        self._check_base_model(cfg, issues)
        self._check_dataset(cfg, issues)
        self._check_training_knobs(cfg, issues)
        self._check_attention(cfg, issues)
        self._check_distributed(cfg, issues)
        self._check_schedule_and_checkpoints(cfg, issues)
        self._check_integrations(cfg, issues)
        self._check_rl_eval(cfg, issues)
        self._check_output_space(cfg, issues)
        if not any(issue.severity == "error" for issue in issues):
            issues.append(PreflightIssue("ok", "Launch Gate", "No blocking Axolotl preflight errors."))
        return issues

    def has_errors(self, cfg: dict[str, Any]) -> bool:
        return any(issue.severity == "error" for issue in self.validate(cfg))

    def _check_base_model(self, cfg: dict[str, Any], issues: list[PreflightIssue]) -> None:
        base_model = str(cfg.get("base_model") or "").strip()
        if not base_model:
            issues.append(
                PreflightIssue(
                    "error",
                    "base_model",
                    "Axolotl requires a Hugging Face model id or local model directory.",
                )
            )
            return

        if _looks_gguf(base_model):
            issues.append(
                PreflightIssue(
                    "error",
                    "Model Format",
                    "GGUF repos/files are for llama.cpp/Ollama runtime use. Axolotl base_model needs HF/Transformers weights such as safetensors, pt, or bin.",
                )
            )
            return

        local = self._resolve_local_ref(base_model)
        if local is not None:
            self._check_local_model_dir(local, issues)
        else:
            self._check_cached_hf_model(base_model, issues)
            matching_ollama = self._ollama_model(base_model)
            if matching_ollama is not None and not matching_ollama.compatible:
                issues.append(
                    PreflightIssue(
                        "error",
                        "Ollama Model",
                        f"{base_model} is an Ollama runtime model, not an Axolotl-readable local model path: {matching_ollama.reason}",
                    )
                )

        for key in ("base_model_config", "tokenizer_config"):
            value = str(cfg.get(key) or "").strip()
            if not value or "/" in value and not value.startswith((".", "/", "~")):
                continue
            path = self._resolve_path(value)
            if value and not path.exists():
                issues.append(PreflightIssue("error", key, f"Configured local path does not exist: {path}"))

    def _check_local_model_dir(self, path: Path, issues: list[PreflightIssue]) -> None:
        if path.is_file() and path.suffix.lower() == ".gguf":
            issues.append(
                PreflightIssue(
                    "error",
                    "Model Format",
                    f"{path} is GGUF. Axolotl cannot train directly from Ollama/llama.cpp GGUF files.",
                )
            )
            return
        if not path.exists():
            issues.append(PreflightIssue("error", "Local Model", f"Path does not exist: {path}"))
            return
        if not path.is_dir():
            issues.append(
                PreflightIssue("error", "Local Model", f"Local base_model must be a directory, not a file: {path}")
            )
            return
        if not (path / "config.json").exists():
            issues.append(
                PreflightIssue("error", "Local Model", f"Missing config.json in local model directory: {path}")
            )
        has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin")) or any(path.glob("*.pt"))
        if not has_weights:
            issues.append(
                PreflightIssue(
                    "error",
                    "Local Model",
                    f"No .safetensors, .bin, or .pt weights found in local model directory: {path}",
                )
            )
        has_tokenizer = any((path / name).exists() for name in ("tokenizer.json", "tokenizer.model"))
        if not has_tokenizer:
            issues.append(
                PreflightIssue(
                    "warn",
                    "Tokenizer",
                    "No tokenizer.json/tokenizer.model found beside the local model. Set tokenizer_config if it lives elsewhere.",
                )
            )

    def _check_cached_hf_model(self, repo_id: str, issues: list[PreflightIssue]) -> None:
        try:
            cache = scan_cache_dir()
        except Exception:
            return
        for repo in cache.repos:
            if repo.repo_id != repo_id or str(repo.repo_type or "model") != "model":
                continue
            filenames = [
                file.file_name.lower()
                for revision in repo.revisions
                for file in revision.files
            ]
            if filenames and all(name.endswith(".gguf") for name in filenames):
                issues.append(
                    PreflightIssue(
                        "error",
                        "Cached HF Model",
                        f"{repo_id} is cached as GGUF-only content. Pick/download the non-GGUF Transformers repo for Axolotl.",
                    )
                )
            return

    def _check_dataset(self, cfg: dict[str, Any], issues: list[PreflightIssue]) -> None:
        datasets = cfg.get("datasets")
        pretraining = cfg.get("pretraining_dataset")
        if not datasets and not pretraining:
            issues.append(
                PreflightIssue(
                    "error",
                    "Dataset",
                    "Set datasets or pretraining_dataset before running Axolotl.",
                )
            )
            return
        if datasets is not None:
            if not isinstance(datasets, list) or not datasets:
                issues.append(PreflightIssue("error", "datasets", "datasets must be a non-empty list."))
                return
            first = datasets[0]
            if not isinstance(first, dict):
                issues.append(PreflightIssue("error", "datasets[0]", "Each dataset entry must be a mapping."))
                return
            path = str(first.get("path") or "").strip()
            if not path:
                issues.append(PreflightIssue("error", "Dataset Path", "datasets[0].path is required."))
            ds_type = str(first.get("type") or "").strip()
            if not ds_type:
                issues.append(
                    PreflightIssue(
                        "warn",
                        "Dataset Type",
                        "datasets[0].type is empty. Axolotl may need an explicit format such as alpaca, completion, or chat_template.",
                    )
                )
            if ds_type == "completion" and not first.get("field"):
                issues.append(
                    PreflightIssue(
                        "warn",
                        "Completion Dataset",
                        "completion datasets usually need datasets[0].field pointing at the text column.",
                    )
                )
            if ds_type in {"chat_template", "sharegpt"} and not (
                first.get("field_messages") or first.get("chat_template")
            ):
                issues.append(
                    PreflightIssue(
                        "warn",
                        "Chat Dataset",
                        "Chat-style datasets usually need field_messages and an explicit chat_template or tokenizer default.",
                    )
                )
            if first.get("chat_template") == "jinja" and not first.get("chat_template_jinja"):
                issues.append(
                    PreflightIssue(
                        "error",
                        "Chat Template",
                        "chat_template: jinja requires chat_template_jinja.",
                    )
                )
            if path and self._looks_local(path):
                local = self._resolve_path(path)
                if not local.exists():
                    issues.append(PreflightIssue("error", "Dataset Path", f"Local dataset path does not exist: {local}"))
                elif local.is_file() and not first.get("ds_type"):
                    suffix = local.suffix.lower().lstrip(".")
                    hint = "json" if suffix == "jsonl" else suffix
                    issues.append(
                        PreflightIssue(
                            "warn",
                            "Local Dataset",
                            f"Local dataset files should set datasets[0].ds_type. Suggested value: {hint or 'json'}",
                        )
                    )
            if _truthy(cfg.get("streaming")) and cfg.get("dataset_prepared_path"):
                issues.append(
                    PreflightIssue(
                        "warn",
                        "Streaming Dataset",
                        "streaming: true usually bypasses/precludes a prepared dataset cache path.",
                    )
                )
            if cfg.get("dataset_processes") and cfg.get("dataset_num_proc"):
                issues.append(
                    PreflightIssue(
                        "warn",
                        "Dataset Workers",
                        "dataset_processes and dataset_num_proc both tune preprocessing parallelism; keep one source of truth.",
                    )
                )

    def _check_training_knobs(self, cfg: dict[str, Any], issues: list[PreflightIssue]) -> None:
        load_in_8bit = _truthy(cfg.get("load_in_8bit"))
        load_in_4bit = _truthy(cfg.get("load_in_4bit"))
        if load_in_8bit and load_in_4bit:
            issues.append(
                PreflightIssue(
                    "error",
                    "Quantization",
                    "load_in_8bit and load_in_4bit are mutually exclusive.",
                )
            )
        adapter = str(cfg.get("adapter") or "").lower()
        if adapter == "qlora" and not load_in_4bit:
            issues.append(
                PreflightIssue("warn", "QLoRA", "adapter: qlora normally pairs with load_in_4bit: true.")
            )
        if load_in_4bit and adapter not in {"qlora", "lora"}:
            issues.append(
                PreflightIssue("warn", "4-bit", "4-bit loading is normally used with LoRA/QLoRA adapters.")
            )
        if _truthy(cfg.get("bf16")) and _truthy(cfg.get("fp16")):
            issues.append(PreflightIssue("error", "Precision", "bf16 and fp16 cannot both be true."))
        if _truthy(cfg.get("fp8")) and not _truthy_or_auto(cfg.get("torch_compile")):
            issues.append(
                PreflightIssue("warn", "FP8", "fp8 usually requires torch_compile enabled for Axolotl's FP8 path.")
            )
        if _truthy(cfg.get("gptq")) and not load_in_4bit:
            issues.append(PreflightIssue("warn", "GPTQ", "gptq models should normally set load_in_4bit: true."))
        if not adapter and any(key in cfg for key in ("lora_r", "lora_alpha", "lora_dropout")):
            issues.append(
                PreflightIssue(
                    "warn",
                    "Adapter",
                    "LoRA parameters are set but adapter is empty. Set adapter: lora/qlora or remove LoRA fields.",
                )
            )
        if _truthy(cfg.get("lora_target_linear")) and cfg.get("lora_target_modules"):
            issues.append(
                PreflightIssue(
                    "warn",
                    "LoRA Targets",
                    "lora_target_linear and lora_target_modules both select target modules; using both can be confusing.",
                )
            )
        for key in ("sequence_len", "micro_batch_size", "gradient_accumulation_steps"):
            value = cfg.get(key)
            if value is None:
                continue
            try:
                if float(value) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                issues.append(PreflightIssue("error", key, f"{key} must be a positive number."))
        deepspeed = str(cfg.get("deepspeed") or "").strip()
        if deepspeed:
            path = self._resolve_path(deepspeed)
            if not path.exists():
                issues.append(PreflightIssue("error", "DeepSpeed", f"DeepSpeed config not found: {path}"))

    def _check_attention(self, cfg: dict[str, Any], issues: list[PreflightIssue]) -> None:
        legacy_keys = (
            "flash_attention",
            "xformers_attention",
            "sdp_attention",
            "sage_attention",
            "flex_attention",
        )
        enabled = [key for key in legacy_keys if _truthy(cfg.get(key))]
        attn_impl = str(cfg.get("attn_implementation") or "").strip()
        if len(enabled) > 1:
            issues.append(
                PreflightIssue(
                    "error",
                    "Attention",
                    f"Only one legacy attention flag should be enabled, found: {', '.join(enabled)}.",
                )
            )
        if attn_impl and enabled:
            issues.append(
                PreflightIssue(
                    "warn",
                    "Attention",
                    "attn_implementation is the modern Axolotl control; clear legacy attention booleans unless needed.",
                )
            )
        if _truthy(cfg.get("flash_attn_fuse_mlp")) and not (
            attn_impl.startswith("flash_attention") or _truthy(cfg.get("flash_attention"))
        ):
            issues.append(
                PreflightIssue(
                    "warn",
                    "Flash Attention",
                    "flash_attn_fuse_mlp is only useful when using a Flash Attention backend.",
                )
            )

    def _check_distributed(self, cfg: dict[str, Any], issues: list[PreflightIssue]) -> None:
        deepspeed = str(cfg.get("deepspeed") or "").strip()
        fsdp = cfg.get("fsdp")
        if deepspeed and fsdp:
            issues.append(
                PreflightIssue(
                    "error",
                    "Distributed",
                    "Configure either DeepSpeed or FSDP for a run, not both.",
                )
            )
        adapter = str(cfg.get("adapter") or "").lower()
        fsdp_enabled = bool(fsdp)
        if adapter == "qlora" and fsdp_enabled and not cfg.get("qlora_sharded_model_loading"):
            issues.append(
                PreflightIssue(
                    "warn",
                    "FSDP QLoRA",
                    "QLoRA with FSDP should usually enable qlora_sharded_model_loading.",
                )
            )
        fsdp_config = cfg.get("fsdp_config")
        if isinstance(fsdp_config, dict):
            if _truthy(fsdp_config.get("offload_params")) and not _truthy(fsdp_config.get("cpu_offload_pin_memory")):
                issues.append(
                    PreflightIssue(
                        "warn",
                        "FSDP Offload",
                        "FSDP parameter offload without cpu_offload_pin_memory can be slower and less predictable.",
                    )
                )
            version = cfg.get("fsdp_version")
            if version is not None:
                try:
                    if int(version) not in {1, 2}:
                        raise ValueError
                except (TypeError, ValueError):
                    issues.append(PreflightIssue("error", "FSDP", "fsdp_version must be 1 or 2."))
        for key in ("context_parallel_size", "tensor_parallel_size"):
            value = cfg.get(key)
            if value is None:
                continue
            try:
                if int(value) < 1:
                    raise ValueError
            except (TypeError, ValueError):
                issues.append(PreflightIssue("error", key, f"{key} must be a positive integer."))

    def _check_schedule_and_checkpoints(self, cfg: dict[str, Any], issues: list[PreflightIssue]) -> None:
        if cfg.get("warmup_steps") is not None and cfg.get("warmup_ratio") is not None:
            issues.append(
                PreflightIssue(
                    "warn",
                    "Warmup",
                    "warmup_steps and warmup_ratio are both set; prefer one warmup policy.",
                )
            )
        if cfg.get("max_steps") is not None and cfg.get("num_epochs") is not None:
            issues.append(
                PreflightIssue(
                    "warn",
                    "Training Length",
                    "max_steps overrides epoch-based training length. Confirm this is intentional.",
                )
            )
        if cfg.get("eval_steps") is not None and cfg.get("evals_per_epoch") is not None:
            issues.append(
                PreflightIssue(
                    "warn",
                    "Evaluation Cadence",
                    "eval_steps and evals_per_epoch are both set; keep one eval cadence.",
                )
            )
        if cfg.get("save_steps") is not None and cfg.get("saves_per_epoch") is not None:
            issues.append(
                PreflightIssue(
                    "warn",
                    "Checkpoint Cadence",
                    "save_steps and saves_per_epoch are both set; keep one checkpoint cadence.",
                )
            )
        if _truthy(cfg.get("save_only_model")) and (
            cfg.get("resume_from_checkpoint") or _truthy(cfg.get("auto_resume_from_checkpoints"))
        ):
            issues.append(
                PreflightIssue(
                    "warn",
                    "Resume",
                    "save_only_model omits trainer state, so future resume behavior may not work as expected.",
                )
            )
        if _truthy(cfg.get("load_best_model_at_end")) and str(cfg.get("eval_strategy") or "").lower() in {"", "no"}:
            issues.append(
                PreflightIssue(
                    "error",
                    "Best Model",
                    "load_best_model_at_end requires an evaluation strategy.",
                )
            )
        if _truthy(cfg.get("load_best_model_at_end")) and str(cfg.get("save_strategy") or "").lower() == "no":
            issues.append(
                PreflightIssue(
                    "error",
                    "Best Model",
                    "load_best_model_at_end requires checkpoint saving.",
                )
            )

    def _check_integrations(self, cfg: dict[str, Any], issues: list[PreflightIssue]) -> None:
        wants_hf_write = bool(cfg.get("hub_model_id") or cfg.get("push_dataset_to_hub"))
        has_hf_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        if wants_hf_write and not (has_hf_token or _truthy(cfg.get("hf_use_auth_token"))):
            issues.append(
                PreflightIssue(
                    "warn",
                    "Hugging Face Auth",
                    "Hub push is configured but no HF token was detected in the environment.",
                )
            )
        wandb_mode = str(cfg.get("wandb_mode") or "").lower()
        if _truthy(cfg.get("use_wandb")) and wandb_mode not in {"offline", "disabled"} and not os.environ.get("WANDB_API_KEY"):
            issues.append(
                PreflightIssue(
                    "warn",
                    "Weights & Biases",
                    "use_wandb is enabled but WANDB_API_KEY is not set. Use offline/disabled mode or authenticate first.",
                )
            )
        if _truthy(cfg.get("use_mlflow")) and not (cfg.get("mlflow_tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")):
            issues.append(
                PreflightIssue(
                    "warn",
                    "MLflow",
                    "use_mlflow is enabled but no tracking URI was configured.",
                )
            )
        if _truthy(cfg.get("use_comet")) and not os.environ.get("COMET_API_KEY"):
            issues.append(
                PreflightIssue(
                    "warn",
                    "Comet",
                    "use_comet is enabled but COMET_API_KEY is not set.",
                )
            )

    def _check_rl_eval(self, cfg: dict[str, Any], issues: list[PreflightIssue]) -> None:
        rl_mode = str(cfg.get("rl") or "").strip()
        datasets = cfg.get("datasets")
        first = datasets[0] if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict) else {}
        ds_type = str(first.get("type") or "").lower()
        if rl_mode and ds_type in {"alpaca", "completion"}:
            issues.append(
                PreflightIssue(
                    "warn",
                    "RL Dataset",
                    f"rl: {rl_mode} usually needs preference/reward-style dataset fields, not plain {ds_type}.",
                )
            )
        trl = cfg.get("trl")
        use_vllm = isinstance(trl, dict) and _truthy(trl.get("use_vllm"))
        vllm = cfg.get("vllm")
        if use_vllm and not (isinstance(vllm, dict) and vllm.get("gpu_memory_utilization")):
            issues.append(
                PreflightIssue(
                    "warn",
                    "vLLM",
                    "TRL use_vllm is enabled; set vllm.gpu_memory_utilization to avoid surprise memory pressure.",
                )
            )
        lm_eval_tasks = cfg.get("lm_eval_tasks")
        if lm_eval_tasks is not None and not lm_eval_tasks:
            issues.append(PreflightIssue("warn", "LM Eval", "lm_eval_tasks is set but empty."))

    def _check_output_space(self, cfg: dict[str, Any], issues: list[PreflightIssue]) -> None:
        output = str(cfg.get("output_dir") or "./outputs").strip()
        path = self._resolve_path(output)
        probe = path if path.exists() else path.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
        except OSError:
            return
        if usage.free < 10 * 1024**3:
            issues.append(
                PreflightIssue(
                    "warn",
                    "Disk Space",
                    f"Only {format_bytes(usage.free)} free near output_dir. Fine-tuning checkpoints can consume tens of GB.",
                )
            )

    def _resolve_local_ref(self, value: str) -> Path | None:
        if self._looks_local(value):
            return self._resolve_path(value)
        path = self._resolve_path(value)
        return path if path.exists() else None

    def _looks_local(self, value: str) -> bool:
        return value.startswith((".", "/", "~")) or "\\" in value

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _ollama_model(self, name: str):
        if not self.ollama.models:
            self.ollama.refresh()
        for model in self.ollama.models:
            if model.name == name:
                return model
        return None


def _looks_gguf(value: str) -> bool:
    lowered = value.lower()
    parts = [part for chunk in lowered.replace("_", "-").split("/") for part in chunk.split("-")]
    return lowered.endswith(".gguf") or "gguf" in parts or "-gguf" in lowered or "_gguf" in lowered


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _truthy_or_auto(value: Any) -> bool:
    if _truthy(value):
        return True
    return str(value or "").strip().lower() == "auto"


def issue_rows(issues: list[PreflightIssue]) -> list[dict[str, str]]:
    if not issues:
        return [{"Level": "ok", "Check": "Preflight", "Detail": "No checks have run yet."}]
    return [
        {
            "Level": issue.severity.upper(),
            "Check": issue.check,
            "Detail": issue.detail,
        }
        for issue in issues
    ]

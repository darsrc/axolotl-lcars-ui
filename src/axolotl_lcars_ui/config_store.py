"""Axolotl YAML configuration storage and structured-field mapping."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


FieldKind = Literal["text", "number", "bool", "select", "tri_bool", "csv_list", "json"]


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: FieldKind
    widget_id: str
    group: str
    placeholder: str = ""
    options: tuple[str, ...] = ()
    default: Any = None
    optional: bool = False
    step: float = 1.0
    minimum: float | None = None
    maximum: float | None = None


OPTIMIZER_OPTIONS = (
    "adamw_torch_fused",
    "adamw_torch",
    "adamw_torch_xla",
    "adamw_torch_npu_fused",
    "adamw_apex_fused",
    "adopt_adamw",
    "adafactor",
    "adamw_anyprecision",
    "adamw_torch_4bit",
    "ademamix",
    "sgd",
    "adagrad",
    "adamw_bnb_8bit",
    "adamw_8bit",
    "ademamix_8bit",
    "lion_8bit",
    "lion_32bit",
    "paged_adamw_32bit",
    "paged_adamw_8bit",
    "paged_ademamix_32bit",
    "paged_ademamix_8bit",
    "paged_lion_32bit",
    "paged_lion_8bit",
    "rmsprop",
    "rmsprop_bnb",
    "rmsprop_bnb_8bit",
    "rmsprop_bnb_32bit",
    "galore_adamw",
    "galore_adamw_8bit",
    "galore_adafactor",
    "galore_adamw_layerwise",
    "galore_adamw_8bit_layerwise",
    "galore_adafactor_layerwise",
    "lomo",
    "adalomo",
    "grokadamw",
    "schedule_free_adamw",
    "schedule_free_sgd",
    "apollo_adamw",
    "apollo_adamw_layerwise",
    "optimi_adamw",
    "ao_adamw_8bit",
    "ao_adamw_fp8",
    "came_pytorch",
)

LR_SCHEDULER_OPTIONS = (
    "cosine",
    "linear",
    "constant",
    "constant_with_warmup",
    "cosine_with_restarts",
    "polynomial",
    "inverse_sqrt",
    "reduce_lr_on_plateau",
)

ATTENTION_IMPL_OPTIONS = (
    "",
    "eager",
    "sdpa",
    "flash_attention_2",
    "flash_attention_3",
    "flex_attention",
    "xformers",
    "sage",
    "fp8",
)

CHAT_TEMPLATE_OPTIONS = (
    "",
    "tokenizer_default",
    "alpaca",
    "inst",
    "chatml",
    "gemma",
    "cohere",
    "llama3",
    "phi_3",
    "deepseek_v2",
    "jamba",
    "tokenizer_default_fallback_chatml",
    "jinja",
)


def _widget_id_for(key: str) -> str:
    return "cfg-adv-" + key.replace("_", "-").replace(".", "-")


def _extra(
    key: str,
    label: str,
    kind: FieldKind,
    group: str,
    placeholder: str = "",
    *,
    options: tuple[str, ...] = (),
    default: Any = None,
    optional: bool = True,
    step: float = 1.0,
    minimum: float | None = None,
    maximum: float | None = None,
) -> FieldSpec:
    return FieldSpec(
        key=key,
        label=label,
        kind=kind,
        widget_id=_widget_id_for(key),
        group=group,
        placeholder=placeholder,
        options=options,
        default=default,
        optional=optional,
        step=step,
        minimum=minimum,
        maximum=maximum,
    )


BOOL_STRING_SELECT_KEYS = {
    "activation_offloading",
    "batch_flattening",
    "bf16",
    "bfloat16",
    "float16",
    "float32",
    "fp16",
    "fp8",
    "gradient_checkpointing",
    "tf32",
    "torch_compile",
}

DEFAULT_CONFIG_NAME = "lora-starter.yml"


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("strict", "Strict Config", "bool", "cfg-strict", "Run Safety", default=False),
    FieldSpec("resume_from_checkpoint", "Resume Checkpoint", "text", "cfg-resume-from-checkpoint", "Run Safety", "checkpoint path", optional=True),
    FieldSpec("auto_resume_from_checkpoints", "Auto Resume Checkpoints", "tri_bool", "cfg-auto-resume", "Run Safety", optional=True),
    FieldSpec("save_only_model", "Save Only Model", "tri_bool", "cfg-save-only-model", "Run Safety", optional=True),
    FieldSpec("base_model", "Base Model", "text", "cfg-base-model", "Model", "HF repo id or local HF snapshot path"),
    FieldSpec("revision_of_model", "Model Revision", "text", "cfg-revision-of-model", "Model", "branch, tag, or commit", optional=True),
    FieldSpec("base_model_config", "Base Model Config", "text", "cfg-base-model-config", "Model", "Optional alternate config path", optional=True),
    FieldSpec("base_model_ignore_patterns", "Model Ignore Patterns", "csv_list", "cfg-base-model-ignore-patterns", "Model", "*.gguf, *.onnx", optional=True),
    FieldSpec("cls_model_config", "Model Config Class", "text", "cfg-cls-model-config", "Model", "LlamaConfig, MistralConfig, ...", optional=True),
    FieldSpec("tokenizer_config", "Tokenizer Config", "text", "cfg-tokenizer-config", "Model", "Optional tokenizer path", optional=True),
    FieldSpec("model_type", "Model Type", "select", "cfg-model-type", "Model", options=("AutoModelForCausalLM", "AutoModelForSeq2SeqLM"), default="AutoModelForCausalLM"),
    FieldSpec("tokenizer_type", "Tokenizer Type", "select", "cfg-tokenizer-type", "Model", options=("AutoTokenizer", "LlamaTokenizer", "MistralCommonTokenizer"), default="AutoTokenizer"),
    FieldSpec("tokenizer_use_fast", "Fast Tokenizer", "tri_bool", "cfg-tokenizer-use-fast", "Model", optional=True),
    FieldSpec("tokenizer_use_mistral_common", "Mistral Common Tokenizer", "tri_bool", "cfg-tokenizer-use-mistral-common", "Model", optional=True),
    FieldSpec("trust_remote_code", "Trust Remote Code", "tri_bool", "cfg-trust-remote-code", "Model", optional=True),
    FieldSpec("processor_type", "Processor Type", "text", "cfg-processor-type", "Model", "AutoProcessor or model-specific processor", optional=True),
    FieldSpec("datasets.0.path", "Dataset Path", "text", "cfg-dataset-path", "Dataset", "HF dataset repo id or local path"),
    FieldSpec("datasets.0.type", "Dataset Type", "text", "cfg-dataset-type", "Dataset", "alpaca, completion, chat_template, sharegpt, ...", default="alpaca"),
    FieldSpec("datasets.0.split", "Dataset Split", "text", "cfg-dataset-split", "Dataset", "train", optional=True),
    FieldSpec("datasets.0.name", "Dataset Config Name", "text", "cfg-dataset-name", "Dataset", "optional HF dataset subset/config", optional=True),
    FieldSpec("datasets.0.data_files", "Dataset Data Files", "csv_list", "cfg-dataset-data-files", "Dataset", "file1.jsonl, file2.jsonl", optional=True),
    FieldSpec("datasets.0.ds_type", "Local File Type", "text", "cfg-dataset-ds-type", "Dataset", "json, jsonl, parquet, csv", optional=True),
    FieldSpec("datasets.0.field", "Completion Text Field", "text", "cfg-dataset-field", "Dataset", "text", optional=True),
    FieldSpec("datasets.0.field_messages", "Messages Field", "text", "cfg-dataset-field-messages", "Dataset", "messages", optional=True),
    FieldSpec("datasets.0.chat_template", "Dataset Chat Template", "select", "cfg-dataset-chat-template", "Dataset", options=CHAT_TEMPLATE_OPTIONS, optional=True),
    FieldSpec("datasets.0.chat_template_jinja", "Chat Template Jinja", "text", "cfg-chat-template-jinja", "Dataset", "template text or jinja file path", optional=True),
    FieldSpec("datasets.0.train_on_eos", "Train On EOS", "select", "cfg-train-on-eos", "Dataset", options=("", "all", "turn", "last", "none"), optional=True),
    FieldSpec("dataset_prepared_path", "Prepared Dataset Path", "text", "cfg-dataset-prepared-path", "Dataset", "Preprocessed dataset output path", optional=True),
    FieldSpec("val_set_size", "Validation Set Size", "number", "cfg-val-set-size", "Dataset", default=0.1, step=0.01, minimum=0),
    FieldSpec("shuffle_merged_datasets", "Shuffle Merged Datasets", "tri_bool", "cfg-shuffle-merged-datasets", "Dataset", optional=True),
    FieldSpec("shuffle_before_merging_datasets", "Shuffle Before Merge", "tri_bool", "cfg-shuffle-before-merging", "Dataset", optional=True),
    FieldSpec("streaming", "Streaming Dataset", "tri_bool", "cfg-streaming", "Dataset", optional=True),
    FieldSpec("dataset_processes", "Dataset Processes", "number", "cfg-dataset-processes", "Dataset", optional=True, step=1, minimum=1),
    FieldSpec("dataset_num_proc", "Dataset Num Proc", "number", "cfg-dataset-num-proc", "Dataset", optional=True, step=1, minimum=1),
    FieldSpec("dataset_exact_deduplication", "Exact Deduplication", "tri_bool", "cfg-dataset-exact-deduplication", "Dataset", optional=True),
    FieldSpec("dataset_keep_in_memory", "Keep Dataset In Memory", "tri_bool", "cfg-dataset-keep-in-memory", "Dataset", optional=True),
    FieldSpec("dataloader_num_workers", "Dataloader Workers", "number", "cfg-dataloader-num-workers", "Dataset", optional=True, step=1, minimum=0),
    FieldSpec("dataloader_pin_memory", "Dataloader Pin Memory", "tri_bool", "cfg-dataloader-pin-memory", "Dataset", optional=True),
    FieldSpec("push_dataset_to_hub", "Push Prepared Dataset", "text", "cfg-push-dataset-to-hub", "Dataset", "org/repo", optional=True),
    FieldSpec("hf_use_auth_token", "HF Auth Token", "tri_bool", "cfg-hf-use-auth-token", "Dataset", optional=True),
    FieldSpec("sequence_len", "Sequence Length", "number", "cfg-sequence-len", "Sequence / Packing", default=2048, step=128, minimum=1),
    FieldSpec("eval_sequence_len", "Eval Sequence Length", "number", "cfg-eval-sequence-len", "Sequence / Packing", optional=True, step=128, minimum=1),
    FieldSpec("excess_length_strategy", "Excess Length Strategy", "select", "cfg-excess-length-strategy", "Sequence / Packing", options=("", "drop", "truncate", "raise"), optional=True),
    FieldSpec("min_sample_len", "Min Sample Length", "number", "cfg-min-sample-len", "Sequence / Packing", optional=True, step=1, minimum=0),
    FieldSpec("max_prompt_len", "Max Prompt Length", "number", "cfg-max-prompt-len", "Sequence / Packing", optional=True, step=128, minimum=1),
    FieldSpec("sample_packing", "Sample Packing", "bool", "cfg-sample-packing", "Sequence / Packing", default=True),
    FieldSpec("eval_sample_packing", "Eval Sample Packing", "tri_bool", "cfg-eval-sample-packing", "Sequence / Packing", optional=True),
    FieldSpec("pad_to_sequence_len", "Pad To Sequence Length", "bool", "cfg-pad-to-sequence-len", "Sequence / Packing", default=True),
    FieldSpec("pad_to_multiple_of", "Pad To Multiple Of", "number", "cfg-pad-to-multiple-of", "Sequence / Packing", optional=True, step=8, minimum=1),
    FieldSpec("sample_packing_group_size", "Packing Group Size", "number", "cfg-sample-packing-group-size", "Sequence / Packing", optional=True, step=1000, minimum=1),
    FieldSpec("sample_packing_bin_size", "Packing Bin Size", "number", "cfg-sample-packing-bin-size", "Sequence / Packing", optional=True, step=1, minimum=1),
    FieldSpec("sample_packing_sequentially", "Pack Sequentially", "tri_bool", "cfg-sample-packing-sequentially", "Sequence / Packing", optional=True),
    FieldSpec("curriculum_sampling", "Curriculum Sampling", "tri_bool", "cfg-curriculum-sampling", "Sequence / Packing", optional=True),
    FieldSpec("batch_flattening", "Batch Flattening", "select", "cfg-batch-flattening", "Sequence / Packing", options=("", "auto", "true", "false"), optional=True),
    FieldSpec("output_dir", "Output Dir", "text", "cfg-output-dir", "Training", "./outputs/lora-out", default="./outputs/lora-out"),
    FieldSpec("hub_model_id", "Push Model Repo", "text", "cfg-hub-model-id", "Training", "org/repo", optional=True),
    FieldSpec("hub_strategy", "Hub Strategy", "select", "cfg-hub-strategy", "Training", options=("", "end", "checkpoint", "all_checkpoints", "every_save"), optional=True),
    FieldSpec("hub_revision", "Hub Revision", "text", "cfg-hub-revision", "Training", "main", optional=True),
    FieldSpec("save_safetensors", "Save Safetensors", "tri_bool", "cfg-save-safetensors", "Training", optional=True),
    FieldSpec("adapter", "Adapter", "select", "cfg-adapter", "Adapter / PEFT", options=("", "lora", "qlora", "ia3"), default="lora", optional=True),
    FieldSpec("lora_model_dir", "LoRA Model Dir", "text", "cfg-lora-model-dir", "Adapter / PEFT", "existing adapter or output checkpoint", optional=True),
    FieldSpec("lora_r", "LoRA Rank", "number", "cfg-lora-r", "Adapter / PEFT", default=16, step=1, minimum=1),
    FieldSpec("lora_alpha", "LoRA Alpha", "number", "cfg-lora-alpha", "Adapter / PEFT", default=32, step=1, minimum=1),
    FieldSpec("lora_dropout", "LoRA Dropout", "number", "cfg-lora-dropout", "Adapter / PEFT", default=0.05, step=0.01, minimum=0, maximum=1),
    FieldSpec("lora_target_modules", "LoRA Target Modules", "csv_list", "cfg-lora-target-modules", "Adapter / PEFT", "q_proj, v_proj, k_proj, o_proj", optional=True),
    FieldSpec("lora_target_linear", "Target All Linear", "tri_bool", "cfg-lora-target-linear", "Adapter / PEFT", optional=True),
    FieldSpec("lora_modules_to_save", "LoRA Modules To Save", "csv_list", "cfg-lora-modules-to-save", "Adapter / PEFT", "embed_tokens, lm_head", optional=True),
    FieldSpec("peft_use_dora", "Use DoRA", "tri_bool", "cfg-peft-use-dora", "Adapter / PEFT", optional=True),
    FieldSpec("peft_use_rslora", "Use RSLoRA", "tri_bool", "cfg-peft-use-rslora", "Adapter / PEFT", optional=True),
    FieldSpec("qlora_sharded_model_loading", "QLoRA Sharded Loading", "tri_bool", "cfg-qlora-sharded-model-loading", "Adapter / PEFT", optional=True),
    FieldSpec("lora_on_cpu", "LoRA On CPU", "tri_bool", "cfg-lora-on-cpu", "Adapter / PEFT", optional=True),
    FieldSpec("gptq", "GPTQ 4-bit Model", "tri_bool", "cfg-gptq", "Adapter / PEFT", optional=True),
    FieldSpec("bnb_config_kwargs", "BNB Config Kwargs", "json", "cfg-bnb-config-kwargs", "Adapter / PEFT", "{bnb_4bit_quant_type: nf4}", optional=True),
    FieldSpec("loraplus_lr_ratio", "LoRA+ LR Ratio", "number", "cfg-loraplus-lr-ratio", "Adapter / PEFT", optional=True, step=1, minimum=0),
    FieldSpec("merge_lora", "Merge LoRA", "tri_bool", "cfg-merge-lora", "Adapter / PEFT", optional=True),
    FieldSpec("merge_method", "Merge Method", "select", "cfg-merge-method", "Adapter / PEFT", options=("", "memory_efficient", "legacy"), optional=True),
    FieldSpec("load_in_8bit", "Load In 8 Bit", "bool", "cfg-load-in-8bit", "Precision / Memory", default=True),
    FieldSpec("load_in_4bit", "Load In 4 Bit", "bool", "cfg-load-in-4bit", "Precision / Memory", default=False),
    FieldSpec("micro_batch_size", "Micro Batch Size", "number", "cfg-micro-batch-size", "Optimizer", default=2, step=1, minimum=1),
    FieldSpec("gradient_accumulation_steps", "Gradient Accumulation", "number", "cfg-gradient-accumulation", "Optimizer", default=4, step=1, minimum=1),
    FieldSpec("batch_size", "Total Batch Size", "number", "cfg-batch-size", "Optimizer", optional=True, step=1, minimum=1),
    FieldSpec("eval_batch_size", "Eval Batch Size", "number", "cfg-eval-batch-size", "Optimizer", optional=True, step=1, minimum=1),
    FieldSpec("auto_find_batch_size", "Auto Find Batch Size", "tri_bool", "cfg-auto-find-batch-size", "Optimizer", optional=True),
    FieldSpec("num_epochs", "Epochs", "number", "cfg-num-epochs", "Optimizer", default=3, step=1, minimum=0),
    FieldSpec("max_steps", "Max Steps", "number", "cfg-max-steps", "Optimizer", optional=True, step=1, minimum=1),
    FieldSpec("learning_rate", "Learning Rate", "number", "cfg-learning-rate", "Optimizer", default=0.0001, step=0.00001, minimum=0),
    FieldSpec("embedding_lr", "Embedding LR", "number", "cfg-embedding-lr", "Optimizer", optional=True, step=0.000001, minimum=0),
    FieldSpec("optimizer", "Optimizer", "select", "cfg-optimizer", "Optimizer", options=OPTIMIZER_OPTIONS, default="adamw_bnb_8bit"),
    FieldSpec("optim_args", "Optimizer Args", "json", "cfg-optim-args", "Optimizer", "{rank: 128, update_proj_gap: 200}", optional=True),
    FieldSpec("optim_target_modules", "Optimizer Target Modules", "csv_list", "cfg-optim-target-modules", "Optimizer", "self_attn, mlp, all_linear", optional=True),
    FieldSpec("weight_decay", "Weight Decay", "number", "cfg-weight-decay", "Optimizer", optional=True, step=0.001, minimum=0),
    FieldSpec("max_grad_norm", "Max Grad Norm", "number", "cfg-max-grad-norm", "Optimizer", optional=True, step=0.1, minimum=0),
    FieldSpec("adam_beta1", "Adam Beta1", "number", "cfg-adam-beta1", "Optimizer", optional=True, step=0.01, minimum=0, maximum=1),
    FieldSpec("adam_beta2", "Adam Beta2", "number", "cfg-adam-beta2", "Optimizer", optional=True, step=0.001, minimum=0, maximum=1),
    FieldSpec("adam_epsilon", "Adam Epsilon", "number", "cfg-adam-epsilon", "Optimizer", optional=True, step=0.00000001, minimum=0),
    FieldSpec("lr_scheduler", "LR Scheduler", "select", "cfg-lr-scheduler", "Optimizer", options=LR_SCHEDULER_OPTIONS, default="cosine"),
    FieldSpec("lr_scheduler_kwargs", "LR Scheduler Kwargs", "json", "cfg-lr-scheduler-kwargs", "Optimizer", "{num_cycles: 0.5}", optional=True),
    FieldSpec("warmup_steps", "Warmup Steps", "number", "cfg-warmup-steps", "Optimizer", default=10, step=1, minimum=0),
    FieldSpec("warmup_ratio", "Warmup Ratio", "number", "cfg-warmup-ratio", "Optimizer", optional=True, step=0.01, minimum=0, maximum=1),
    FieldSpec("cosine_min_lr_ratio", "Cosine Min LR Ratio", "number", "cfg-cosine-min-lr-ratio", "Optimizer", optional=True, step=0.01, minimum=0, maximum=1),
    FieldSpec("bf16", "BF16", "select", "cfg-bf16", "Precision / Memory", options=("auto", "true", "false"), default="auto"),
    FieldSpec("fp16", "FP16", "tri_bool", "cfg-fp16", "Precision / Memory", optional=True),
    FieldSpec("fp8", "FP8", "tri_bool", "cfg-fp8", "Precision / Memory", optional=True),
    FieldSpec("tf32", "TF32", "select", "cfg-tf32", "Precision / Memory", options=("", "auto", "true", "false"), optional=True),
    FieldSpec("gradient_checkpointing", "Gradient Checkpointing", "select", "cfg-gradient-checkpointing", "Precision / Memory", options=("", "true", "false", "offload", "offload_disk"), default="true"),
    FieldSpec("activation_offloading", "Activation Offloading", "select", "cfg-activation-offloading", "Precision / Memory", options=("", "true", "false", "legacy", "disk", "hidden_states"), optional=True),
    FieldSpec("low_cpu_mem_usage", "Low CPU Mem Usage", "tri_bool", "cfg-low-cpu-mem-usage", "Precision / Memory", optional=True),
    FieldSpec("gpu_memory_limit", "GPU Memory Limit", "text", "cfg-gpu-memory-limit", "Precision / Memory", "24GiB or 24", optional=True),
    FieldSpec("max_memory", "Max Memory Map", "json", "cfg-max-memory", "Precision / Memory", "{0: 24GiB, cpu: 64GiB}", optional=True),
    FieldSpec("torch_empty_cache_steps", "CUDA Empty Cache Steps", "number", "cfg-torch-empty-cache-steps", "Precision / Memory", optional=True, step=1, minimum=1),
    FieldSpec("gc_collect_steps", "GC Collect Steps", "number", "cfg-gc-collect-steps", "Precision / Memory", optional=True, step=1),
    FieldSpec("attn_implementation", "Attention Backend", "select", "cfg-attn-implementation", "Attention / Kernels", options=ATTENTION_IMPL_OPTIONS, optional=True),
    FieldSpec("flash_attention", "Flash Attention Legacy", "tri_bool", "cfg-flash-attention", "Attention / Kernels", optional=True),
    FieldSpec("xformers_attention", "XFormers Legacy", "tri_bool", "cfg-xformers-attention", "Attention / Kernels", optional=True),
    FieldSpec("sdp_attention", "SDP Legacy", "tri_bool", "cfg-sdp-attention", "Attention / Kernels", optional=True),
    FieldSpec("flex_attention", "Flex Attention", "tri_bool", "cfg-flex-attention", "Attention / Kernels", optional=True),
    FieldSpec("sage_attention", "Sage Attention Legacy", "tri_bool", "cfg-sage-attention", "Attention / Kernels", optional=True),
    FieldSpec("flash_attn_cross_entropy", "FlashAttn Cross Entropy", "tri_bool", "cfg-flash-attn-cross-entropy", "Attention / Kernels", optional=True),
    FieldSpec("flash_attn_fuse_mlp", "FlashAttn Fuse MLP", "tri_bool", "cfg-flash-attn-fuse-mlp", "Attention / Kernels", optional=True),
    FieldSpec("flash_optimum", "BetterTransformers", "tri_bool", "cfg-flash-optimum", "Attention / Kernels", optional=True),
    FieldSpec("torch_compile", "Torch Compile", "select", "cfg-torch-compile", "Attention / Kernels", options=("", "auto", "true", "false"), optional=True),
    FieldSpec("torch_compile_backend", "Compile Backend", "text", "cfg-torch-compile-backend", "Attention / Kernels", "inductor", optional=True),
    FieldSpec("torch_compile_mode", "Compile Mode", "select", "cfg-torch-compile-mode", "Attention / Kernels", options=("", "default", "reduce-overhead", "max-autotune"), optional=True),
    FieldSpec("deepspeed", "DeepSpeed Config", "text", "cfg-deepspeed", "Distributed", "deepspeed_configs/zero3.json", optional=True),
    FieldSpec("deepcompile", "DeepCompile", "tri_bool", "cfg-deepcompile", "Distributed", optional=True),
    FieldSpec("fsdp", "FSDP", "csv_list", "cfg-fsdp", "Distributed", "full_shard, auto_wrap", optional=True),
    FieldSpec("fsdp_version", "FSDP Version", "number", "cfg-fsdp-version", "Distributed", optional=True, step=1, minimum=1, maximum=2),
    FieldSpec("fsdp_config.activation_checkpointing", "FSDP Activation Ckpt", "tri_bool", "cfg-fsdp-activation-checkpointing", "Distributed", optional=True),
    FieldSpec("fsdp_config.offload_params", "FSDP Offload Params", "tri_bool", "cfg-fsdp-offload-params", "Distributed", optional=True),
    FieldSpec("fsdp_config.cpu_ram_efficient_loading", "FSDP CPU RAM Efficient", "tri_bool", "cfg-fsdp-cpu-ram-efficient", "Distributed", optional=True),
    FieldSpec("fsdp_config.cpu_offload_pin_memory", "FSDP CPU Pin Memory", "tri_bool", "cfg-fsdp-cpu-offload-pin-memory", "Distributed", optional=True),
    FieldSpec("fsdp_config.use_orig_params", "FSDP Use Orig Params", "tri_bool", "cfg-fsdp-use-orig-params", "Distributed", optional=True),
    FieldSpec("fsdp_config.state_dict_type", "FSDP State Dict", "select", "cfg-fsdp-state-dict-type", "Distributed", options=("", "FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"), optional=True),
    FieldSpec("context_parallel_size", "Context Parallel Size", "number", "cfg-context-parallel-size", "Distributed", optional=True, step=1, minimum=1),
    FieldSpec("tensor_parallel_size", "Tensor Parallel Size", "number", "cfg-tensor-parallel-size", "Distributed", optional=True, step=1, minimum=1),
    FieldSpec("ddp", "DDP", "tri_bool", "cfg-ddp", "Distributed", optional=True),
    FieldSpec("ddp_find_unused_parameters", "DDP Find Unused Params", "tri_bool", "cfg-ddp-find-unused-parameters", "Distributed", optional=True),
    FieldSpec("logging_steps", "Logging Steps", "number", "cfg-logging-steps", "Tracking", default=1, step=1, minimum=1),
    FieldSpec("save_steps", "Save Steps", "number", "cfg-save-steps", "Tracking", default=100, step=1, minimum=1),
    FieldSpec("saves_per_epoch", "Saves Per Epoch", "number", "cfg-saves-per-epoch", "Tracking", optional=True, step=1, minimum=1),
    FieldSpec("save_strategy", "Save Strategy", "select", "cfg-save-strategy", "Tracking", options=("", "no", "epoch", "steps", "best"), optional=True),
    FieldSpec("save_total_limit", "Save Total Limit", "number", "cfg-save-total-limit", "Tracking", optional=True, step=1, minimum=1),
    FieldSpec("save_first_step", "Save First Step", "tri_bool", "cfg-save-first-step", "Tracking", optional=True),
    FieldSpec("eval_steps", "Eval Steps", "number", "cfg-eval-steps", "Tracking", default=100, step=1, minimum=1),
    FieldSpec("evals_per_epoch", "Evals Per Epoch", "number", "cfg-evals-per-epoch", "Tracking", optional=True, step=1, minimum=1),
    FieldSpec("eval_strategy", "Eval Strategy", "select", "cfg-eval-strategy", "Tracking", options=("", "no", "epoch", "steps"), optional=True),
    FieldSpec("early_stopping_patience", "Early Stop Patience", "number", "cfg-early-stopping-patience", "Tracking", optional=True, step=1, minimum=1),
    FieldSpec("load_best_model_at_end", "Load Best At End", "tri_bool", "cfg-load-best-model-at-end", "Tracking", optional=True),
    FieldSpec("metric_for_best_model", "Best Model Metric", "text", "cfg-metric-for-best-model", "Tracking", "eval_loss", optional=True),
    FieldSpec("greater_is_better", "Greater Is Better", "tri_bool", "cfg-greater-is-better", "Tracking", optional=True),
    FieldSpec("loss_watchdog_threshold", "Loss Watchdog Threshold", "number", "cfg-loss-watchdog-threshold", "Tracking", optional=True, step=0.1, minimum=0),
    FieldSpec("loss_watchdog_patience", "Loss Watchdog Patience", "number", "cfg-loss-watchdog-patience", "Tracking", optional=True, step=1, minimum=1),
    FieldSpec("include_tkps", "Tokens/sec Per GPU", "tri_bool", "cfg-include-tkps", "Tracking", optional=True),
    FieldSpec("generate_samples", "Generate Samples", "tri_bool", "cfg-generate-samples", "Tracking", optional=True),
    FieldSpec("num_generation_samples", "Generation Samples", "number", "cfg-num-generation-samples", "Tracking", optional=True, step=1, minimum=1),
    FieldSpec("generation_max_new_tokens", "Generation Max Tokens", "number", "cfg-generation-max-new-tokens", "Tracking", optional=True, step=1, minimum=1),
    FieldSpec("generation_temperature", "Generation Temperature", "number", "cfg-generation-temperature", "Tracking", optional=True, step=0.1, minimum=0),
    FieldSpec("generation_top_p", "Generation Top P", "number", "cfg-generation-top-p", "Tracking", optional=True, step=0.05, minimum=0, maximum=1),
    FieldSpec("use_wandb", "Use W&B", "tri_bool", "cfg-use-wandb", "Integrations", optional=True),
    FieldSpec("wandb_project", "W&B Project", "text", "cfg-wandb-project", "Integrations", optional=True),
    FieldSpec("wandb_name", "W&B Run Name", "text", "cfg-wandb-name", "Integrations", optional=True),
    FieldSpec("wandb_mode", "W&B Mode", "select", "cfg-wandb-mode", "Integrations", options=("", "online", "offline", "disabled"), optional=True),
    FieldSpec("wandb_entity", "W&B Entity", "text", "cfg-wandb-entity", "Integrations", optional=True),
    FieldSpec("wandb_log_model", "W&B Log Model", "select", "cfg-wandb-log-model", "Integrations", options=("", "checkpoint", "end", "false"), optional=True),
    FieldSpec("use_tensorboard", "Use TensorBoard", "tri_bool", "cfg-use-tensorboard", "Integrations", optional=True),
    FieldSpec("use_mlflow", "Use MLflow", "tri_bool", "cfg-use-mlflow", "Integrations", optional=True),
    FieldSpec("mlflow_tracking_uri", "MLflow URI", "text", "cfg-mlflow-tracking-uri", "Integrations", optional=True),
    FieldSpec("mlflow_experiment_name", "MLflow Experiment", "text", "cfg-mlflow-experiment-name", "Integrations", optional=True),
    FieldSpec("use_comet", "Use Comet", "tri_bool", "cfg-use-comet", "Integrations", optional=True),
    FieldSpec("comet_project_name", "Comet Project", "text", "cfg-comet-project-name", "Integrations", optional=True),
    FieldSpec("use_otel_metrics", "OpenTelemetry Metrics", "tri_bool", "cfg-use-otel-metrics", "Integrations", optional=True),
    FieldSpec("otel_metrics_host", "OTEL Metrics Host", "text", "cfg-otel-metrics-host", "Integrations", "localhost", optional=True),
    FieldSpec("otel_metrics_port", "OTEL Metrics Port", "number", "cfg-otel-metrics-port", "Integrations", optional=True, step=1, minimum=1),
    FieldSpec("rl", "RL Mode", "select", "cfg-rl", "RL / Evaluation", options=("", "dpo", "ipo", "kto", "simpo", "orpo", "grpo", "ebft"), optional=True),
    FieldSpec("reward_model", "Reward Model", "tri_bool", "cfg-reward-model", "RL / Evaluation", optional=True),
    FieldSpec("process_reward_model", "Process Reward Model", "tri_bool", "cfg-process-reward-model", "RL / Evaluation", optional=True),
    FieldSpec("dpo_beta", "DPO Beta", "number", "cfg-dpo-beta", "RL / Evaluation", optional=True, step=0.01, minimum=0),
    FieldSpec("rl_beta", "RL Beta", "number", "cfg-rl-beta", "RL / Evaluation", optional=True, step=0.01, minimum=0),
    FieldSpec("trl.use_vllm", "TRL Use vLLM", "tri_bool", "cfg-trl-use-vllm", "RL / Evaluation", optional=True),
    FieldSpec("trl.vllm_mode", "TRL vLLM Mode", "select", "cfg-trl-vllm-mode", "RL / Evaluation", options=("", "server", "colocate"), optional=True),
    FieldSpec("trl.num_generations", "TRL Generations", "number", "cfg-trl-num-generations", "RL / Evaluation", optional=True, step=1, minimum=1),
    FieldSpec("vllm.gpu_memory_utilization", "vLLM GPU Memory", "number", "cfg-vllm-gpu-memory-utilization", "RL / Evaluation", optional=True, step=0.05, minimum=0, maximum=1),
    FieldSpec("vllm.tensor_parallel_size", "vLLM Tensor Parallel", "number", "cfg-vllm-tensor-parallel-size", "RL / Evaluation", optional=True, step=1, minimum=1),
    FieldSpec("lm_eval_model", "LM Eval Model", "text", "cfg-lm-eval-model", "RL / Evaluation", optional=True),
    FieldSpec("lm_eval_tasks", "LM Eval Tasks", "csv_list", "cfg-lm-eval-tasks", "RL / Evaluation", "arc_challenge, hellaswag", optional=True),
    FieldSpec("lm_eval_batch_size", "LM Eval Batch Size", "number", "cfg-lm-eval-batch-size", "RL / Evaluation", optional=True, step=1, minimum=1),
)

ADVANCED_FIELD_SPECS: tuple[FieldSpec, ...] = (
    _extra("seed", "Seed", "number", "Run Safety", step=1, minimum=0),
    _extra("resize_token_embeddings_to_32x", "Resize Embeddings To 32x", "tri_bool", "Run Safety"),
    _extra("mean_resizing_embeddings", "Mean Resize Embeddings", "tri_bool", "Run Safety"),
    _extra("shrink_embeddings", "Shrink Embeddings", "tri_bool", "Run Safety"),
    _extra("embeddings_skip_upcast", "Skip Embedding Upcast", "tri_bool", "Run Safety"),
    _extra("reinit_weights", "Reinitialize Weights", "tri_bool", "Run Safety"),
    _extra("trainer_cls", "Custom Trainer Class", "text", "Run Safety", "module.ClassName"),
    _extra("dynamic_checkpoint.enabled", "Dynamic Checkpoints", "tri_bool", "Run Safety"),
    _extra("dynamic_checkpoint.check_interval", "Dynamic Ckpt Interval", "number", "Run Safety", step=1, minimum=1),
    _extra("dynamic_checkpoint.trigger_file_path", "Dynamic Ckpt Trigger", "text", "Run Safety", "axolotl_checkpoint.save"),
    _extra("tokenizer_legacy", "Legacy Tokenizer", "tri_bool", "Model"),
    _extra("tokenizer_save_jinja_files", "Save Tokenizer Jinja", "tri_bool", "Model"),
    _extra("processor_kwargs", "Processor Kwargs", "json", "Model", "{image_seq_length: 576}"),
    _extra("special_tokens", "Special Tokens", "json", "Model", "{bos_token: <s>, eos_token: </s>}"),
    _extra("tokens", "Extra Tokens", "csv_list", "Model", "<think>, </think>"),
    _extra("added_tokens_overrides", "Added Token Overrides", "json", "Model", "{32000: <extra>}"),
    _extra("chat_template", "Top-level Chat Template", "select", "Model", options=CHAT_TEMPLATE_OPTIONS),
    _extra("chat_template_jinja", "Top-level Jinja", "text", "Model", "template text or file path"),
    _extra("chat_template_kwargs", "Chat Template Kwargs", "json", "Model", "{enable_thinking: false}"),
    _extra("eot_tokens", "EOT Tokens", "csv_list", "Model", "</s>, [/INST]"),
    _extra("default_system_message", "Default System Message", "text", "Model"),
    _extra("fix_untrained_tokens", "Fix Untrained Tokens", "json", "Model", "[32000, 32001]"),
    _extra("overrides_of_model_config", "Model Config Overrides", "json", "Model", "{rope_scaling: {...}}"),
    _extra("overrides_of_model_kwargs", "Model Load Overrides", "json", "Model", "{attn_implementation: sdpa}"),
    _extra("type_of_model", "Type Of Model", "text", "Model", "AutoModelForCausalLM"),
    _extra("rope_scaling", "RoPE Scaling", "json", "Model", "{type: linear, factor: 2.0}"),
    _extra("noisy_embedding_alpha", "Noisy Embedding Alpha", "number", "Model", step=0.1, minimum=0),
    _extra("use_kernels", "Use Custom Kernels", "tri_bool", "Model"),
    _extra("model_quantization_config", "Model Quant Config", "select", "Model", options=("", "Mxfp4Config", "FineGrainedFP8Config")),
    _extra("model_quantization_config_kwargs", "Model Quant Kwargs", "json", "Model", "{...}"),
    _extra("use_onebitllms", "Use 1.58-bit Kernels", "tri_bool", "Model"),
    _extra("image_size", "Image Size", "json", "Model", "1024 or [1024, 768]"),
    _extra("image_resize_algorithm", "Image Resize Algorithm", "select", "Model", options=("", "bilinear", "bicubic", "lanczos")),
    _extra("datasets.0.input_transform", "Input Transform", "text", "Dataset", "module.function"),
    _extra("datasets.0.shards", "Dataset Shards", "number", "Dataset", step=1, minimum=1),
    _extra("datasets.0.shards_idx", "Dataset Shard Index", "number", "Dataset", step=1, minimum=0),
    _extra("datasets.0.preprocess_shards", "Preprocess Shards", "number", "Dataset", step=1, minimum=1),
    _extra("datasets.0.conversation", "Conversation Format", "text", "Dataset"),
    _extra("datasets.0.input_format", "Input Format", "text", "Dataset"),
    _extra("datasets.0.field_human", "Human Field", "text", "Dataset", "human"),
    _extra("datasets.0.field_model", "Model Field", "text", "Dataset", "assistant"),
    _extra("datasets.0.field_tools", "Tools Field", "text", "Dataset", "tools"),
    _extra("datasets.0.field_thinking", "Thinking Field", "text", "Dataset", "reasoning_content"),
    _extra("datasets.0.template_thinking_key", "Template Thinking Key", "text", "Dataset"),
    _extra("datasets.0.message_field_role", "Message Role Field", "text", "Dataset", "role"),
    _extra("datasets.0.message_field_content", "Message Content Field", "text", "Dataset", "content"),
    _extra("datasets.0.message_property_mappings", "Message Property Map", "json", "Dataset", "{role: from, content: value}"),
    _extra("datasets.0.message_field_training", "Message Training Field", "text", "Dataset"),
    _extra("datasets.0.message_field_training_detail", "Training Detail Field", "text", "Dataset"),
    _extra("datasets.0.split_thinking", "Split Thinking Trace", "tri_bool", "Dataset"),
    _extra("datasets.0.logprobs_field", "Logprobs Field", "text", "Dataset"),
    _extra("datasets.0.temperature", "Dataset Temperature", "number", "Dataset", step=0.1, minimum=0),
    _extra("datasets.0.roles_to_train", "Roles To Train", "csv_list", "Dataset", "assistant"),
    _extra("datasets.0.train_on_eot", "Train On EOT", "select", "Dataset", options=("", "all", "turn", "last", "none")),
    _extra("datasets.0.roles", "Role Mapping", "json", "Dataset", "{user: [human, user], assistant: [gpt, assistant]}"),
    _extra("datasets.0.drop_system_message", "Drop System Message", "tri_bool", "Dataset"),
    _extra("datasets.0.revision", "Dataset Revision", "text", "Dataset", "branch, tag, or commit"),
    _extra("test_datasets", "Test Datasets", "json", "Dataset", "[{path: owner/eval, type: alpaca}]"),
    _extra("pretraining_dataset", "Pretraining Dataset", "json", "Dataset", "{path: owner/data, type: ...}"),
    _extra("dataloader_prefetch_factor", "Dataloader Prefetch", "number", "Dataset", step=1, minimum=1),
    _extra("dataloader_drop_last", "Dataloader Drop Last", "tri_bool", "Dataset"),
    _extra("accelerator_config", "Accelerator Config", "json", "Dataset", "{...}"),
    _extra("remove_unused_columns", "Remove Unused Columns", "tri_bool", "Dataset"),
    _extra("role_boundaries", "Role Boundaries", "json", "Dataset", "[{role: assistant, start: '<|assistant|>'}]"),
    _extra("sample_packing_mp_start_method", "Packing MP Start", "select", "Sequence / Packing", options=("", "fork", "spawn", "forkserver")),
    _extra("multipack_real_batches", "Multipack Real Batches", "tri_bool", "Sequence / Packing"),
    _extra("use_pose", "Use PoSE", "tri_bool", "Sequence / Packing"),
    _extra("pose_split_on_token_ids", "PoSE Split Token IDs", "csv_list", "Sequence / Packing", "13, 128009"),
    _extra("pose_max_context_len", "PoSE Max Context", "number", "Sequence / Packing", step=128, minimum=1),
    _extra("pose_num_chunks", "PoSE Chunks", "number", "Sequence / Packing", step=1, minimum=1),
    _extra("pretrain_multipack_buffer_size", "Pretrain Multipack Buffer", "number", "Sequence / Packing", step=1000, minimum=1),
    _extra("pretrain_multipack_attn", "Pretrain Multipack Attn", "tri_bool", "Sequence / Packing"),
    _extra("pretraining_sample_concatenation", "Pretrain Concatenate", "tri_bool", "Sequence / Packing"),
    _extra("streaming_multipack_buffer_size", "Streaming Multipack Buffer", "number", "Sequence / Packing", step=1000, minimum=1),
    _extra("max_packed_sequence_len", "Max Packed Sequence Len", "number", "Sequence / Packing", step=128, minimum=1),
    _extra("sample_packing_eff_est", "Packing Efficiency Est", "number", "Sequence / Packing", step=0.01, minimum=0, maximum=1),
    _extra("save_first_step", "Save First Step", "tri_bool", "Training"),
    _extra("hub_private_repo", "Hub Private Repo", "tri_bool", "Training"),
    _extra("push_to_hub", "Push To Hub", "tri_bool", "Training"),
    _extra("save_4bit", "Save 4-bit", "tri_bool", "Training"),
    _extra("saves_per_epoch", "Saves Per Epoch", "number", "Training", step=1, minimum=1),
    _extra("save_total_limit", "Save Total Limit", "number", "Training", step=1, minimum=1),
    _extra("train_on_inputs", "Train On Inputs", "tri_bool", "Optimizer"),
    _extra("group_by_length", "Group By Length", "tri_bool", "Optimizer"),
    _extra("embedding_lr_scale", "Embedding LR Scale", "number", "Optimizer", step=0.01, minimum=0),
    _extra("lr_quadratic_warmup", "Quadratic Warmup", "tri_bool", "Optimizer"),
    _extra("cosine_constant_lr_ratio", "Cosine Constant Ratio", "number", "Optimizer", step=0.01, minimum=0, maximum=1),
    _extra("lr_div_factor", "LR Div Factor", "number", "Optimizer", step=0.1, minimum=0),
    _extra("lr_groups", "LR Groups", "json", "Optimizer", "[{name: embed, modules: [embed_tokens], lr: 1e-5}]"),
    _extra("adam_epsilon2", "Adam Epsilon2", "number", "Optimizer", step=0.00000001, minimum=0),
    _extra("adam_beta3", "Adam Beta3", "number", "Optimizer", step=0.001, minimum=0, maximum=1),
    _extra("dion_lr", "Dion LR", "number", "Optimizer", step=0.00001, minimum=0),
    _extra("dion_momentum", "Dion Momentum", "number", "Optimizer", step=0.01, minimum=0, maximum=1),
    _extra("dion_rank_fraction", "Dion Rank Fraction", "number", "Optimizer", step=0.01, minimum=0),
    _extra("dion_rank_multiple_of", "Dion Rank Multiple", "number", "Optimizer", step=1, minimum=1),
    _extra("qgalore_rank", "Q-GaLore Rank", "number", "Optimizer", step=1, minimum=1),
    _extra("qgalore_update_proj_gap", "Q-GaLore Update Gap", "number", "Optimizer", step=1, minimum=1),
    _extra("qgalore_scale", "Q-GaLore Scale", "number", "Optimizer", step=0.01, minimum=0),
    _extra("qgalore_proj_type", "Q-GaLore Proj Type", "select", "Optimizer", options=("", "std", "reverse_std", "right", "left", "full")),
    _extra("qgalore_proj_quant", "Q-GaLore Proj Quant", "tri_bool", "Optimizer"),
    _extra("qgalore_proj_bits", "Q-GaLore Proj Bits", "number", "Optimizer", step=1, minimum=1),
    _extra("qgalore_proj_group_size", "Q-GaLore Group Size", "number", "Optimizer", step=1, minimum=1),
    _extra("qgalore_cos_threshold", "Q-GaLore Cos Threshold", "number", "Optimizer", step=0.01, minimum=0, maximum=1),
    _extra("qgalore_gamma_proj", "Q-GaLore Gamma", "number", "Optimizer", step=1, minimum=1),
    _extra("qgalore_queue_size", "Q-GaLore Queue", "number", "Optimizer", step=1, minimum=1),
    _extra("lora_fan_in_fan_out", "LoRA Fan In/Out", "tri_bool", "Adapter / PEFT"),
    _extra("lora_target_parameters", "LoRA Target Parameters", "csv_list", "Adapter / PEFT", "feed_forward.experts.*"),
    _extra("peft_layers_to_transform", "PEFT Layers", "csv_list", "Adapter / PEFT", "0, 1, 2"),
    _extra("peft_layers_pattern", "PEFT Layer Pattern", "csv_list", "Adapter / PEFT", "layers"),
    _extra("peft", "PEFT Config", "json", "Adapter / PEFT", "{loftq_config: {loftq_bits: 4}}"),
    _extra("peft_layer_replication", "PEFT Layer Replication", "json", "Adapter / PEFT", "[[0, 4], [2, 5]]"),
    _extra("peft_init_lora_weights", "PEFT Init LoRA Weights", "text", "Adapter / PEFT", "true, gaussian, loftq, ..."),
    _extra("peft_trainable_token_indices", "PEFT Trainable Token IDs", "json", "Adapter / PEFT", "[32000, 32001]"),
    _extra("peft_ensure_weight_tying", "PEFT Tie Weights", "tri_bool", "Adapter / PEFT"),
    _extra("peft_autocast_adapter_dtype", "PEFT Autocast Adapter", "tri_bool", "Adapter / PEFT"),
    _extra("loraplus_lr_embedding", "LoRA+ Embedding LR", "number", "Adapter / PEFT", step=0.000001, minimum=0),
    _extra("relora", "ReLoRA", "tri_bool", "Adapter / PEFT"),
    _extra("relora_prune_ratio", "ReLoRA Prune Ratio", "number", "Adapter / PEFT", step=0.01, minimum=0, maximum=1),
    _extra("relora_prune_method", "ReLoRA Prune Method", "select", "Adapter / PEFT", options=("", "magnitude", "random", "reset")),
    _extra("relora_cpu_offload", "ReLoRA CPU Offload", "tri_bool", "Adapter / PEFT"),
    _extra("jagged_restart_steps", "Jagged Restart Steps", "number", "Adapter / PEFT", step=1, minimum=1),
    _extra("jagged_restart_warmup_steps", "Jagged Warmup Steps", "number", "Adapter / PEFT", step=1, minimum=0),
    _extra("jagged_restart_anneal_steps", "Jagged Anneal Steps", "number", "Adapter / PEFT", step=1, minimum=0),
    _extra("gradient_checkpointing_kwargs", "Gradient Ckpt Kwargs", "json", "Precision / Memory", "{use_reentrant: false}"),
    _extra("selective_checkpointing", "Selective Checkpointing", "json", "Precision / Memory", "true or {save: [attention]}"),
    _extra("layer_offloading", "Layer Offloading", "tri_bool", "Precision / Memory"),
    _extra("freeze_mm_modules", "Freeze MM Modules", "tri_bool", "Precision / Memory"),
    _extra("unfrozen_parameters", "Unfrozen Parameters", "csv_list", "Precision / Memory", "lm_head.*, embed_tokens.*"),
    _extra("fp8_enable_fsdp_float8_all_gather", "FP8 FSDP All-gather", "tri_bool", "Precision / Memory"),
    _extra("bfloat16", "No-AMP BF16", "tri_bool", "Precision / Memory"),
    _extra("float16", "No-AMP FP16", "tri_bool", "Precision / Memory"),
    _extra("float32", "Force FP32", "tri_bool", "Precision / Memory"),
    _extra("gc_steps", "GC Steps Deprecated", "number", "Precision / Memory", step=1, minimum=1),
    _extra("eager_attention", "Eager Attention Legacy", "tri_bool", "Attention / Kernels"),
    _extra("gemma4_hybrid_attn_impl", "Gemma 4 Hybrid Attn", "tri_bool", "Attention / Kernels"),
    _extra("large_head_attention", "Large-head Attention", "select", "Attention / Kernels", options=("", "sdpa", "auto", "triton_flash")),
    _extra("flash_attn_d512", "FlashAttn D512 Legacy", "tri_bool", "Attention / Kernels"),
    _extra("sdpa_varlen", "SDPA Varlen", "tri_bool", "Attention / Kernels"),
    _extra("fused_attn_kernel", "Fused Attn Kernel", "tri_bool", "Attention / Kernels"),
    _extra("flex_attn_compile_kwargs", "Flex Compile Kwargs", "json", "Attention / Kernels", "{...}"),
    _extra("experts_implementation", "Experts Implementation", "text", "Attention / Kernels"),
    _extra("quantize_moe_experts", "Quantize MoE Experts", "tri_bool", "Attention / Kernels"),
    _extra("scaling_softmax", "Scaling Softmax", "tri_bool", "Attention / Kernels"),
    _extra("scaling_softmax_factor", "SSMax Factor", "number", "Attention / Kernels", step=0.01),
    _extra("scaling_softmax_bias", "SSMax Bias", "number", "Attention / Kernels", step=0.01),
    _extra("lora_mlp_kernel", "LoRA MLP Kernel", "tri_bool", "Attention / Kernels"),
    _extra("lora_qkv_kernel", "LoRA QKV Kernel", "tri_bool", "Attention / Kernels"),
    _extra("lora_o_kernel", "LoRA O Kernel", "tri_bool", "Attention / Kernels"),
    _extra("lora_embedding_kernel", "LoRA Embedding Kernel", "tri_bool", "Attention / Kernels"),
    _extra("chunked_cross_entropy", "Chunked Cross Entropy", "tri_bool", "Attention / Kernels"),
    _extra("chunked_cross_entropy_num_chunks", "CE Chunks", "number", "Attention / Kernels", step=1, minimum=1),
    _extra("use_eaft", "Use EAFT Loss", "tri_bool", "Attention / Kernels"),
    _extra("eaft_alpha", "EAFT Alpha", "number", "Attention / Kernels", step=0.1, minimum=0),
    _extra("eaft_k", "EAFT Top-k", "number", "Attention / Kernels", step=1, minimum=1),
    _extra("tiled_mlp", "Tiled MLP", "tri_bool", "Attention / Kernels"),
    _extra("tiled_mlp_num_shards", "Tiled MLP Shards", "number", "Attention / Kernels", step=1, minimum=1),
    _extra("tiled_mlp_use_original_mlp", "Tiled MLP Original", "tri_bool", "Attention / Kernels"),
    _extra("llama4_linearized_experts", "Llama4 Linearized Experts", "tri_bool", "Attention / Kernels"),
    _extra("device", "Device", "text", "Distributed", "cuda, cpu, auto"),
    _extra("device_map", "Device Map", "json", "Distributed", "auto or {\"\": 0}"),
    _extra("world_size", "World Size", "number", "Distributed", step=1, minimum=1),
    _extra("local_rank", "Local Rank", "number", "Distributed", step=1, minimum=0),
    _extra("ddp_timeout", "DDP Timeout", "number", "Distributed", step=1, minimum=1),
    _extra("ddp_bucket_cap_mb", "DDP Bucket MB", "number", "Distributed", step=1, minimum=1),
    _extra("ddp_broadcast_buffers", "DDP Broadcast Buffers", "tri_bool", "Distributed"),
    _extra("fsdp_config.sync_module_states", "FSDP Sync Module States", "tri_bool", "Distributed"),
    _extra("fsdp_config.final_state_dict_type", "FSDP Final State Dict", "select", "Distributed", options=("", "FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT")),
    _extra("fsdp_config.auto_wrap_policy", "FSDP Auto Wrap", "select", "Distributed", options=("", "TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP")),
    _extra("fsdp_config.transformer_layer_cls_to_wrap", "FSDP Layer Class", "text", "Distributed", "LlamaDecoderLayer"),
    _extra("fsdp_config.min_num_params", "FSDP Min Params", "number", "Distributed", step=1000000, minimum=1),
    _extra("fsdp_config.reshard_after_forward", "FSDP Reshard After Fwd", "tri_bool", "Distributed"),
    _extra("fsdp_config.mixed_precision_policy", "FSDP Mixed Precision", "text", "Distributed", "fp16 or bf16"),
    _extra("fp32_norms", "FSDP FP32 Norms", "tri_bool", "Distributed"),
    _extra("fp32_norm_classes", "FP32 Norm Classes", "csv_list", "Distributed", "RMSNorm, LayerNorm"),
    _extra("fsdp_final_state_dict_type", "FSDP Final State Dict", "select", "Distributed", options=("", "FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT")),
    _extra("dp_shard_size", "DP Shard Size", "number", "Distributed", step=1, minimum=1),
    _extra("dp_replicate_size", "DP Replicate Size", "number", "Distributed", step=1, minimum=1),
    _extra("sequence_parallel_degree", "Sequence Parallel Degree", "number", "Distributed", step=1, minimum=1),
    _extra("heads_k_stride", "Heads K Stride", "number", "Distributed", step=1, minimum=1),
    _extra("ring_attn_func", "Ring Attention Func", "select", "Distributed", options=("", "varlen_llama3", "batch_ring", "batch_zigzag", "batch_stripe")),
    _extra("use_ray", "Use Ray", "tri_bool", "Distributed"),
    _extra("ray_run_name", "Ray Run Name", "text", "Distributed"),
    _extra("ray_num_workers", "Ray Workers", "number", "Distributed", step=1, minimum=1),
    _extra("resources_per_worker", "Resources Per Worker", "json", "Distributed", "{GPU: 1}"),
    _extra("do_causal_lm_eval", "Causal LM Eval", "tri_bool", "Tracking"),
    _extra("eval_causal_lm_metrics", "Causal LM Metrics", "csv_list", "Tracking", "sacrebleu, perplexity"),
    _extra("do_bench_eval", "Benchmark Eval", "tri_bool", "Tracking"),
    _extra("bench_dataset", "Bench Dataset", "text", "Tracking"),
    _extra("bench_split", "Bench Split", "text", "Tracking"),
    _extra("profiler_steps", "Profiler Steps", "number", "Tracking", step=1, minimum=1),
    _extra("profiler_steps_start", "Profiler Start Step", "number", "Tracking", step=1, minimum=0),
    _extra("include_tokens_per_second", "Tokens/sec Summary", "tri_bool", "Tracking"),
    _extra("generation_top_k", "Generation Top K", "number", "Tracking", step=1, minimum=1),
    _extra("generation_prompt_ratio", "Generation Prompt Ratio", "number", "Tracking", step=0.05, minimum=0, maximum=1),
    _extra("generation_do_sample", "Generation Sampling", "tri_bool", "Tracking"),
    _extra("wandb_run_id", "W&B Run ID", "text", "Integrations"),
    _extra("wandb_watch", "W&B Watch", "text", "Integrations"),
    _extra("mlflow_run_name", "MLflow Run Name", "text", "Integrations"),
    _extra("hf_mlflow_log_artifacts", "MLflow Log Artifacts", "tri_bool", "Integrations"),
    _extra("comet_api_key", "Comet API Key", "text", "Integrations"),
    _extra("comet_workspace", "Comet Workspace", "text", "Integrations"),
    _extra("comet_experiment_key", "Comet Experiment Key", "text", "Integrations"),
    _extra("comet_mode", "Comet Mode", "select", "Integrations", options=("", "create", "get", "get_or_create")),
    _extra("comet_online", "Comet Online", "tri_bool", "Integrations"),
    _extra("comet_experiment_config", "Comet Config", "json", "Integrations", "{...}"),
    _extra("use_trackio", "Use Trackio", "tri_bool", "Integrations"),
    _extra("trackio_project_name", "Trackio Project", "text", "Integrations"),
    _extra("trackio_run_name", "Trackio Run", "text", "Integrations"),
    _extra("trackio_space_id", "Trackio Space", "text", "Integrations", "owner/space"),
    _extra("plugins", "Axolotl Plugins", "csv_list", "Integrations", "axolotl.integrations.cut_cross_entropy"),
    _extra("trl.beta", "TRL Beta", "number", "RL / Evaluation", step=0.01, minimum=0),
    _extra("trl.max_completion_length", "TRL Max Completion", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.vllm_server_host", "vLLM Server Host", "text", "RL / Evaluation", "0.0.0.0"),
    _extra("trl.vllm_server_port", "vLLM Server Port", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.vllm_server_timeout", "vLLM Timeout", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.vllm_guided_decoding_regex", "vLLM Guided Regex", "text", "RL / Evaluation"),
    _extra("trl.reward_funcs", "Reward Functions", "csv_list", "RL / Evaluation", "rewards.length_reward"),
    _extra("trl.reward_weights", "Reward Weights", "csv_list", "RL / Evaluation", "1.0, 0.5"),
    _extra("trl.generation_batch_size", "Generation Batch Size", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.log_completions", "Log Completions", "tri_bool", "RL / Evaluation"),
    _extra("trl.num_completions_to_print", "Completions To Print", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.importance_sampling_level", "Importance Sampling", "select", "RL / Evaluation", options=("", "sequence", "token")),
    _extra("trl.sync_ref_model", "Sync Ref Model", "tri_bool", "RL / Evaluation"),
    _extra("trl.ref_model_mixup_alpha", "Ref Mixup Alpha", "number", "RL / Evaluation", step=0.01, minimum=0),
    _extra("trl.ref_model_sync_steps", "Ref Sync Steps", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.scale_rewards", "Scale Rewards", "tri_bool", "RL / Evaluation"),
    _extra("trl.temperature", "Policy Temperature", "number", "RL / Evaluation", step=0.1, minimum=0),
    _extra("trl.top_p", "Policy Top P", "number", "RL / Evaluation", step=0.05, minimum=0, maximum=1),
    _extra("trl.top_k", "Policy Top K", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.min_p", "Policy Min P", "number", "RL / Evaluation", step=0.01, minimum=0, maximum=1),
    _extra("trl.repetition_penalty", "Repetition Penalty", "number", "RL / Evaluation", step=0.05, minimum=0),
    _extra("trl.generation_kwargs", "Generation Kwargs", "json", "RL / Evaluation", "{seed: 42}"),
    _extra("trl.chat_template_kwargs", "TRL Chat Kwargs", "json", "RL / Evaluation", "{enable_thinking: false}"),
    _extra("trl.num_iterations", "GRPO Iterations", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.epsilon", "GRPO Epsilon", "number", "RL / Evaluation", step=0.01, minimum=0),
    _extra("trl.epsilon_high", "GRPO Epsilon High", "number", "RL / Evaluation", step=0.01, minimum=0),
    _extra("trl.use_liger_loss", "Use Liger Loss", "tri_bool", "RL / Evaluation"),
    _extra("trl.loss_type", "TRL Loss Type", "select", "RL / Evaluation", options=("", "grpo", "bnpo", "dr_grpo")),
    _extra("trl.mask_truncated_completions", "Mask Truncated", "tri_bool", "RL / Evaluation"),
    _extra("trl.vllm_enable_sleep_mode", "vLLM Sleep Mode", "tri_bool", "RL / Evaluation"),
    _extra("trl.rollout_func", "Rollout Function", "text", "RL / Evaluation", "module.function"),
    _extra("trl.multi_objective_aggregation", "Reward Aggregation", "select", "RL / Evaluation", options=("", "sum_then_normalize", "normalize_then_sum")),
    _extra("trl.use_data_producer", "Use Data Producer", "tri_bool", "RL / Evaluation"),
    _extra("trl.async_prefetch", "Async Prefetch", "tri_bool", "RL / Evaluation"),
    _extra("trl.prefetch_depth", "Prefetch Depth", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.vllm_sync_interval", "vLLM Sync Interval", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.streaming_partial_batch", "Streaming Partial Batch", "tri_bool", "RL / Evaluation"),
    _extra("trl.streaming_min_groups", "Streaming Min Groups", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.vllm_importance_sampling_correction", "vLLM IS Correction", "tri_bool", "RL / Evaluation"),
    _extra("trl.vllm_importance_sampling_mode", "vLLM IS Mode", "select", "RL / Evaluation", options=("", "token_truncate", "token_mask", "sequence_truncate", "sequence_mask")),
    _extra("trl.vllm_importance_sampling_cap", "vLLM IS Cap", "number", "RL / Evaluation", step=0.1, minimum=0),
    _extra("trl.off_policy_mask_threshold", "Off-policy Mask Threshold", "number", "RL / Evaluation", step=0.01, minimum=0),
    _extra("trl.use_bias_correction_kl", "Bias-correct KL", "tri_bool", "RL / Evaluation"),
    _extra("trl.reward_num_workers", "Reward Workers", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.replay_buffer_size", "Replay Buffer Size", "number", "RL / Evaluation", step=1, minimum=0),
    _extra("trl.replay_recompute_logps", "Replay Recompute Logps", "tri_bool", "RL / Evaluation"),
    _extra("trl.reroll_start_fraction", "Reroll Start Fraction", "number", "RL / Evaluation", step=0.01, minimum=0, maximum=1),
    _extra("trl.reroll_max_groups", "Reroll Max Groups", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("trl.skip_zero_advantage_batches", "Skip Zero Advantage", "tri_bool", "RL / Evaluation"),
    _extra("trl.vllm_lora_sync", "vLLM LoRA Sync", "tri_bool", "RL / Evaluation"),
    _extra("vllm.device", "vLLM Device", "text", "RL / Evaluation", "auto"),
    _extra("vllm.data_parallel_size", "vLLM Data Parallel", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("vllm.dtype", "vLLM DType", "text", "RL / Evaluation", "auto, bfloat16, float16"),
    _extra("vllm.max_model_len", "vLLM Max Context", "number", "RL / Evaluation", step=128, minimum=1),
    _extra("vllm.enable_prefix_caching", "vLLM Prefix Cache", "tri_bool", "RL / Evaluation"),
    _extra("vllm.host", "vLLM Host", "text", "RL / Evaluation", "0.0.0.0"),
    _extra("vllm.port", "vLLM Port", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("vllm.enable_reasoning", "vLLM Reasoning", "tri_bool", "RL / Evaluation"),
    _extra("vllm.reasoning_parser", "Reasoning Parser", "text", "RL / Evaluation"),
    _extra("vllm.enforce_eager", "vLLM Enforce Eager", "tri_bool", "RL / Evaluation"),
    _extra("vllm.serve_module", "vLLM Serve Module", "text", "RL / Evaluation"),
    _extra("vllm.worker_extension_cls", "vLLM Worker Extension", "text", "RL / Evaluation"),
    _extra("ebft", "EBFT Config", "json", "RL / Evaluation", "{feature_layers: [0.25, 0.5, 0.75]}"),
    _extra("qat", "QAT Config", "json", "RL / Evaluation", "{weight_dtype: int8, group_size: 32}"),
    _extra("quantization", "PTQ Quantization", "json", "RL / Evaluation", "{weight_dtype: int8, group_size: 32}"),
    _extra("center_rewards_coefficient", "Center Rewards Coeff", "number", "RL / Evaluation", step=0.001, minimum=0),
    _extra("num_labels", "Reward Num Labels", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("dpo_use_weighting", "DPO Weighting", "tri_bool", "RL / Evaluation"),
    _extra("dpo_label_smoothing", "DPO Label Smoothing", "number", "RL / Evaluation", step=0.01, minimum=0, maximum=1),
    _extra("precompute_ref_log_probs", "Precompute Ref Logprobs", "tri_bool", "RL / Evaluation"),
    _extra("dpo_use_liger_kernel", "DPO Liger Kernel", "tri_bool", "RL / Evaluation"),
    _extra("dpo_padding_free", "DPO Padding Free", "tri_bool", "RL / Evaluation"),
    _extra("dpo_loss_type", "DPO Loss Types", "csv_list", "RL / Evaluation", "sigmoid, ipo"),
    _extra("dpo_loss_weights", "DPO Loss Weights", "csv_list", "RL / Evaluation", "1.0, 0.5"),
    _extra("orpo_alpha", "ORPO Alpha", "number", "RL / Evaluation", step=0.01, minimum=0),
    _extra("simpo_gamma", "SimPO Gamma", "number", "RL / Evaluation", step=0.01),
    _extra("cpo_alpha", "CPO Alpha", "number", "RL / Evaluation", step=0.01, minimum=0),
    _extra("kto_desirable_weight", "KTO Desirable Weight", "number", "RL / Evaluation", step=0.01, minimum=0),
    _extra("kto_undesirable_weight", "KTO Undesirable Weight", "number", "RL / Evaluation", step=0.01, minimum=0),
    _extra("eval_table_size", "Eval Table Size", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("eval_max_new_tokens", "Eval Max New Tokens", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("dpo_use_logits_to_keep", "DPO Logits To Keep", "tri_bool", "RL / Evaluation"),
    _extra("dpo_generate_during_eval", "DPO Generate During Eval", "tri_bool", "RL / Evaluation"),
    _extra("dpo_norm_loss", "DPO Normalize Loss", "tri_bool", "RL / Evaluation"),
    _extra("rpo_alpha", "RPO Alpha", "number", "RL / Evaluation", step=0.01, minimum=0),
    _extra("gradio_title", "Gradio Title", "text", "RL / Evaluation"),
    _extra("gradio_share", "Gradio Share", "tri_bool", "RL / Evaluation"),
    _extra("gradio_server_name", "Gradio Server", "text", "RL / Evaluation"),
    _extra("gradio_server_port", "Gradio Port", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("gradio_max_new_tokens", "Gradio Max Tokens", "number", "RL / Evaluation", step=1, minimum=1),
    _extra("gradio_temperature", "Gradio Temperature", "number", "RL / Evaluation", step=0.1, minimum=0),
)

_BASE_FIELD_KEYS = {spec.key for spec in FIELD_SPECS}
FIELD_SPECS = (*FIELD_SPECS, *(spec for spec in ADVANCED_FIELD_SPECS if spec.key not in _BASE_FIELD_KEYS))


class ConfigError(RuntimeError):
    """Raised for configuration file errors."""


class ConfigStore:
    """Owns local Axolotl config files and maps them to LCARS editor fields."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.config_dir = project_root / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.active_name = DEFAULT_CONFIG_NAME
        self.ensure_default_config()

    @property
    def active_path(self) -> Path:
        return self.config_dir / self.active_name

    def ensure_default_config(self) -> None:
        path = self.config_dir / self.active_name
        if path.exists():
            return
        bundled = Path(__file__).resolve().parents[2] / "configs" / self.active_name
        if bundled.exists() and bundled != path:
            shutil.copyfile(bundled, path)
            return
        path.write_text("base_model: NousResearch/Llama-3.2-1B\noutput_dir: ./outputs/lora-out\n", encoding="utf-8")

    def list_configs(self) -> list[str]:
        names = sorted(p.name for p in self.config_dir.glob("*.y*ml") if p.is_file())
        if self.active_name not in names and names:
            self.active_name = names[0]
        return names or [self.active_name]

    def set_active(self, name: str) -> None:
        if name not in self.list_configs():
            raise ConfigError(f"Config does not exist: {name}")
        self.active_name = name

    def load(self) -> dict[str, Any]:
        try:
            raw = self.active_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"Unable to read {self.active_path}: {exc}") from exc
        try:
            payload = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML parse failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConfigError("Axolotl config must be a YAML mapping at the top level.")
        return payload

    def save(self, cfg: dict[str, Any]) -> None:
        text = yaml.safe_dump(cfg, sort_keys=False, width=100)
        self.active_path.write_text(text, encoding="utf-8")

    def validate_text(self, text: str) -> dict[str, Any]:
        try:
            payload = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML parse failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConfigError("Axolotl config must be a YAML mapping at the top level.")
        return payload

    def save_raw_text(self, text: str) -> None:
        self.validate_text(text)
        self.active_path.write_text(text, encoding="utf-8")

    def create_copy(self, name: str) -> str:
        clean = Path(name).name
        if not clean.endswith((".yml", ".yaml")):
            clean = f"{clean}.yml"
        target = self.config_dir / clean
        if target.exists():
            raise ConfigError(f"Config already exists: {clean}")
        shutil.copyfile(self.active_path, target)
        self.active_name = clean
        return clean

    def create_named(self, name: str) -> str:
        clean = Path(name).name
        if not clean:
            raise ConfigError("Config name is required.")
        if not clean.endswith((".yml", ".yaml")):
            clean = f"{clean}.yml"
        target = self.config_dir / clean
        if target.exists():
            raise ConfigError(f"Config already exists: {clean}")
        target.write_text(
            yaml.safe_dump(_starter_config(), sort_keys=False, width=100),
            encoding="utf-8",
        )
        self.active_name = clean
        return clean

    def field_value(self, spec: FieldSpec, cfg: dict[str, Any] | None = None) -> Any:
        payload = cfg if cfg is not None else self.load()
        value = self._get_path(payload, spec.key)
        if value is None:
            return spec.default if spec.default is not None else ""
        if spec.kind == "bool":
            return bool(value)
        if spec.kind == "tri_bool":
            if isinstance(value, bool):
                return "true" if value else "false"
            return "unset"
        if spec.kind == "csv_list":
            if isinstance(value, list):
                return ", ".join(str(item) for item in value)
            return str(value)
        if spec.kind == "json":
            if isinstance(value, str):
                return value
            return yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
        if spec.key in BOOL_STRING_SELECT_KEYS and isinstance(value, bool):
            return "true" if value else "false"
        return value

    def editor_values(self) -> dict[str, Any]:
        cfg = self.load()
        return {spec.widget_id: self.field_value(spec, cfg) for spec in FIELD_SPECS}

    def save_editor_values(self, values: dict[str, Any]) -> None:
        cfg = self.load()
        for spec in FIELD_SPECS:
            raw = values.get(spec.widget_id)
            self._set_path(cfg, spec.key, self._coerce(spec, raw), optional=spec.optional)
        self.save(cfg)

    def apply_model(self, model_ref: str) -> None:
        cfg = self.load()
        cfg["base_model"] = model_ref
        self.save(cfg)

    def apply_dataset(self, dataset_ref: str, dataset_type: str = "alpaca") -> None:
        cfg = self.load()
        datasets = cfg.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            datasets = [{}]
            cfg["datasets"] = datasets
        if not isinstance(datasets[0], dict):
            datasets[0] = {}
        datasets[0]["path"] = dataset_ref
        datasets[0].setdefault("type", dataset_type or "alpaca")
        self.save(cfg)

    def summary_rows(self) -> list[dict[str, str]]:
        cfg = self.load()
        rows = []
        for key in (
            "base_model",
            "adapter",
            "output_dir",
            "sequence_len",
            "sample_packing",
            "load_in_8bit",
            "load_in_4bit",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "optimizer",
            "attn_implementation",
            "deepspeed",
            "fsdp",
        ):
            rows.append({"Key": key, "Value": str(cfg.get(key, ""))})
        datasets = cfg.get("datasets")
        if isinstance(datasets, list) and datasets:
            first = datasets[0] if isinstance(datasets[0], dict) else {}
            rows.append({"Key": "dataset", "Value": str(first.get("path", ""))})
            rows.append({"Key": "dataset_type", "Value": str(first.get("type", ""))})
        return rows

    def _get_path(self, cfg: dict[str, Any], dotted: str) -> Any:
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

    def _set_path(self, cfg: dict[str, Any], dotted: str, value: Any, *, optional: bool) -> None:
        if optional and (value is None or value == ""):
            self._delete_path(cfg, dotted)
            return

        parts = dotted.split(".")
        node: Any = cfg
        for index, part in enumerate(parts[:-1]):
            next_part = parts[index + 1]
            if isinstance(node, list):
                item_index = int(part)
                while len(node) <= item_index:
                    node.append({} if not next_part.isdigit() else [])
                node = node[item_index]
            else:
                if part not in node or node[part] is None:
                    node[part] = [] if next_part.isdigit() else {}
                node = node[part]

        last = parts[-1]
        if isinstance(node, list):
            item_index = int(last)
            while len(node) <= item_index:
                node.append(None)
            node[item_index] = value
        else:
            node[last] = value

    def _delete_path(self, cfg: dict[str, Any], dotted: str) -> None:
        parts = dotted.split(".")
        node: Any = cfg
        for part in parts[:-1]:
            if isinstance(node, list):
                try:
                    node = node[int(part)]
                except (ValueError, IndexError):
                    return
            elif isinstance(node, dict):
                node = node.get(part)
            else:
                return
        if isinstance(node, dict):
            node.pop(parts[-1], None)

    def _coerce(self, spec: FieldSpec, value: Any) -> Any:
        if spec.kind == "bool":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
        if spec.kind == "tri_bool":
            text = "" if value is None else str(value).strip().lower()
            if text in {"", "unset", "none", "null"}:
                return None
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
            raise ConfigError(f"{spec.label} must be unset, true, or false.")
        if spec.kind == "number":
            if value in (None, ""):
                return None if spec.optional else spec.default
            number = float(value)
            if spec.minimum is not None and number < spec.minimum:
                raise ConfigError(f"{spec.label} must be >= {spec.minimum}.")
            if spec.maximum is not None and number > spec.maximum:
                raise ConfigError(f"{spec.label} must be <= {spec.maximum}.")
            if number.is_integer() and spec.step >= 1:
                return int(number)
            return number
        if spec.kind == "csv_list":
            text = "" if value is None else str(value).strip()
            if not text:
                return None
            parts = [part.strip() for part in text.split(",") if part.strip()]
            if spec.key == "optim_target_modules" and parts == ["all_linear"]:
                return "all_linear"
            return parts
        if spec.kind == "json":
            text = "" if value is None else str(value).strip()
            if not text:
                return None
            try:
                return yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise ConfigError(f"{spec.label} is not valid YAML/JSON: {exc}") from exc
        text = "" if value is None else str(value).strip()
        if spec.optional and text == "":
            return None
        if spec.key in BOOL_STRING_SELECT_KEYS:
            if text.lower() == "true":
                return True
            if text.lower() == "false":
                return False
            return text or None
        return text


def _starter_config() -> dict[str, Any]:
    return {
        "base_model": "NousResearch/Llama-3.2-1B",
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "adapter": "lora",
        "load_in_8bit": True,
        "load_in_4bit": False,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "datasets": [{"path": "teknium/GPT4-LLM-Cleaned", "type": "alpaca"}],
        "dataset_prepared_path": "last_run_prepared",
        "val_set_size": 0.1,
        "sequence_len": 2048,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        "output_dir": "./outputs/lora-out",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_epochs": 3,
        "learning_rate": 0.0001,
        "optimizer": "adamw_bnb_8bit",
        "lr_scheduler": "cosine",
        "warmup_steps": 10,
        "bf16": "auto",
        "fp16": False,
        "gradient_checkpointing": True,
        "attn_implementation": "flash_attention_2",
        "logging_steps": 1,
        "save_steps": 100,
        "eval_steps": 100,
        "strict": False,
    }

# Axolotl Guardrails

The UI is intentionally strict about launch readiness because Axolotl runs can consume substantial GPU time and disk space.

## Model Format

Axolotl expects a Hugging Face model id or a local Transformers-style model directory with files such as:

- `config.json`
- tokenizer files
- `.safetensors`, `.bin`, or `.pt` weights

GGUF files and Ollama runtime blobs are blocked as `base_model` values because they are not directly trainable by Axolotl.

The HF Hub page classifies search results before use:

- `base_model`: Transformers config plus `.safetensors`, `.bin`, or `.pt` weights.
- `peft_adapter`: adapter files that should be applied as `lora_model_dir` with a matching base model.
- `runtime_quant`: GGUF/Ollama/other runtime artifacts that are blocked as `base_model`.

Downloads are filtered to compatible model or dataset file extensions so GGUF-only search results do not silently become training inputs.

## Dataset Shape

The preflight checks warn when dataset type and fields look incomplete, such as:

- `completion` datasets without a text field.
- chat datasets without message fields or chat template settings.
- local dataset files without `ds_type`.

## Training Safety

The validator checks for:

- mutually exclusive `load_in_8bit` and `load_in_4bit`.
- suspicious QLoRA, LoRA, GPTQ, and target-module combinations.
- incompatible precision settings.
- conflicting attention backend controls.
- DeepSpeed and FSDP being enabled together.
- checkpoint and resume combinations that may prevent clean recovery.

## Workflow and Runner Commands

The Workflow page exposes these config commands as typed, connectable stages:

- `preprocess`
- `train`
- `inference`
- `merge-lora`
- `merge-sharded-fsdp-weights`
- `evaluate`
- `lm-eval`
- `quantize`

It also supports utility actions:

- `fetch`
- `delinearize-llama4`

Launcher arguments for `python`, `accelerate`, or `torchrun` are separated from Axolotl command arguments and passed after `--`.

Workflow launch always reruns preflight first. A workflow must contain exactly one Active Config
source and one continuous, unbranched chain of action stages. Stages run sequentially through the
single-process runner; a failure or cancellation prevents later stages from starting. Utility
actions remain available through the **Console** page's **Single Action Override** because they do
not consume the active config like workflow stages do.

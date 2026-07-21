# Axolotl Guardrails

The UI is intentionally strict about launch readiness because Axolotl runs can consume substantial GPU time and disk space.

## Model Format

Axolotl expects a Hugging Face model id or a local Transformers-style model directory with files such as:

- `config.json`
- tokenizer files
- `.safetensors`, `.bin`, or `.pt` weights

GGUF files and Ollama runtime blobs are blocked as `base_model` values because they are not directly trainable by Axolotl.

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

## Runner Commands

The run page supports Axolotl config commands:

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

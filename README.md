# Axolotl LCARS UI

LCARS frontend and manager for Axolotl config editing, run monitoring, local resource telemetry,
Hugging Face content management, and Ollama model detection.

Built with [LCARS WebUI](https://github.com/darsrc/LCARS-WebUI).

## Launch

The project is intended to run from a Python virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For local `lcars-ui` development, install your own checkout into the venv in editable mode before
launching the app.

Start the app:

```bash
./launch.sh
```

Useful launcher args:

```bash
./launch.sh help
./launch.sh port 8080
./launch.sh ip 0.0.0.0 port 8080
./launch.sh --ip 0.0.0.0 --port 8080 --open
```

The launcher traps `Ctrl+C` and asks the Python server to terminate cleanly.

## Current Features

- Structured Axolotl config editor split into Setup, Train, Hardware, and Tracking pages.
- 484 surfaced Axolotl config keys, including advanced dataset, tokenizer, PEFT, optimizer,
  kernel, FSDP, DeepSpeed, TRL, vLLM, evaluation, and integration settings.
- Raw YAML editor at `/raw`.
- Preflight gate that blocks or warns on model formats, local file paths, dataset shape,
  quantization/adapters, precision conflicts, attention backends, distributed settings,
  checkpoint/resume hazards, hub auth, and tracking integrations.
- Axolotl subprocess start/stop and live log viewer for preprocess, train, inference, merge,
  evaluate, lm-eval, and quantize commands.
- Separate Axolotl action args and launcher args. Launcher args are placed after `--`, matching
  Axolotl's launcher command shape.
- CPU, RAM, GPU, and disk telemetry.
- Hugging Face model/dataset search and `snapshot_download` downloads into the standard HF cache.
- Hugging Face cache table, size accounting, and cached repo deletion.
- Ollama local model detection with compatibility notes.

## Axolotl Model Format Guardrail

Axolotl `base_model` should be a Hugging Face model id or a local Transformers-style model
directory containing files such as `config.json` and `.safetensors`, `.bin`, or `.pt` weights.
Ollama GGUF/internal blob models are detected and blocked as Axolotl `base_model` values.

Ollama models are only applied automatically when Ollama exposes a readable local
Safetensors/Transformers model directory as the model source.

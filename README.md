# Axolotl LCARS UI

LCARS frontend and manager for Axolotl config editing, run monitoring, local resource telemetry,
Hugging Face content management, and Ollama model detection.

Built with [LCARS WebUI](https://github.com/darsrc/LCARS-WebUI).

Project documentation is available in [docs/wiki/Home.md](docs/wiki/Home.md). The GitHub wiki
setting is enabled, but GitHub does not create the hidden `.wiki.git` repo until the first page is
created in the GitHub web UI.

## Launch

Use Python 3.11 unless your Axolotl install target requires a different supported Python version.
The repository includes `.python-version` for tools that honor it.

### Install With uv

```bash
uv python install 3.11
uv venv --python 3.11
uv pip install -r requirements.txt
```

### Install With venv/pip

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` installs `lcars-ui` directly from
[darsrc/LCARS-WebUI](https://github.com/darsrc/LCARS-WebUI). For local `lcars-ui` development,
install your own checkout into the venv in editable mode after installing requirements:

```bash
uv pip install -e /path/to/LCARS-WebUI/lcars-ui --reinstall-package lcars-ui
```

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
- Smart Setup recipes, model/dataset presets, and an Axolotl-defaults table that separates
  upstream defaults from this UI's starter suggestions.
- Raw YAML editor at `/raw`.
- Preflight gate that blocks or warns on model formats, local file paths, dataset shape,
  quantization/adapters, precision conflicts, attention backends, distributed settings,
  checkpoint/resume hazards, hub auth, and tracking integrations.
- Axolotl subprocess start/stop and live log viewer for preprocess, train, inference, merge,
  evaluate, lm-eval, and quantize commands.
- Separate Axolotl action args and launcher args. Launcher args are placed after `--`, matching
  Axolotl's launcher command shape.
- CPU, RAM, GPU, disk, top-process, GPU-process, and training-artifact storage telemetry.
- Hugging Face model/dataset search with compatibility classification, repo inspection,
  copyable selected repo ids, compatible-file listings, fine-tune lookup, and filtered
  `snapshot_download` downloads into the standard HF cache.
- Hugging Face cache table, size accounting, and cached repo deletion.
- Ollama local model detection that can apply real local Transformers directories or launch a
  compatible Hugging Face source search for runtime-only Ollama/GGUF models.

## Axolotl Model Format Guardrail

Axolotl `base_model` should be a Hugging Face model id or a local Transformers-style model
directory containing files such as `config.json` and `.safetensors`, `.bin`, or `.pt` weights.
Ollama GGUF/internal blob models are detected and blocked as Axolotl `base_model` values.

Ollama models are only applied automatically when Ollama exposes a readable local
Safetensors/Transformers model directory as the model source.

The HF Hub browser only downloads Axolotl-relevant file patterns by default: model config,
tokenizer/support files, `.safetensors`, `.bin`, and `.pt` weights for models, and
JSON/JSONL/Parquet/CSV/Arrow/text-style files for datasets.

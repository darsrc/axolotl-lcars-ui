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

`requirements.txt` installs the tested LCARS WebUI v4.4.0 release tag directly from
[darsrc/LCARS-WebUI](https://github.com/darsrc/LCARS-WebUI). The tag is pinned so a future
upstream `main` change cannot silently break this app. For local `lcars-ui` development, install
your own checkout into the venv in editable mode after installing requirements:

```bash
uv pip install -e /path/to/LCARS-WebUI/lcars-ui --reinstall-package lcars-ui
```

Start the app:

```bash
./launch.sh
```

The launcher prefers `.venv`, then `venv`, then a single other virtualenv directly inside the
project. It activates the selected environment for the child process, so an `axolotl` executable
installed in that environment is visible to the UI. Select a specific environment when needed:

```bash
AXOLOTL_LCARS_VENV=axolotl-training ./launch.sh
```

If no project virtualenv exists, the launcher creates `.venv` with `uv` and installs the UI
requirements. It does not install Axolotl automatically because its install depends on the
machine's accelerator stack. If neither a virtualenv nor `uv` is available, the launcher prints
manual setup instructions and exits.

Useful launcher args:

```bash
./launch.sh help
./launch.sh port 8080
./launch.sh ip 0.0.0.0 port 8080
./launch.sh --ip 0.0.0.0 --port 8080 --open
```

The launcher traps `Ctrl+C` and asks the Python server to terminate cleanly.

## Current Features

- Beginner-first **LoRA Studio** with a four-step Setup → Data → Train → Test journey for
  personality, agent-behavior, and hybrid adapters. The setup wizard translates four plain
  choices into a normal Axolotl YAML config; the JSONL editor validates chat examples, calls out
  unfinished placeholders, and keeps a backup when a draft is overwritten.
- Focused LoRA training gate and monitor with plain-language settings, preflight/data readiness,
  live process/GPU/RAM state, Axolotl logs, safe start/stop controls, and automatic Safetensors
  adapter discovery.
- Ollama adapter packaging and held-out base-vs-LoRA comparison. The test page creates a managed
  Modelfile, requires an explicitly selected installed base, imports the Axolotl adapter through
  `ollama create`, and runs both models through the local Ollama chat API.
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
- Editable, typed Axolotl workflow canvas with a stage palette, connection validation,
  import/export, undo/redo, persisted layouts, preflight-gated sequential execution,
  node-level live status, and safe cancellation. The one-off command launcher remains available
  as a manual override on the focused Console page.
- Separate Axolotl action args and launcher args. Launcher args are placed after `--`, matching
  Axolotl's launcher command shape.
- CPU, RAM, GPU, disk, top-process, GPU-process, and training-artifact storage telemetry in
  aligned overview/detail rows with compact GPU readouts and independent table scrolling.
- LCARS WebUI v4.4 content-sized, viewport-aware mosaic layouts with edge-aware operator
  arrangement, editable rows, columns and sections, stable panel grouping, dense-page filler
  control, native sortable/filterable/pageable data tables, and an immersive typed node-graph
  editor with groups, comments, reroutes, a searchable palette, and JSON interchange. Tables keep
  stable selection, rich expandable details, linked/copyable cells, and inline actions; controls
  and logs are searchable; defaults and persisted preferences are validated and typed; and
  consequential process, download, and cache actions require confirmation.
- Hugging Face model/dataset search in a responsive master/detail workspace with a dominant
  repository table and a unified operations rail for atomic search/exact-repository queries,
  selected-repository actions, and advanced filters. The rail stacks below results on compact
  screens. Search and repository-target types remain independent; typed sorting, local metadata
  filters, model
  VRAM/data-size fit; compatibility classification; stable row selection; automatic metadata
  hydration for each visible result page; rich expandable metadata/file/lineage views; in-place
  inspect/copy/queue/config actions; fine-tune lookup; and filtered `snapshot_download` downloads
  into the standard HF cache. Transfer monitoring now lives beside cache operations on the
  Content page.
- The workflow graph and control selections (search query, filters, sort, run action and args,
  active config, Ollama model) persist to `.lcars-ui-state.json` and are restored after a browser
  reload or server restart. Structured config values persist in the active YAML file itself.
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

# Setup

## Local Launch

Use Python 3.11 unless your Axolotl install target requires a different supported Python version.
The repository includes `.python-version` for tools that honor it.

### uv

```bash
uv python install 3.11
uv venv --python 3.11
uv pip install -r requirements.txt
./launch.sh
```

### venv/pip

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
./launch.sh
```

Useful launch forms:

```bash
./launch.sh help
./launch.sh port 8080
./launch.sh ip 0.0.0.0 port 8080
./launch.sh --ip 0.0.0.0 --port 8080 --open
```

The launcher traps `Ctrl+C` and asks the Python server to terminate cleanly.

`launch.sh` first uses an already-active virtualenv from `VIRTUAL_ENV`. Otherwise it prefers
`.venv`, then `venv`, then a single other virtualenv directly inside the project. It exports the
selected environment's `VIRTUAL_ENV` and prepends its `bin` directory to `PATH`, which lets the UI
discover an `axolotl` executable installed there. To choose a specific environment with the
highest-priority override:

```bash
AXOLOTL_LCARS_VENV=axolotl-training ./launch.sh
```

When no project virtualenv exists, the launcher uses `uv` to create `.venv` with Python 3.11 and
installs `requirements.txt`. Axolotl itself remains a separate, accelerator-specific install. If
`uv` is unavailable, the launcher exits with manual virtualenv instructions.

## Axolotl Requirement

The UI can edit configs and manage content without Axolotl installed, but run launch is blocked
until the `axolotl` executable is available inside the selected app virtualenv (or otherwise on
its `PATH`).

Install Axolotl according to the official docs for your CUDA/ROCm/Mac environment, then restart the UI.

## Beginner LoRA Studio

The first five navigation items form one guided workflow:

1. **LoRA Studio** explains what a LoRA can and cannot teach and shows readiness for each step.
2. **1 · Setup** asks only for a project name, goal, base model, and plain-language smart preset.
   It detects GPU VRAM, marks the best starting recipe, and fills the related adapter, rank,
   context, batch, optimizer, dataset, output, and safety fields. Presets cover a quick pipeline
   check, everyday chat, low-VRAM QLoRA, and higher-capacity behavior training.
3. **2 · Data** lets you enter a user prompt and exact ideal answer in a normal form; **Add &
   Save** generates the chat JSONL automatically. The first form entry replaces the untouched
   `EDIT ME` starter template and later entries append safely. A collapsible raw editor remains
   available for bulk work. An overwritten dataset keeps its immediately previous version as
   `filename.jsonl.bak`. Guided JSONL files are gitignored by default because personality data can
   contain private material.
4. **3 · Train** blocks its beginner start button until the configured data passes the Studio
   checks and Axolotl preflight. **Start Training** reruns those checks automatically; standalone
   preprocessing remains an optional diagnostic. The page exposes stop, live logs, GPU/RAM
   telemetry, and generated adapter directories without requiring the full workflow editor.
5. **4 · Test** packages a detected Safetensors adapter with a selected local Ollama base and
   compares the same held-out prompt against the base and tuned models.

A personality LoRA can teach voice, tone, boundaries, and response habits. An agent-behavior
LoRA can teach planning, clarification, tool-call, and recovery patterns, but the runtime must
still provide actual tools, permissions, memory, and an execution loop.

Start with the recommended preset and tune only after observing a concrete problem on held-out
prompts. The Setup page shows each active value alongside a starter range, when to change it, and
the cost of doing so. The structured Config pages also place descriptions beneath every Axolotl
field, with richer per-option explanations for the core LoRA, precision, attention, save, and
evaluation choices.

For Ollama import, use the same base-model lineage used during training. The UI intentionally
does not auto-select a merely similar model. Ollama documents Safetensors adapter import for
Llama, Mistral/Mixtral, and Gemma families and recommends non-quantized adapters for the smoothest
path; QLoRA remains available when VRAM is the tighter constraint.

## Setup Page

The Setup page separates starter choices from raw Axolotl fields:

- Smart Setup recipes apply coherent LoRA, QLoRA, chat-template, or local-completion starter values.
- Model and dataset presets fill the required `base_model` and `datasets[0].path` fields.
- The defaults table distinguishes Axolotl upstream defaults from this UI's starter suggestions.
- Required setup fields are labeled as required; optional fields are omitted from YAML when left unset.

## LCARS WebUI

This app uses [LCARS WebUI](https://github.com/darsrc/LCARS-WebUI). `requirements.txt` installs the
tested v4.4.0 tag directly from GitHub for reproducible installs. For local LCARS WebUI development,
install your checkout into the venv in editable mode after installing requirements:

```bash
uv pip install -e /path/to/LCARS-WebUI/lcars-ui --reinstall-package lcars-ui
```

## Workflow Page

The Workflow page uses LCARS WebUI's typed node canvas as an executable Axolotl lifecycle:

- The starter graph runs `preprocess → train → evaluate` from the active config.
- Add other supported Axolotl stages from the searchable palette, then connect them into one
  continuous chain from **Active Config**.
- Launcher mode, Axolotl arguments, and launcher arguments live on the stage that uses them.
- **Validate Workflow** checks the graph and the active config without launching a process.
- **Start Workflow** runs connected stages sequentially after confirmation and preflight.
- The graph is locked during execution; queued nodes are cancelled if a stage fails or the
  operator cancels the plan.
- Graph layout and values persist in `.lcars-ui-state.json`. Native JSON import/export provides a
  portable copy; imported graphs are normalized to this app's trusted Axolotl node templates.
- The **Console** page's **Single Action Override** retains the direct command launcher for utility
  or ad-hoc work while keeping the immersive graph page focused.

## Hugging Face

Set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in your shell when downloading private repos or pushing prepared datasets/models.

The HF Hub is a responsive master/detail discovery workspace. The repository browser owns the
main lane; its operations rail keeps the unified Hub query and selected-repository actions above
advanced filters, and automatically stacks below results on compact screens. The transfer queue
and activity log live with cache operations on the **Content** page. The repository table remains
the center of the workflow:

- Click a row to target Selected Repository Actions.
- Hub Query submits its mode, query/repository id, repo type, and optional revision atomically.
  **Search Hub** discovers repositories; **Inspect Exact Repository** opens an `owner/repository`
  outside the current results. Result Filters atomically refreshes Hub sort, compatibility,
  result limit, metadata/artifact/weight filters, and model VRAM fit.
- Search repo type and selected repository type are independent, so switching a search between
  models and datasets cannot silently retarget actions for an already selected repository.
- The visible page is inspected automatically to populate model VRAM fit or dataset size,
  weight/data formats, and exact file counts. Moving to another page hydrates that page.
- Expand a row for its full manifest, compatibility, lineage, exact per-file sizes, related
  fine-tunes, and inline config/download actions.
- Repository ids and file paths have explicit copy controls; repository ids also open the
  corresponding Hugging Face page.
- Inspection failures stay in the expanded row with a retry action.
- Queue and activity details remain visible on **Content** while downloads run; the last
  completed local snapshot can also be applied to the active config there.

Model downloads are filtered to Axolotl-relevant config/tokenizer/support files plus
`.safetensors`, `.bin`, and `.pt` weights. Dataset downloads are filtered to JSON, JSONL,
Parquet, CSV, Arrow, and text-style files.

## Tracking Integrations

Set integration credentials only in your shell or `.env`-style local environment files. Do not commit secrets.

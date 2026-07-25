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

## Axolotl Requirement

The UI can edit configs and manage content without Axolotl installed, but run launch is blocked until the `axolotl` executable is available on `PATH` inside the app environment.

Install Axolotl according to the official docs for your CUDA/ROCm/Mac environment, then restart the UI.

## Setup Page

The Setup page separates starter choices from raw Axolotl fields:

- Smart Setup recipes apply coherent LoRA, QLoRA, chat-template, or local-completion starter values.
- Model and dataset presets fill the required `base_model` and `datasets[0].path` fields.
- The defaults table distinguishes Axolotl upstream defaults from this UI's starter suggestions.
- Required setup fields are labeled as required; optional fields are omitted from YAML when left unset.

## LCARS WebUI

This app uses [LCARS WebUI](https://github.com/darsrc/LCARS-WebUI). `requirements.txt` installs the
tested v4.3.0 tag directly from GitHub for reproducible installs. For local LCARS WebUI development,
install your checkout into the venv in editable mode after installing requirements:

```bash
uv pip install -e /path/to/LCARS-WebUI/lcars-ui --reinstall-package lcars-ui
```

## Hugging Face

Set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in your shell when downloading private repos or pushing prepared datasets/models.

The HF Hub is a content-sized discovery workspace. Its repository browser, unified Hub query,
advanced result filters, and selected-repository actions form one workflow; the
transfer queue and activity log now live with cache operations on the **Content** page. Use the
rail's **Arrange** control to move or resize panels. Edge drops place a panel beside or
above/below another; **+ Row**, **+ Column**, and **+ Section** create explicit workflow bands.
The repository table remains the center of the workflow:

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

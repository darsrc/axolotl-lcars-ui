# Setup

## Local Launch

```bash
python3 -m venv .venv
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

## LCARS WebUI

This app uses [LCARS WebUI](https://github.com/darsrc/LCARS-WebUI). For local LCARS WebUI development, install your checkout into the venv in editable mode before launching this app.

## Hugging Face

Set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in your shell when downloading private repos or pushing prepared datasets/models.

## Tracking Integrations

Set integration credentials only in your shell or `.env`-style local environment files. Do not commit secrets.

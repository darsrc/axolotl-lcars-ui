# Axolotl LCARS UI

Axolotl LCARS UI is a browser-based control surface for configuring, validating, launching, and monitoring [Axolotl](https://docs.axolotl.ai/) workflows.

It is built with [LCARS WebUI](https://github.com/darsrc/LCARS-WebUI).

## What It Manages

- Structured Axolotl YAML editing with a raw YAML escape hatch.
- Preflight validation for common expensive mistakes before a run starts.
- Axolotl CLI process launch, stop, and log monitoring.
- CPU, RAM, disk, and GPU telemetry.
- Hugging Face model and dataset search/download/cache management.
- Ollama model detection with format guardrails.

## Main Pages

- **Command**: readiness summary, preflight matrix, quick actions.
- **Config**: config file management and structured coverage map.
- **Setup**: model, tokenizer, dataset, and sequence/packing options.
- **Train**: output, PEFT/adapters, optimizer, schedule, and batch sizing.
- **Hardware**: precision, quantization, kernels, DeepSpeed, FSDP, and parallelism.
- **Tracking**: logging, eval, integrations, RL, TRL, vLLM, and lm-eval settings.
- **Run**: Axolotl command launcher and live logs.
- **Resources**: local system telemetry.
- **HF Hub**: Hugging Face search and downloads.
- **Content**: downloaded content size and cleanup.
- **Ollama**: local Ollama detection and compatibility notes.

## Related Pages

- [Setup](Setup.md)
- [Axolotl Guardrails](Axolotl-Guardrails.md)

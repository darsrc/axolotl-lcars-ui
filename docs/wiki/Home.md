# Axolotl LCARS UI

Axolotl LCARS UI is a browser-based control surface for configuring, validating, launching, and monitoring [Axolotl](https://docs.axolotl.ai/) workflows.

It is built with the pinned v4.2.0 release of
[LCARS WebUI](https://github.com/darsrc/LCARS-WebUI).

## What It Manages

- Structured Axolotl YAML editing with a raw YAML escape hatch.
- Smart setup recipes with visible Axolotl defaults versus UI starter suggestions.
- Preflight validation for common expensive mistakes before a run starts.
- Axolotl CLI process launch, stop, and log monitoring.
- CPU, RAM, disk, GPU, top-process, GPU-process, and storage-hotspot telemetry.
- Native v4.2 viewport-aware mosaic layouts with operator arrangement controls, stable panel
  grouping and sizing, dense-page filler control, sortable/filterable/pageable tables with
  stable selection, rich expansion, linked/copyable cells and inline actions; searchable
  controls and logs; typed defaults, validated preferences, and atomic form submissions;
  collapsible panels; richer telemetry; and confirmation prompts for consequential actions.
- Hugging Face model and dataset search/download/cache management with typed result sorting,
  stable row selection, automatic visible-page metadata hydration, rich repository expansion,
  in-place inspect/copy/queue/config actions, local metadata filters, model VRAM fit, dataset
  size, file compatibility, and fine-tune lookup.
- Ollama model detection with local-source apply or Hugging Face source search.
- Control selections persist across browser reloads and server restarts via
  `.lcars-ui-state.json`; structured config values persist in the active YAML file.

## Main Pages

- **Command**: readiness summary, preflight matrix, quick actions.
- **Config**: config file management and structured coverage map.
- **Setup**: recipes, defaults/examples, model, tokenizer, dataset, and sequence/packing options.
- **Train**: output, PEFT/adapters, optimizer, schedule, and batch sizing.
- **Hardware**: precision, quantization, kernels, DeepSpeed, FSDP, and parallelism.
- **Tracking**: logging, eval, integrations, RL, TRL, vLLM, and lm-eval settings.
- **Run**: Axolotl command launcher and live logs.
- **Resources**: local system telemetry and resource attribution.
- **HF Hub**: arrangeable search, sift, result, target, workflow, transfer, and activity panels
  with independent search/target types, lazy expandable manifests, inline file/config actions,
  fine-tunes, and filtered downloads.
- **Content**: downloaded content size and cleanup.
- **Ollama**: local Ollama detection and compatibility notes.

## Related Pages

- [Setup](Setup.md)
- [Axolotl Guardrails](Axolotl-Guardrails.md)

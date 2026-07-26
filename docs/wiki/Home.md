# Axolotl LCARS UI

Axolotl LCARS UI is a browser-based control surface for configuring, validating, launching, and monitoring [Axolotl](https://docs.axolotl.ai/) workflows.

It is built with the pinned v4.4.0 release of
[LCARS WebUI](https://github.com/darsrc/LCARS-WebUI).

## What It Manages

- Structured Axolotl YAML editing with a raw YAML escape hatch.
- Smart setup recipes with visible Axolotl defaults versus UI starter suggestions.
- Preflight validation for common expensive mistakes before a run starts.
- Axolotl CLI process launch, stop, and log monitoring.
- Typed workflow graph editing with validated connections, sequential stage execution, live
  node status, import/export, and persisted layouts.
- CPU, RAM, disk, GPU, top-process, GPU-process, and storage-hotspot telemetry.
- Native v4.4 content-sized, viewport-aware mosaic layouts with edge-aware operator arrangement,
  editable rows, columns and sections, stable panel grouping, dense-page filler control, and
  sortable/filterable/pageable tables with stable selection, rich expansion, linked/copyable
  cells and inline actions; an immersive typed node canvas with groups, comments, reroutes,
  palette search, undo/redo, and JSON interchange; searchable controls and logs; typed defaults,
  validated preferences, and atomic form submissions; collapsible panels; richer telemetry; and
  confirmation prompts for consequential actions.
- Hugging Face model and dataset search/download/cache management with typed result sorting,
  stable row selection, automatic visible-page metadata hydration, rich repository expansion,
  in-place inspect/copy/queue/config actions, local metadata filters, model VRAM fit, dataset
  size, file compatibility, and fine-tune lookup.
- Ollama model detection with local-source apply or Hugging Face source search.
- The workflow graph and control selections persist across browser reloads and server restarts
  via `.lcars-ui-state.json`; structured config values persist in the active YAML file.

## Main Pages

- **Command**: readiness summary, preflight matrix, quick actions.
- **Config**: config file management and structured coverage map.
- **Setup**: recipes, defaults/examples, model, tokenizer, dataset, and sequence/packing options.
- **Train**: output, PEFT/adapters, optimizer, schedule, and batch sizing.
- **Hardware**: precision, quantization, kernels, DeepSpeed, FSDP, and parallelism.
- **Tracking**: logging, eval, integrations, RL, TRL, vLLM, and lm-eval settings.
- **Workflow**: editable lifecycle graph, preflight-gated sequential execution, node status,
  cancellation, and lifecycle controls.
- **Console**: live Axolotl process output and the one-off action override.
- **Resources**: aligned system-load, process, GPU, filesystem, and storage-pressure telemetry
  with independently scrolling detail tables.
- **HF Hub**: responsive master/detail repository browser with results beside a unified operations
  rail for search, selected-repository actions, and advanced filters; independent search/target
  types, lazy expandable manifests, inline file/config actions, fine-tunes, and filtered downloads.
- **Content**: transfer queue, Hub activity, downloaded content size, and cache cleanup.
- **Ollama**: local Ollama detection and compatibility notes.

## Related Pages

- [Setup](Setup.md)
- [Axolotl Guardrails](Axolotl-Guardrails.md)

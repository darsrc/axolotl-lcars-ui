"""Editable Axolotl workflow graphs and sequential execution.

LCARS WebUI owns the graph editor and its interchange format.  This module owns
the application-specific meaning layered on top of that format: which node
templates are allowed, how a valid graph becomes an ordered Axolotl plan, and
how that plan advances through the existing single-process runner.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lcars_ui as lcars

from axolotl_lcars_ui.runner import CONFIG_ACTIONS, LAUNCHER_ACTIONS, AxolotlRunner


SOURCE_TEMPLATE_ID = "active-config"
SOURCE_NODE_ID = "active-config"
STAGE_PORT_TYPE = "axolotl-stage"
VALID_LAUNCHERS = {"", "python", "accelerate", "torchrun"}
DEFAULT_WORKFLOW_ACTIONS = ("preprocess", "train", "evaluate")

_ACTION_LABELS = {
    "preprocess": "Preprocess",
    "train": "Train",
    "inference": "Inference",
    "merge-lora": "Merge LoRA",
    "merge-sharded-fsdp-weights": "Merge FSDP",
    "evaluate": "Evaluate",
    "lm-eval": "LM Eval",
    "quantize": "Quantize",
}

_ACTION_CATEGORIES = {
    "preprocess": "Training",
    "train": "Training",
    "inference": "Interactive",
    "merge-lora": "Artifacts",
    "merge-sharded-fsdp-weights": "Artifacts",
    "evaluate": "Evaluation",
    "lm-eval": "Evaluation",
    "quantize": "Artifacts",
}

_ACTION_COLORS = {
    "preprocess": "#fdcb64",
    "train": "#f89800",
    "inference": "#9897fc",
    "merge-lora": "#cc9bcc",
    "merge-sharded-fsdp-weights": "#6e6cd6",
    "evaluate": "#9897fc",
    "lm-eval": "#6e6cd6",
    "quantize": "#ce6262",
}


class WorkflowError(RuntimeError):
    """A workflow cannot be edited, compiled, or executed safely."""


@dataclass(frozen=True)
class WorkflowStep:
    """One compiled Axolotl invocation."""

    node_id: str
    action: str
    label: str
    launcher: str = ""
    cli_args: str = ""
    launcher_args: str = ""


def _field_values(action: str) -> dict[str, str]:
    values = {"cli_args": ""}
    if action in LAUNCHER_ACTIONS:
        values.update({"launcher": "", "launcher_args": ""})
    return values


def _action_fields(action: str) -> list[lcars.GraphField]:
    fields: list[lcars.GraphField] = []
    if action in LAUNCHER_ACTIONS:
        fields.append(
            lcars.GraphField(
                id="launcher",
                label="Launcher",
                kind="select",
                default="",
                options=[
                    lcars.GraphFieldOption(value="", label="Axolotl default"),
                    lcars.GraphFieldOption(value="python", label="Python"),
                    lcars.GraphFieldOption(value="accelerate", label="Accelerate"),
                    lcars.GraphFieldOption(value="torchrun", label="Torchrun"),
                ],
            )
        )
    fields.append(
        lcars.GraphField(
            id="cli_args",
            label="Axolotl Args",
            kind="text",
            default="",
            placeholder="optional action flags",
        )
    )
    if action in LAUNCHER_ACTIONS:
        fields.append(
            lcars.GraphField(
                id="launcher_args",
                label="Launcher Args",
                kind="text",
                default="",
                placeholder="arguments placed after --",
            )
        )
    return fields


def workflow_templates() -> list[lcars.NodeTemplate]:
    """Return the canonical, executable workflow node templates."""

    source = lcars.NodeTemplate(
        id=SOURCE_TEMPLATE_ID,
        label="Active Config",
        category="Source",
        color="#fbab3b",
        outputs=[
            lcars.GraphPort(
                id="next",
                label="PLAN",
                type=STAGE_PORT_TYPE,
                capacity=1,
            )
        ],
    )
    actions = [
        lcars.NodeTemplate(
            id=action,
            label=_ACTION_LABELS[action],
            category=_ACTION_CATEGORIES[action],
            color=_ACTION_COLORS[action],
            inputs=[
                lcars.GraphPort(
                    id="previous",
                    label="IN",
                    type=STAGE_PORT_TYPE,
                )
            ],
            outputs=[
                lcars.GraphPort(
                    id="next",
                    label="OUT",
                    type=STAGE_PORT_TYPE,
                    capacity=1,
                )
            ],
            fields=_action_fields(action),
        )
        for action in CONFIG_ACTIONS
    ]
    return [source, *actions]


def default_workflow(active_config: str) -> lcars.GraphDocument:
    """Build the useful starter lifecycle shown on a fresh install."""

    nodes = [
        lcars.GraphNode(
            id=SOURCE_NODE_ID,
            template=SOURCE_TEMPLATE_ID,
            label=f"Config · {active_config}",
            position=(20.0, 30.0),
            group="training-lifecycle",
        )
    ]
    for index, action in enumerate(DEFAULT_WORKFLOW_ACTIONS, start=1):
        nodes.append(
            lcars.GraphNode(
                id=f"{action}-{index}",
                template=action,
                position=(20.0 + index * 245.0, 30.0),
                values=_field_values(action),
                group="training-lifecycle",
            )
        )

    edge_nodes = [node.id for node in nodes]
    edges = [
        lcars.GraphEdge(
            id=f"stage-{index}",
            source=source_id,
            source_port="next",
            target=target_id,
            target_port="previous",
        )
        for index, (source_id, target_id) in enumerate(
            zip(edge_nodes, edge_nodes[1:]),
            start=1,
        )
    ]
    return lcars.GraphDocument(
        templates=workflow_templates(),
        nodes=nodes,
        edges=edges,
        groups=[
            lcars.GraphGroup(
                id="training-lifecycle",
                label="Training Lifecycle",
                position=(-20.0, -20.0),
                size=(1010.0, 245.0),
                color="#f89800",
            )
        ],
        comments=[
            lcars.GraphComment(
                id="workflow-guide",
                text=(
                    "WIRE ONE CONTINUOUS PLAN FROM ACTIVE CONFIG. "
                    "ADD STAGES FROM THE PALETTE; EXECUTION FOLLOWS THE WIRES."
                ),
                position=(20.0, 275.0),
                size=(650.0, 82.0),
            )
        ],
        viewport=lcars.GraphViewport(x=55.0, y=80.0, zoom=0.78),
    )


def canonicalize_workflow(
    document: lcars.GraphDocument | dict[str, Any],
    active_config: str,
) -> lcars.GraphDocument:
    """Apply trusted templates while preserving an operator's graph edits.

    Imported graph files are valid generic LCARS graphs, but only this
    application's templates may become subprocess commands.  Rebuilding the
    document against the canonical templates keeps that boundary explicit.
    """

    parsed = (
        document
        if isinstance(document, lcars.GraphDocument)
        else lcars.GraphDocument.model_validate(document)
    )
    allowed = {SOURCE_TEMPLATE_ID, *CONFIG_ACTIONS}
    unknown = sorted({node.template for node in parsed.nodes} - allowed)
    if unknown:
        raise WorkflowError(f"Unsupported workflow node type(s): {', '.join(unknown)}")

    nodes = [
        node.model_copy(
            update={"label": f"Config · {active_config}"}
            if node.template == SOURCE_TEMPLATE_ID
            else {}
        )
        for node in parsed.nodes
    ]
    try:
        return lcars.GraphDocument(
            templates=workflow_templates(),
            nodes=nodes,
            edges=parsed.edges,
            reroutes=parsed.reroutes,
            groups=parsed.groups,
            comments=parsed.comments,
            viewport=parsed.viewport,
        )
    except Exception as exc:
        raise WorkflowError(f"Workflow does not match the Axolotl node contract: {exc}") from exc


def _text_value(values: dict[str, Any], key: str) -> str:
    value = values.get(key, "")
    return "" if value is None else str(value).strip()


def _argument_value(values: dict[str, Any], key: str, stage_label: str) -> str:
    value = _text_value(values, key)
    if len(value) > 4096:
        raise WorkflowError(f"{stage_label} {key.replace('_', ' ')} exceeds 4096 characters.")
    if "\x00" in value:
        raise WorkflowError(f"{stage_label} {key.replace('_', ' ')} contains a NUL byte.")
    try:
        shlex.split(value)
    except ValueError as exc:
        raise WorkflowError(
            f"{stage_label} {key.replace('_', ' ')} is not valid shell-style input: {exc}"
        ) from exc
    return value


def compile_workflow(document: lcars.GraphDocument) -> list[WorkflowStep]:
    """Compile one continuous source-to-terminal graph into execution order."""

    source_nodes = [node for node in document.nodes if node.template == SOURCE_TEMPLATE_ID]
    if len(source_nodes) != 1:
        raise WorkflowError("Workflow requires exactly one Active Config source node.")

    action_nodes = [node for node in document.nodes if node.template in CONFIG_ACTIONS]
    if not action_nodes:
        raise WorkflowError("Workflow needs at least one Axolotl action stage.")

    outgoing: dict[str, list[lcars.GraphEdge]] = {node.id: [] for node in document.nodes}
    incoming: dict[str, list[lcars.GraphEdge]] = {node.id: [] for node in document.nodes}
    for edge in document.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)

    source = source_nodes[0]
    if incoming[source.id]:
        raise WorkflowError("The Active Config source cannot have an incoming stage.")

    visited = {source.id}
    steps: list[WorkflowStep] = []
    current_id = source.id
    while outgoing[current_id]:
        if len(outgoing[current_id]) != 1:
            raise WorkflowError("Workflow stages must form one unbranched execution chain.")
        edge = outgoing[current_id][0]
        target = document.node(edge.target)
        if target is None:
            raise WorkflowError(f"Workflow edge {edge.id} has no target node.")
        if target.id in visited:
            raise WorkflowError("Workflow contains an execution cycle.")
        if len(incoming[target.id]) != 1:
            raise WorkflowError(f"Stage {target.label or target.id} has multiple predecessors.")
        if target.template not in CONFIG_ACTIONS:
            raise WorkflowError("Only Axolotl action stages may follow Active Config.")

        stage_label = target.label or _ACTION_LABELS[target.template]
        launcher = _text_value(target.values, "launcher")
        cli_args = _argument_value(target.values, "cli_args", stage_label)
        launcher_args = _argument_value(target.values, "launcher_args", stage_label)
        if launcher not in VALID_LAUNCHERS:
            raise WorkflowError(f"{stage_label} has an unknown launcher.")
        if launcher and target.template not in LAUNCHER_ACTIONS:
            raise WorkflowError(f"{_ACTION_LABELS[target.template]} does not support a launcher.")
        if launcher_args and not launcher:
            raise WorkflowError(
                f"{_ACTION_LABELS[target.template]} has launcher args but no launcher."
            )

        steps.append(
            WorkflowStep(
                node_id=target.id,
                action=target.template,
                label=stage_label,
                launcher=launcher,
                cli_args=cli_args,
                launcher_args=launcher_args,
            )
        )
        visited.add(target.id)
        current_id = target.id

    unconnected = [node.label or node.id for node in action_nodes if node.id not in visited]
    if unconnected:
        raise WorkflowError(
            "Every action must belong to the Active Config chain; disconnected: "
            + ", ".join(unconnected)
        )
    return steps


class WorkflowManager:
    """Persist an editable graph and advance its compiled plan one process at a time."""

    def __init__(
        self,
        active_config: str,
        document: lcars.GraphDocument | dict[str, Any] | None = None,
    ) -> None:
        self.document = (
            default_workflow(active_config)
            if document is None
            else canonicalize_workflow(document, active_config)
        )
        self.status = "idle"
        self.message = "Edit the graph, validate it, then start the workflow."
        self.node_execution: dict[str, lcars.GraphNodeExecution] = {}
        self._plan: list[WorkflowStep] = []
        self._next_index = 0
        self._active_step: WorkflowStep | None = None
        self._config_path: Path | None = None

    @property
    def is_active(self) -> bool:
        return self.status in {"queued", "running"}

    @property
    def current_label(self) -> str:
        if self._active_step is not None:
            return self._active_step.label
        if self.status == "success":
            return "Workflow complete"
        if self.status == "error":
            return "Workflow failed"
        if self.status == "cancelled":
            return "Workflow cancelled"
        return "No active stage"

    @property
    def progress_percent(self) -> float:
        if not self._plan:
            return 0.0
        completed = sum(
            1
            for step in self._plan
            if self.node_execution.get(step.node_id, lcars.GraphNodeExecution()).status == "success"
        )
        return completed / len(self._plan) * 100.0

    def plan(self) -> list[WorkflowStep]:
        return compile_workflow(self.document)

    def sync_active_config(self, active_config: str) -> bool:
        updated = canonicalize_workflow(self.document, active_config)
        if updated == self.document:
            return False
        self.document = updated
        return True

    def replace_document(
        self,
        document: lcars.GraphDocument | dict[str, Any],
        active_config: str,
    ) -> None:
        if self.is_active:
            raise WorkflowError("The workflow is locked while a plan is running.")
        self.document = canonicalize_workflow(document, active_config)
        self.status = "idle"
        self.message = "Workflow updated. Validate before launch."
        self.node_execution = {}
        self._plan = []
        self._next_index = 0
        self._active_step = None
        self._config_path = None

    def reset(self, active_config: str) -> None:
        if self.is_active:
            raise WorkflowError("Cancel the active workflow before resetting it.")
        self.document = default_workflow(active_config)
        self.status = "idle"
        self.message = "Starter preprocess → train → evaluate workflow restored."
        self.node_execution = {}
        self._plan = []
        self._next_index = 0
        self._active_step = None
        self._config_path = None

    def start(self, runner: AxolotlRunner, config_path: Path) -> None:
        if self.is_active:
            raise WorkflowError("A workflow is already running.")
        if runner.is_running():
            raise WorkflowError("Another Axolotl process is already running.")

        self._plan = self.plan()
        self._next_index = 0
        self._active_step = None
        self._config_path = config_path
        self.status = "queued"
        self.message = f"Queued {len(self._plan)} stage(s)."
        source_id = next(
            node.id for node in self.document.nodes if node.template == SOURCE_TEMPLATE_ID
        )
        self.node_execution = {
            source_id: lcars.GraphNodeExecution(
                status="success",
                progress=1.0,
                message=str(config_path),
            ),
            **{step.node_id: lcars.GraphNodeExecution(status="queued") for step in self._plan},
        }
        error = self._launch_next(runner)
        if error is not None:
            raise WorkflowError(error)

    def tick(self, runner: AxolotlRunner) -> bool:
        """Advance after process completion. Return whether execution state changed."""

        if self.status != "running" or self._active_step is None:
            return False
        if runner.is_running() or runner.state.status == "running":
            return False

        step = self._active_step
        if runner.state.status == "complete":
            self.node_execution[step.node_id] = lcars.GraphNodeExecution(
                status="success",
                progress=1.0,
                message=f"Exited with code {runner.state.returncode or 0}.",
            )
            self._active_step = None
            error = self._launch_next(runner)
            if error is not None:
                self.message = error
            return True

        if runner.state.status in {"failed", "stopped"}:
            cancelled = runner.state.status == "stopped"
            node_status = "cancelled" if cancelled else "error"
            self.node_execution[step.node_id] = lcars.GraphNodeExecution(
                status=node_status,
                message=f"Exited with code {runner.state.returncode}.",
            )
            self._active_step = None
            self.status = node_status
            self.message = (
                f"{step.label} was cancelled."
                if cancelled
                else f"{step.label} failed with exit code {runner.state.returncode}."
            )
            self._cancel_queued_nodes()
            return True
        return False

    def cancel(self, runner: AxolotlRunner) -> None:
        if not self.is_active:
            raise WorkflowError("No workflow is active.")
        active = self._active_step
        if runner.is_running():
            runner.stop()
        if active is not None:
            self.node_execution[active.node_id] = lcars.GraphNodeExecution(
                status="cancelled",
                message="Cancelled by operator.",
            )
        self._active_step = None
        self._cancel_queued_nodes()
        self.status = "cancelled"
        self.message = "Workflow cancelled by operator."

    def execution_state(self) -> lcars.GraphExecutionState:
        graph_status = self.status
        if graph_status not in {
            "idle",
            "queued",
            "running",
            "success",
            "error",
            "cancelled",
        }:
            graph_status = "idle"
        return lcars.GraphExecutionState(
            status=graph_status,
            nodes=dict(self.node_execution),
            message=self.message,
        )

    def _launch_next(self, runner: AxolotlRunner) -> str | None:
        if self._next_index >= len(self._plan):
            self.status = "success"
            self.message = f"Completed all {len(self._plan)} workflow stage(s)."
            self._active_step = None
            return None

        if self._config_path is None:
            self.status = "error"
            self.message = "Workflow lost its active config path."
            return self.message

        step = self._plan[self._next_index]
        self._next_index += 1
        self._active_step = step
        self.status = "running"
        self.message = f"Running {step.label} ({self._next_index}/{len(self._plan)})."
        self.node_execution[step.node_id] = lcars.GraphNodeExecution(
            status="running",
            message="Launching…",
        )
        try:
            runner.start(
                step.action,
                self._config_path,
                launcher=step.launcher,
                cli_args=step.cli_args,
                launcher_args=step.launcher_args,
            )
        except Exception as exc:
            self.node_execution[step.node_id] = lcars.GraphNodeExecution(
                status="error",
                message=str(exc),
            )
            self._active_step = None
            self.status = "error"
            self.message = f"Could not start {step.label}: {exc}"
            self._cancel_queued_nodes()
            return self.message

        self.node_execution[step.node_id] = lcars.GraphNodeExecution(
            status="running",
            message=shlex.join(runner.state.command),
        )
        return None

    def _cancel_queued_nodes(self) -> None:
        for step in self._plan[self._next_index :]:
            self.node_execution[step.node_id] = lcars.GraphNodeExecution(
                status="cancelled",
                message="Not run.",
            )


__all__ = [
    "DEFAULT_WORKFLOW_ACTIONS",
    "SOURCE_NODE_ID",
    "SOURCE_TEMPLATE_ID",
    "WorkflowError",
    "WorkflowManager",
    "WorkflowStep",
    "canonicalize_workflow",
    "compile_workflow",
    "default_workflow",
    "workflow_templates",
]

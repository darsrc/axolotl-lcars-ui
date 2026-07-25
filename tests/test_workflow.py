from __future__ import annotations

from types import SimpleNamespace

import lcars_ui as lcars
import pytest

from axolotl_lcars_ui.workflow import (
    SOURCE_NODE_ID,
    WorkflowError,
    WorkflowManager,
    canonicalize_workflow,
    compile_workflow,
    default_workflow,
)


class FakeRunner:
    def __init__(self) -> None:
        self.running = False
        self.starts: list[dict[str, str]] = []
        self.state = SimpleNamespace(
            command=[],
            status="idle",
            returncode=None,
        )

    def is_running(self) -> bool:
        return self.running

    def start(
        self,
        action: str,
        config_path: object,
        *,
        launcher: str = "",
        cli_args: str = "",
        launcher_args: str = "",
    ) -> None:
        self.starts.append(
            {
                "action": action,
                "config": str(config_path),
                "launcher": launcher,
                "cli_args": cli_args,
                "launcher_args": launcher_args,
            }
        )
        self.running = True
        self.state.command = ["axolotl", action, str(config_path)]
        self.state.status = "running"
        self.state.returncode = None

    def finish(self, returncode: int = 0) -> None:
        self.running = False
        self.state.status = "complete" if returncode == 0 else "failed"
        self.state.returncode = returncode

    def stop(self) -> None:
        self.running = False
        self.state.status = "stopped"
        self.state.returncode = -15


def test_default_workflow_compiles_into_the_training_lifecycle() -> None:
    document = default_workflow("starter.yml")

    assert [step.action for step in compile_workflow(document)] == [
        "preprocess",
        "train",
        "evaluate",
    ]
    assert document.node(SOURCE_NODE_ID).label == "Config · starter.yml"
    assert {template.id for template in document.templates} >= {
        "active-config",
        "train",
        "quantize",
        "merge-lora",
    }
    assert document.groups
    assert document.comments


def test_compile_preserves_typed_stage_launch_options() -> None:
    document = default_workflow("starter.yml")
    nodes = [
        node.model_copy(
            update={
                "values": {
                    "launcher": "accelerate",
                    "cli_args": "--debug",
                    "launcher_args": "--num_processes 2",
                }
            }
        )
        if node.template == "train"
        else node
        for node in document.nodes
    ]
    edited = document.model_copy(update={"nodes": nodes})

    train = next(step for step in compile_workflow(edited) if step.action == "train")

    assert train.launcher == "accelerate"
    assert train.cli_args == "--debug"
    assert train.launcher_args == "--num_processes 2"


def test_compile_rejects_disconnected_stages() -> None:
    document = default_workflow("starter.yml")
    disconnected = lcars.GraphDocument(
        templates=document.templates,
        nodes=document.nodes,
        edges=document.edges[:-1],
        groups=document.groups,
        comments=document.comments,
    )

    with pytest.raises(WorkflowError, match="disconnected"):
        compile_workflow(disconnected)


def test_compile_rejects_malformed_stage_arguments_before_launch() -> None:
    document = default_workflow("starter.yml")
    nodes = [
        node.model_copy(update={"values": {**node.values, "cli_args": "--name 'unterminated"}})
        if node.template == "train"
        else node
        for node in document.nodes
    ]

    with pytest.raises(WorkflowError, match="not valid shell-style input"):
        compile_workflow(document.model_copy(update={"nodes": nodes}))


def test_canonicalize_rejects_imported_non_axolotl_templates() -> None:
    imported = lcars.GraphDocument(
        templates=[lcars.NodeTemplate(id="shell")],
        nodes=[lcars.GraphNode(id="danger", template="shell")],
    )

    with pytest.raises(WorkflowError, match="Unsupported workflow node"):
        canonicalize_workflow(imported, "starter.yml")


def test_manager_runs_each_stage_only_after_the_previous_one_completes(tmp_path) -> None:
    manager = WorkflowManager("starter.yml")
    runner = FakeRunner()
    config_path = tmp_path / "starter.yml"

    manager.start(runner, config_path)  # type: ignore[arg-type]

    assert manager.status == "running"
    assert [item["action"] for item in runner.starts] == ["preprocess"]
    assert manager.execution_state().nodes["preprocess-1"].status == "running"

    runner.finish()
    assert manager.tick(runner)  # type: ignore[arg-type]
    assert [item["action"] for item in runner.starts] == ["preprocess", "train"]
    assert manager.progress_percent == pytest.approx(100 / 3)

    runner.finish()
    assert manager.tick(runner)  # type: ignore[arg-type]
    assert [item["action"] for item in runner.starts] == [
        "preprocess",
        "train",
        "evaluate",
    ]

    runner.finish()
    assert manager.tick(runner)  # type: ignore[arg-type]
    assert manager.status == "success"
    assert manager.progress_percent == 100
    assert manager.execution_state().nodes["evaluate-3"].status == "success"


def test_manager_failure_marks_the_stage_and_cancels_the_rest(tmp_path) -> None:
    manager = WorkflowManager("starter.yml")
    runner = FakeRunner()
    manager.start(runner, tmp_path / "starter.yml")  # type: ignore[arg-type]

    runner.finish(returncode=2)
    assert manager.tick(runner)  # type: ignore[arg-type]

    execution = manager.execution_state()
    assert manager.status == "error"
    assert execution.nodes["preprocess-1"].status == "error"
    assert execution.nodes["train-2"].status == "cancelled"
    assert execution.nodes["evaluate-3"].status == "cancelled"
    assert [item["action"] for item in runner.starts] == ["preprocess"]


def test_manager_cancel_stops_the_process_and_unlocks_the_graph(tmp_path) -> None:
    manager = WorkflowManager("starter.yml")
    runner = FakeRunner()
    manager.start(runner, tmp_path / "starter.yml")  # type: ignore[arg-type]

    manager.cancel(runner)  # type: ignore[arg-type]

    assert manager.status == "cancelled"
    assert not manager.is_active
    assert runner.state.status == "stopped"
    assert manager.execution_state().nodes["preprocess-1"].status == "cancelled"

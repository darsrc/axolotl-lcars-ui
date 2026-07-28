from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import lcars_ui as lcars

from axolotl_lcars_ui import main
from axolotl_lcars_ui.hf_manager import RepoDetails, RepoFile, SearchResult
from lcars_ui.dsl._state import (
    Mode,
    _LCARSContext,
    clear_session_state,
    get_ctx,
    get_session_state,
    set_ctx,
)


def _manifest_widgets(manifest: object) -> dict[str, object]:
    widgets: dict[str, object] = {}

    def visit(items: list[object]) -> None:
        for item in items:
            widgets[str(getattr(item, "id"))] = item
            for attribute in (
                "children",
                "left_inputs",
                "right_inputs",
                "main_children",
                "side_children",
                "header_children",
                "column_inputs",
                "left_children",
                "right_children",
                "rail_children",
                "content_children",
            ):
                nested = getattr(item, attribute, None)
                if isinstance(nested, list):
                    visit(nested)

    for page in getattr(manifest, "pages").values():
        for row in page.rows:
            for column in row.columns:
                visit(column.widgets)
    return widgets


class V44UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = main._build_manifest(main.build_ui, get_ctx().config)
        cls.widgets = _manifest_widgets(cls.manifest)

    def test_project_builds_with_lcars_v44(self) -> None:
        self.assertEqual(lcars.__version__, "4.4.0")
        self.assertEqual(len(self.manifest.pages), 18)

    def test_beginner_lora_studio_exposes_the_four_step_flow(self) -> None:
        self.assertEqual(
            list(self.manifest.pages)[:5],
            ["lora", "lora-setup", "lora-data", "lora-train", "lora-test"],
        )
        self.assertEqual(self.manifest.pages["lora"].archetype, "grid")
        self.assertEqual(self.manifest.pages["lora-train"].archetype, "console")
        self.assertEqual(self.widgets["lora-home-progress"].options.segments, 4)

        setup_form = self.widgets["lora-setup-form"]
        self.assertEqual(setup_form.action_id, "lora-setup-save")
        self.assertEqual(
            {child.id for child in setup_form.children},
            {
                "lora-project-name",
                "lora-goal",
                "lora-base-model",
                "lora-preset",
            },
        )
        self.assertEqual(self.widgets["lora-preset"].type, "select")
        self.assertEqual(len(self.widgets["lora-preset"].options), 4)
        self.assertTrue(
            all(option.description for option in self.widgets["lora-preset"].options)
        )
        self.assertTrue(
            all(option.description for option in self.widgets["lora-goal"].options)
        )
        model_options = {
            option.value: option.description
            for option in self.widgets["lora-base-model"].options
        }
        for template in main.LORA_MODEL_TEMPLATES:
            with self.subTest(model=template.model_id):
                self.assertIn(template.model_id, model_options)
                self.assertTrue(model_options[template.model_id])
        self.assertEqual(len(self.widgets["lora-model-template-table"].rows), 7)
        self.assertIn(
            "Architecture",
            self.widgets["lora-model-template-table"].headers,
        )
        self.assertIn("only need", self.widgets["lora-train-controls-help"].content.lower())
        self.assertIn("Starter range", self.widgets["lora-tuning-table"].headers)
        self.assertIn(
            "underfitting",
            self.widgets["cfg-lora-r"].options.description.lower(),
        )
        adapter_options = {
            option.value: option.description
            for option in self.widgets["cfg-adapter"].options
        }
        self.assertIn("gpu memory", adapter_options["qlora"].lower())
        data_form = self.widgets["lora-data-form"]
        self.assertEqual(data_form.action_id, "lora-data-save")
        self.assertEqual(
            {child.id for child in data_form.children},
            {"lora-data-filename", "lora-data-editor"},
        )
        self.assertTrue(self.widgets["lora-data-editor"].options.multiline)
        self.assertEqual(self.widgets["lora-data-editor"].options.rows, 12)
        example_form = self.widgets["lora-example-form"]
        self.assertEqual(example_form.action_id, "lora-example-add")
        self.assertEqual(
            {child.id for child in example_form.children},
            {
                "lora-example-user",
                "lora-example-answer",
                "lora-example-system",
            },
        )
        self.assertTrue(self.widgets["lora-data-editor-panel"].options.initial_collapsed)
        downloaded_form = self.widgets["lora-downloaded-dataset-form"]
        self.assertEqual(downloaded_form.action_id, "lora-use-downloaded-dataset")
        self.assertEqual(
            {child.id for child in downloaded_form.children},
            {
                "lora-hf-dataset",
                "lora-hf-dataset-format",
                "lora-hf-dataset-split",
                "lora-hf-dataset-subset",
            },
        )
        self.assertEqual(len(self.widgets["lora-hf-dataset-format"].options), 4)
        self.assertTrue(
            all(
                option.description
                for option in self.widgets["lora-hf-dataset-format"].options
            )
        )
        self.assertIn(
            "one dataset",
            self.widgets["lora-data-current-help"].content.lower(),
        )
        self.assertIn(
            "TRAINING WILL READ",
            self.widgets["lora-data-active-source"].content,
        )

        self.assertEqual(
            self.widgets["lora-training-log"].stream_id,
            main.LOG_AXOLOTL,
        )
        self.assertTrue(self.widgets["lora-train-progress"].options.description)
        self.assertIn(
            "long time",
            self.widgets["lora-train-start"].options.confirm,
        )

        build_form = self.widgets["lora-test-build-form"]
        self.assertEqual(build_form.action_id, "lora-test-build")
        compare_form = self.widgets["lora-test-compare-form"]
        self.assertEqual(compare_form.action_id, "lora-test-compare")
        self.assertEqual(
            {child.id for child in compare_form.children},
            {
                "lora-test-compare-base",
                "lora-test-chat-model",
                "lora-test-system",
                "lora-test-prompt",
            },
        )
        self.assertEqual(self.widgets["lora-test-log"].stream_id, main.LOG_OLLAMA)

    def test_internal_page_links_install_client_side_tab_navigation(self) -> None:
        app = main.FastAPI()

        @app.get("/")
        def root() -> str:
            return "<html><body>LCARS</body></html>"

        @app.get("/{full_path:path}")
        def fallback(full_path: str) -> str:
            return full_path

        main._install_internal_navigation(app)

        root_route = next(
            route
            for route in app.router.routes
            if getattr(route, "path", "") == "/"
            and "GET" in (getattr(route, "methods", set()) or set())
        )
        page = root_route.endpoint()
        self.assertIn(
            '<script type="module" src="/lcars/internal-navigation.js"></script>',
            page,
        )
        script_route = next(
            route
            for route in app.router.routes
            if getattr(route, "path", "") == "/lcars/internal-navigation.js"
        )
        self.assertLess(
            app.router.routes.index(script_route),
            next(
                index
                for index, route in enumerate(app.router.routes)
                if getattr(route, "path", "") == "/{full_path:path}"
            ),
        )
        script_response = script_route.endpoint()
        self.assertEqual(
            Path(script_response.path),
            main.INTERNAL_NAVIGATION_SCRIPT,
        )
        script = main.INTERNAL_NAVIGATION_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("event.preventDefault()", script)
        self.assertIn('document.querySelectorAll(".lcars-rail-btn")', script)

    def test_guided_setup_only_replaces_an_untouched_dataset_template(self) -> None:
        template = main.starter_dataset_template(
            main.LORA_GOALS[0],
            "Helpful Captain",
        )

        self.assertTrue(
            main._lora_editor_is_generated_template(template, "helpful-captain")
        )
        self.assertFalse(
            main._lora_editor_is_generated_template(
                template.replace("[EDIT ME 1:", "My carefully edited answer:"),
                "helpful-captain",
            )
        )

    def test_saving_data_preserves_known_models_top_level_chat_template(self) -> None:
        report = main.DatasetReport(
            source="./data/qwen-project.jsonl",
            source_kind="local",
            dataset_type="chat_template",
            example_count=1,
        )
        with (
            patch.object(
                main,
                "_load_config_or_empty",
                return_value={"base_model": "Qwen/Qwen3.5-4B"},
            ),
            patch.object(
                main,
                "save_chat_jsonl",
                return_value=(Path("/project/data/qwen-project.jsonl"), report),
            ),
            patch.object(main, "_workflow_blocks_config_change", return_value=False),
            patch.object(main.STATE.config_store, "apply_updates") as apply_updates,
            patch.object(main.STATE, "refresh_preflight", return_value=[]),
            patch.object(main, "_set_session_value"),
            patch.object(main, "_update_config_widgets"),
            patch.object(main, "_update_preflight_widgets"),
            patch.object(main, "_update_lora_widgets"),
            patch.object(main.lcars, "notify"),
        ):
            saved = main._lora_save_dataset_action(
                "qwen-project.jsonl",
                '{"messages": []}',
            )

        self.assertTrue(saved)
        updates = apply_updates.call_args.args[0]
        self.assertIsNone(updates["datasets.0.chat_template"])

    def test_downloaded_dataset_action_applies_cache_repo_and_shape_atomically(self) -> None:
        cached = [
            {
                "Type": "dataset",
                "Repo": "example/downloaded-chat",
                "Size": "20MB",
                "Files": "3",
                "Revision": "abcdef123456",
                "Path": "/cache/datasets--example--downloaded-chat",
            }
        ]
        with (
            patch.object(
                main,
                "_lora_downloaded_dataset_cache_rows",
                return_value=cached,
            ),
            patch.object(
                main.STATE.config_store,
                "load",
                return_value={"base_model": "Qwen/Qwen3.5-4B"},
            ),
            patch.object(main.STATE.config_store, "apply_updates") as apply_updates,
            patch.object(main.STATE, "refresh_preflight", return_value=[]),
            patch.object(main, "_workflow_blocks_config_change", return_value=False),
            patch.object(main, "_set_session_value"),
            patch.object(main, "_update_config_widgets"),
            patch.object(main, "_update_preflight_widgets"),
            patch.object(main, "_update_lora_widgets"),
            patch.object(main, "_update_lora_downloaded_dataset_widgets"),
            patch.object(main.lcars, "notify") as notify,
        ):
            main._lora_use_downloaded_dataset_action(
                "example/downloaded-chat",
                "sharegpt",
                "train[:25%]",
                "cleaned",
            )

        updates = apply_updates.call_args.args[0]
        self.assertEqual(updates["datasets.0.path"], "example/downloaded-chat")
        self.assertEqual(updates["datasets.0.split"], "train[:25%]")
        self.assertEqual(updates["datasets.0.name"], "cleaned")
        self.assertEqual(updates["datasets.0.type"], "chat_template")
        self.assertEqual(updates["datasets.0.field_messages"], "conversations")
        self.assertIsNone(updates["datasets.0.chat_template"])
        notify.assert_called_once()

    def test_incomplete_dataset_cache_is_visible_but_cannot_be_selected(self) -> None:
        cache_rows = [
            {
                "Type": "dataset",
                "Repo": "example/ready",
                "Status": "READY",
                "Size": "20MB",
                "Files": "3",
                "Revision": "abcdef123456",
                "Path": "/cache/datasets--example--ready",
                "Problem": "",
            },
            {
                "Type": "dataset",
                "Repo": "nvidia/incomplete",
                "Status": "INCOMPLETE",
                "Size": "40B",
                "Files": "1",
                "Revision": "123456789abc",
                "Path": "/cache/datasets--nvidia--incomplete",
                "Problem": "No completed snapshot exists. Download this dataset again.",
            },
        ]

        ready_rows = main._lora_downloaded_dataset_cache_rows(cache_rows)
        with patch.object(main.STATE.hf, "job_rows", return_value=[]):
            display_rows = main._lora_dataset_download_rows(cache_rows, "")

        self.assertEqual([row["Repo"] for row in ready_rows], ["example/ready"])
        incomplete = next(
            row for row in display_rows if row["Dataset"] == "nvidia/incomplete"
        )
        self.assertEqual(incomplete["Status"], "INCOMPLETE")
        self.assertIn("Download this dataset again", incomplete["What to do"])
        notice = main._lora_dataset_cache_notice(cache_rows, ready_rows)
        self.assertIn("nvidia/incomplete", notice)
        self.assertIn("cannot safely use it", notice)

    def test_easy_example_builder_replaces_the_untouched_template_then_saves(self) -> None:
        original_ctx = get_ctx()
        session_id = "lora-easy-example"
        try:
            clear_session_state(session_id)
            get_session_state(session_id)["lora-data-editor"] = main.starter_dataset_template(
                main.LORA_GOALS[0],
                "Helpful Captain",
            )
            set_ctx(
                _LCARSContext(
                    mode=Mode.HANDLE,
                    session_id=session_id,
                )
            )
            with (
                patch.object(main, "_set_widget_value") as set_value,
                patch.object(
                    main,
                    "_lora_save_dataset_action",
                    return_value=True,
                ) as save,
                patch.object(main.lcars, "notify"),
            ):
                main._lora_add_example_action(
                    project_name="helpful-captain",
                    filename="helpful-captain.jsonl",
                    user_prompt="What happens when a tool fails?",
                    ideal_response="I inspect the error, explain it briefly, and retry safely.",
                    system_prompt="Be calm.",
                )

            editor_text = set_value.call_args_list[0].args[1]
            self.assertNotIn("EDIT ME", editor_text)
            self.assertEqual(
                [message["role"] for message in json.loads(editor_text)["messages"]],
                ["system", "user", "assistant"],
            )
            save.assert_called_once_with(
                "helpful-captain.jsonl",
                editor_text,
                notify=False,
            )
        finally:
            clear_session_state(session_id)
            set_ctx(original_ctx)

    def test_manifest_uses_v44_capabilities(self) -> None:
        run = self.manifest.pages["run"]
        self.assertEqual(run.archetype, "telemetry")
        self.assertFalse(run.fillers)
        workflow = self.widgets[main.WORKFLOW_CANVAS_ID]
        self.assertEqual(workflow.type, "node_canvas")
        self.assertEqual(workflow.document.format, "lcars-node-graph")
        self.assertEqual(
            [node.template for node in workflow.document.nodes],
            ["active-config", "preprocess", "train", "evaluate"],
        )
        self.assertTrue(workflow.options.editable)
        self.assertTrue(workflow.options.snap_to_grid)
        self.assertFalse(workflow.options.minimap)
        self.assertTrue(workflow.options.allow_import_export)
        self.assertEqual(
            workflow.options.interaction.action_id,
            main.WORKFLOW_CANVAS_ID,
        )
        self.assertEqual(workflow.execution.status, "idle")
        self.assertEqual(self.widgets["workflow-graph-panel"].zone, "primary")
        self.assertEqual(self.widgets["workflow-control-panel"].zone, "side")
        self.assertEqual(self.manifest.pages["console"].archetype, "console")
        self.assertEqual(self.widgets["run-process-panel"].zone, "primary")
        self.assertEqual(self.widgets["run-controls-panel"].zone, "dock")
        self.assertNotIn("run-hardware-panel", self.widgets)

        resources = self.manifest.pages["resources"]
        self.assertEqual(resources.archetype, "console")
        self.assertFalse(resources.fillers)
        self.assertEqual(self.widgets["resource-trend-panel"].zone, "primary")
        self.assertEqual(self.widgets["resource-load-panel"].zone, "side")
        self.assertEqual(self.widgets["resource-load-panel"].span, (2, 2))
        self.assertEqual(self.widgets["resource-gpu-panel"].span, (2, 3))
        self.assertEqual(self.widgets["resource-gpu-process-panel"].span, (2, 3))
        self.assertEqual(self.widgets["resource-storage-pressure-panel"].span, (2, 3))
        self.assertFalse(self.widgets["resource-trend-panel"].options.collapsible)
        self.assertFalse(self.widgets["resource-load-panel"].options.collapsible)

        hub = self.manifest.pages["hub"]
        self.assertEqual(hub.archetype, "telemetry")
        self.assertFalse(hub.fillers)
        results_panel = self.widgets["hf-results-panel"]
        self.assertEqual(results_panel.zone, "primary")
        self.assertEqual(results_panel.weight, 12)
        self.assertEqual(results_panel.aspect, "wide")
        self.assertEqual(results_panel.group, "hf-browser")
        self.assertIsNone(results_panel.span)
        self.assertFalse(results_panel.options.collapsible)
        operations_panel = self.widgets["hf-operations-panel"]
        self.assertEqual(operations_panel.zone, "primary")
        self.assertEqual(operations_panel.aspect, "wide")
        self.assertEqual(operations_panel.group, "hf-browser")
        self.assertFalse(operations_panel.options.collapsible)
        self.assertNotIn("hf-search-panel", self.widgets)
        self.assertNotIn("hf-filter-panel", self.widgets)
        self.assertNotIn("hf-workflow-panel", self.widgets)
        self.assertEqual(self.widgets["hf-transfers-panel"].group, "content-transfers")
        self.assertIsNone(self.widgets["hf-transfers-panel"].span)
        self.assertEqual(self.widgets["hf-activity-panel"].group, "content-transfers")
        self.assertIsNone(self.widgets["hf-activity-panel"].span)
        hub_panel_ids = {
            widget.id
            for row in self.manifest.pages["hub"].rows
            for column in row.columns
            for widget in column.widgets
        }
        content_panel_ids = {
            widget.id
            for row in self.manifest.pages["content"].rows
            for column in row.columns
            for widget in column.widgets
        }
        self.assertNotIn("hf-transfers-panel", hub_panel_ids)
        self.assertNotIn("hf-activity-panel", hub_panel_ids)
        self.assertNotIn("hf-target-panel", hub_panel_ids)
        self.assertIn("hf-transfers-panel", content_panel_ids)
        self.assertIn("hf-activity-panel", content_panel_ids)

        results = self.widgets["hf-results-table"]
        self.assertTrue(results.options.expandable)
        self.assertTrue(results.options.sticky_header)
        self.assertEqual(results.options.data_mode, "client")
        self.assertTrue(results.options.emit_state_changes)
        self.assertTrue(results.options.row_click_select)
        self.assertEqual(results.options.selection.mode, "single")
        self.assertEqual(results.options.interaction.action_id, main.HF_RESULTS_TABLE_ID)
        self.assertTrue(all(column.sortable for column in results.options.columns))
        self.assertEqual(
            [column.key for column in results.options.columns],
            ["repo", "fit", "size", "files", "downloads"],
        )
        self.assertTrue(all(column.filter == "none" for column in results.options.columns))
        self.assertEqual(results.options.feedback.state, "empty")

        search = self.widgets["hf-query"]
        self.assertEqual(search.options.input_type, "search")
        self.assertEqual(search.options.commit, "enter")
        search_form = self.widgets["hf-search-form"]
        self.assertEqual(search_form.type, "form")
        self.assertEqual(search_form.action_id, "hf-search")
        self.assertEqual(
            {child.id for child in search_form.children},
            {
                "hf-query",
                "hf-query-mode",
                "hf-search-repo-type",
                "hf-revision",
            },
        )
        filter_form = self.widgets["hf-filter-form"]
        self.assertEqual(filter_form.type, "form")
        self.assertEqual(filter_form.action_id, "hf-filter-results")
        self.assertEqual(
            {child.id for child in filter_form.children},
            {
                "hf-sort",
                "hf-compatibility",
                "hf-limit",
                "hf-sift",
                "hf-artifact-filter",
                "hf-quant-filter",
                "hf-vram-limit",
                "hf-fit-filter",
            },
        )
        self.assertEqual(
            self.widgets["hf-query-mode"].value,
            main.HF_QUERY_MODE_OPTIONS[0],
        )
        self.assertEqual(
            self.widgets["run-cli-args"].options.commit,
            "blur",
        )

        selected_repo = self.widgets["hf-selected-repo-copy"]
        self.assertTrue(selected_repo.options.selectable)

        config_summary = self.widgets["config-summary-table"]
        self.assertTrue(config_summary.rows[0].cells[0].copyable)
        self.assertTrue(config_summary.rows[0].cells[1].copyable)

        log = self.widgets["axolotl-output-log"]
        self.assertTrue(log.options.toolbar)
        self.assertTrue(log.options.search)

        cache_delete = self.widgets["cache-delete"]
        self.assertIn("Permanently remove", cache_delete.options.confirm)

    def test_workflow_canvas_commits_transactional_graph_edits(self) -> None:
        original_ctx = get_ctx()
        original_workflow = main.STATE.workflow
        session_id = "workflow-graph-edit"
        main.STATE.workflow = main.WorkflowManager(
            main.STATE.config_store.active_name,
            original_workflow.document.model_dump(mode="json"),
        )
        edited_nodes = [
            node.model_copy(update={"position": (640.0, 160.0)})
            if node.template == "train"
            else node
            for node in main.STATE.workflow.document.nodes
        ]
        edited = main.STATE.workflow.document.model_copy(update={"nodes": edited_nodes})
        try:
            clear_session_state(session_id)
            set_ctx(
                _LCARSContext(
                    mode=Mode.HANDLE,
                    session_id=session_id,
                    active_action_id=main.WORKFLOW_CANVAS_ID,
                    active_action_value={
                        "kind": "move",
                        "state": {
                            "document": edited.model_dump(mode="json"),
                            "selection": ["train-2"],
                        },
                    },
                )
            )
            with (
                patch.object(main, "_persist_workflow_document") as persist,
                patch.object(main, "_update_workflow_widgets") as update,
            ):
                main._run_page()

            train = next(
                node for node in main.STATE.workflow.document.nodes if node.template == "train"
            )
            self.assertEqual(train.position, (640.0, 160.0))
            persist.assert_called_once_with()
            update.assert_called_once_with()
        finally:
            main.STATE.workflow = original_workflow
            clear_session_state(session_id)
            set_ctx(original_ctx)

    def test_active_workflow_locks_config_mutations(self) -> None:
        original_workflow = main.STATE.workflow
        main.STATE.workflow = main.WorkflowManager(
            main.STATE.config_store.active_name,
            original_workflow.document.model_dump(mode="json"),
        )
        main.STATE.workflow.status = "running"
        try:
            with (
                patch.object(main.STATE.config_store, "set_active") as set_active,
                patch.object(main.lcars, "notify") as notify,
            ):
                main._switch_config_action("other.yml")

            set_active.assert_not_called()
            notify.assert_called_once()
            self.assertIn("locked", notify.call_args.args[0])
        finally:
            main.STATE.workflow = original_workflow

    def test_config_controls_keep_defaults_and_runtime_types(self) -> None:
        cfg = main.STATE.config_store.load()
        for spec in main.FIELD_SPECS:
            widget = self.widgets[spec.widget_id]
            expected = main.STATE.config_store.control_value(spec, cfg)
            with self.subTest(field=spec.key):
                if spec.kind == "bool":
                    self.assertEqual(widget.type, "toggle")
                    self.assertIsInstance(widget.checked, bool)
                    self.assertEqual(widget.checked, expected)
                elif spec.kind == "number" and spec.optional:
                    self.assertEqual(widget.type, "text_input")
                    self.assertIsInstance(widget.value, str)
                    self.assertEqual(widget.value, expected)
                    self.assertIsNotNone(widget.options.validation.pattern)
                    self.assertIn("Optional numeric value", widget.options.description)
                elif spec.kind == "number":
                    self.assertEqual(widget.type, "number_input")
                    self.assertIsInstance(widget.value, float)
                    self.assertEqual(widget.value, expected)
                else:
                    self.assertIsInstance(widget.value, str)
                    self.assertEqual(widget.value, expected)
                if widget.type == "select":
                    self.assertIn(
                        widget.value,
                        [option.value for option in widget.options],
                    )

    def test_config_selects_label_unset_and_preserve_custom_yaml_values(self) -> None:
        spec = next(item for item in main.FIELD_SPECS if item.key == "attn_implementation")

        options = main._config_select_options(
            spec,
            "future_attention_backend",
        )

        labels = {option.value: option.label for option in options}
        self.assertEqual(labels[""], "Unset / Axolotl default")
        self.assertIn("custom YAML value", labels["future_attention_backend"])

    def test_persisted_preferences_are_complete_typed_and_validated(self) -> None:
        defaults = main._persisted_widget_defaults()
        choices = main._persisted_widget_choices()

        self.assertEqual(set(defaults), set(main.PERSISTED_WIDGET_IDS))
        self.assertEqual(
            main._normalized_persisted_widget_value(
                "hf-sort",
                "removed-sort-mode",
                defaults=defaults,
                choices=choices,
            ),
            defaults["hf-sort"],
        )
        self.assertEqual(
            main._normalized_persisted_widget_value(
                "hf-vram-limit",
                "not-a-number",
                defaults=defaults,
                choices=choices,
            ),
            defaults["hf-vram-limit"],
        )
        self.assertEqual(
            main._normalized_persisted_widget_value(
                "new-config-name",
                "../../unsafe.yml",
                defaults=defaults,
                choices=choices,
            ),
            "experiment.yml",
        )

    def test_legacy_hf_type_preference_migrates_to_search_without_retargeting(self) -> None:
        original_ctx = get_ctx()
        session_id = "legacy-hf-type-migration"

        def app_value(key: str, default: object = None) -> object:
            return "model" if key == "hf_repo_type" else default

        try:
            clear_session_state(session_id)
            set_ctx(_LCARSContext(mode=Mode.BUILD, session_id=session_id))
            with (
                patch.object(
                    main.UI_STATE,
                    "widget_values",
                    return_value={"hf-repo-type": "dataset"},
                ),
                patch.object(main.UI_STATE, "get", side_effect=app_value),
            ):
                main._hydrate_widget_state()

            state = get_session_state(session_id)
            self.assertEqual(state["hf-search-repo-type"], "dataset")
            self.assertEqual(state["hf-repo-type"], "model")
        finally:
            clear_session_state(session_id)
            set_ctx(original_ctx)

    def test_config_refresh_uses_checked_for_toggles_and_refreshes_choices(self) -> None:
        original_ctx = get_ctx()
        session_id = "config-refresh-types"
        try:
            clear_session_state(session_id)
            ctx = _LCARSContext(
                mode=Mode.HANDLE,
                session_id=session_id,
            )
            set_ctx(ctx)

            main._update_config_widgets()

            updates = {
                event.payload.id: event.payload.data
                for event in ctx.pending_events
                if event.type == "widget_update"
            }
            self.assertIn("checked", updates["cfg-load-in-8bit"])
            self.assertNotIn("value", updates["cfg-load-in-8bit"])
            self.assertTrue(updates["active-config-select"]["options"])
        finally:
            clear_session_state(session_id)
            set_ctx(original_ctx)

    def test_hf_search_hydrates_the_visible_page_before_local_filters(self) -> None:
        original_ctx = get_ctx()
        original_vram = main.STATE.hf.vram_limit_gb
        session_id = "hf-visible-hydration"
        result = SearchResult(repo_id="example/dataset", repo_type="dataset")
        try:
            clear_session_state(session_id)
            set_ctx(
                _LCARSContext(
                    mode=Mode.HANDLE,
                    session_id=session_id,
                    active_action_id="hf-search",
                )
            )
            with (
                patch.object(main.STATE.hf, "search", return_value=[result]),
                patch.object(main.STATE.hf, "hydrate_results", return_value=1) as hydrate,
                patch.object(main.STATE.hf, "sift_results", return_value=[result]),
                patch.object(main, "_update_hf_widgets"),
                patch.object(main, "_append_hf_logs"),
            ):
                main._hf_search_action(
                    "example",
                    "dataset",
                    vram_limit=24,
                )

            hydrate.assert_called_once_with(
                [result],
                limit=main.HF_RESULTS_PAGE_SIZE,
            )
        finally:
            main.STATE.hf.vram_limit_gb = original_vram
            clear_session_state(session_id)
            set_ctx(original_ctx)

    def test_hf_search_form_submits_core_values_with_saved_filters(self) -> None:
        original_ctx = get_ctx()
        original_vram = main.STATE.hf.vram_limit_gb
        session_id = "hf-atomic-form"
        payload = {
            "hf-query": "atomic dataset query",
            "hf-query-mode": main.HF_QUERY_MODE_OPTIONS[0],
            "hf-search-repo-type": "dataset",
            "hf-revision": "",
        }
        try:
            clear_session_state(session_id)
            get_session_state(session_id).update(
                {
                    "hf-sort": "likes",
                    "hf-compatibility": "include warnings and blocked",
                    "hf-limit": "25",
                }
            )
            with (
                patch.object(main.STATE.hf, "search", return_value=[]) as search,
                patch.object(main.STATE.hf, "hydrate_results", return_value=0),
                patch.object(main.STATE.hf, "sift_results", return_value=[]),
                patch.object(main, "_persist_widget_state"),
            ):
                app = main.create_lcars_app(main.build_ui)
                handler = app.state.plugin_action_handlers["*"]
                asyncio.run(handler("hf-search", payload, session_id))

            search.assert_called_once_with(
                "atomic dataset query",
                "dataset",
                sort="likes",
                compatible_only=False,
                limit=25,
            )
            self.assertEqual(
                get_session_state(session_id)["hf-query"],
                "atomic dataset query",
            )
            self.assertEqual(
                get_session_state(session_id)["hf-search-repo-type"],
                "dataset",
            )
            self.assertEqual(get_session_state(session_id)["hf-repo-type"], "dataset")
        finally:
            main.STATE.hf.vram_limit_gb = original_vram
            clear_session_state(session_id)
            set_ctx(original_ctx)

    def test_hf_filter_form_refreshes_search_with_atomic_filter_values(self) -> None:
        original_ctx = get_ctx()
        original_vram = main.STATE.hf.vram_limit_gb
        session_id = "hf-atomic-filters"
        payload = {
            "hf-sort": "likes",
            "hf-compatibility": "include warnings and blocked",
            "hf-limit": "25",
            "hf-sift": "safetensors",
            "hf-artifact-filter": "base/trainable models",
            "hf-quant-filter": "Transformers safetensors",
            "hf-vram-limit": 16,
            "hf-fit-filter": "fits vram",
        }
        try:
            clear_session_state(session_id)
            get_session_state(session_id).update(
                {
                    "hf-query": "atomic model query",
                    "hf-search-repo-type": "model",
                }
            )
            with (
                patch.object(main.STATE.hf, "search", return_value=[]) as search,
                patch.object(main.STATE.hf, "hydrate_results", return_value=0),
                patch.object(main.STATE.hf, "sift_results", return_value=[]) as sift,
                patch.object(main, "_persist_widget_state"),
            ):
                app = main.create_lcars_app(main.build_ui)
                handler = app.state.plugin_action_handlers["*"]
                asyncio.run(handler("hf-filter-results", payload, session_id))

            search.assert_called_once_with(
                "atomic model query",
                "model",
                sort="likes",
                compatible_only=False,
                limit=25,
            )
            sift.assert_called_once_with(
                text="safetensors",
                sort=main.STATE.hf.local_sort,
                descending=main._kept_sort_direction(main.STATE.hf.local_sort),
                artifact_filter="base/trainable models",
                quant_filter="Transformers safetensors",
                fit_filter="fits vram",
                vram_limit_gb=16.0,
            )
        finally:
            main.STATE.hf.vram_limit_gb = original_vram
            clear_session_state(session_id)
            set_ctx(original_ctx)

    def test_hf_query_form_inspects_atomic_exact_repository_values(self) -> None:
        original_ctx = get_ctx()
        session_id = "hf-atomic-exact-lookup"
        payload = {
            "hf-query": "example/direct-dataset",
            "hf-query-mode": main.HF_QUERY_MODE_OPTIONS[1],
            "hf-search-repo-type": "dataset",
            "hf-revision": "refs/pr/12",
        }
        try:
            clear_session_state(session_id)
            with (
                patch.object(main, "_hf_inspect_action") as inspect,
                patch.object(main, "_persist_widget_state"),
            ):
                app = main.create_lcars_app(main.build_ui)
                handler = app.state.plugin_action_handlers["*"]
                asyncio.run(handler("hf-search", payload, session_id))

            inspect.assert_called_once_with(
                "example/direct-dataset",
                "dataset",
                "refs/pr/12",
            )
        finally:
            clear_session_state(session_id)
            set_ctx(original_ctx)

    def test_hf_repository_actions_use_target_type_not_search_type(self) -> None:
        original_ctx = get_ctx()
        session_id = "hf-independent-target-type"
        try:
            clear_session_state(session_id)
            state = get_session_state(session_id)
            state.update(
                {
                    "hf-search-repo-type": "dataset",
                    "hf-repo-type": "model",
                    "hf-repo-id": "example/model",
                    "hf-revision": "",
                }
            )
            set_ctx(
                _LCARSContext(
                    mode=Mode.HANDLE,
                    session_id=session_id,
                    active_action_id="hf-use-repo",
                )
            )
            with patch.object(main, "_hf_use_repo_action") as use_repo:
                main._hub_page()

            use_repo.assert_called_once_with("example/model", "model")
        finally:
            clear_session_state(session_id)
            set_ctx(original_ctx)

    def test_hf_rows_keep_typed_values_and_native_actions(self) -> None:
        original_results = main.STATE.hf.search_results
        original_details = main.STATE.hf.selected_details
        original_detail_cache = dict(main.STATE.hf.repo_details)
        original_errors = dict(main.STATE.hf.inspection_errors)
        original_expanded = list(main.STATE.hf.expanded_result_ids)
        original_repo_id = main.STATE.hf.last_repo_id
        original_repo_type = main.STATE.hf.last_repo_type
        try:
            result = SearchResult(
                repo_id="example/model",
                repo_type="model",
                downloads=12_345,
                likes=67,
                updated="2026-07-23",
                file_count=8,
                size_bytes=1024,
                fit="fits 24GB",
                weights="Safetensors",
                compatibility="OK: Transformers weights",
                tags="text-generation, llama",
            )
            main.STATE.hf.search_results = [result]
            main.STATE.hf.selected_details = None
            main.STATE.hf.repo_details = {}
            main.STATE.hf.inspection_errors = {}
            main.STATE.hf.last_repo_id = result.repo_id
            main.STATE.hf.last_repo_type = result.repo_type
            main.STATE.hf.expanded_result_ids = [main._hf_result_row_id(result)]
            with patch.object(main, "_hf_configured_repositories", return_value=set()):
                row = main._hf_result_rows()[0]
                options = main._hf_result_table_options()
        finally:
            main.STATE.hf.search_results = original_results
            main.STATE.hf.selected_details = original_details
            main.STATE.hf.repo_details = original_detail_cache
            main.STATE.hf.inspection_errors = original_errors
            main.STATE.hf.expanded_result_ids = original_expanded
            main.STATE.hf.last_repo_id = original_repo_id
            main.STATE.hf.last_repo_type = original_repo_type

        repo_cell = row.cells[0]
        self.assertEqual(repo_cell.link.href, "https://huggingface.co/example/model")
        self.assertIsNone(repo_cell.action)
        self.assertTrue(repo_cell.copyable)
        self.assertEqual(repo_cell.copy_value, result.repo_id)
        self.assertEqual(repo_cell.status, "ok")
        self.assertEqual(row.cells[4], 12_345)
        self.assertEqual(len(row.cells), 5)
        self.assertTrue(row.loading)
        self.assertFalse(row.children)
        self.assertEqual(options.feedback.state, "ready")
        self.assertEqual(options.selection.selected_ids, [row.id])
        self.assertIn(row.id, options.expanded_ids)
        action_ids = {
            item.action_id
            for item in row.expanded_content
            if isinstance(item, lcars.TableDetailAction)
        }
        self.assertIn("hf-inspect-row", action_ids)
        self.assertIn("hf-use-row", action_ids)
        self.assertIn("hf-related-row", action_ids)
        metadata = next(
            item.text
            for item in row.expanded_content
            if isinstance(item, lcars.TableDetailText) and "downloads" in item.text
        )
        self.assertIn("67 likes", metadata)
        self.assertIn("updated 2026-07-23", metadata)

    def test_hf_expansion_exposes_inspected_file_actions_and_config_marker(self) -> None:
        original_results = main.STATE.hf.search_results
        original_details = main.STATE.hf.selected_details
        original_detail_cache = dict(main.STATE.hf.repo_details)
        original_repo_id = main.STATE.hf.last_repo_id
        original_repo_type = main.STATE.hf.last_repo_type
        result = SearchResult(
            repo_id="example/dataset",
            repo_type="dataset",
            file_count=1,
            compatibility="OK: datasets-compatible files",
            role="dataset",
        )
        try:
            main.STATE.hf.search_results = [result]
            details = RepoDetails(
                result=result,
                files=[
                    RepoFile(
                        path="train/data.jsonl",
                        size=2048,
                        kind="dataset",
                        axolotl="data",
                    )
                ],
            )
            main.STATE.hf.selected_details = details
            main.STATE.hf.repo_details = {(result.repo_type, result.repo_id): details}
            main.STATE.hf.last_repo_id = result.repo_id
            main.STATE.hf.last_repo_type = result.repo_type
            with patch.object(
                main,
                "_hf_configured_repositories",
                return_value={("dataset", result.repo_id)},
            ):
                row = main._hf_result_rows()[0]
                options = main._hf_result_table_options()
        finally:
            main.STATE.hf.search_results = original_results
            main.STATE.hf.selected_details = original_details
            main.STATE.hf.repo_details = original_detail_cache
            main.STATE.hf.last_repo_id = original_repo_id
            main.STATE.hf.last_repo_type = original_repo_type

        self.assertTrue(row.cells[0].display.startswith("◆● "))
        self.assertEqual(row.cells[0].status, "ok")
        self.assertFalse(row.loading)
        detail_table = next(
            item for item in row.expanded_content if isinstance(item, lcars.TableDetailTable)
        )
        file_row = detail_table.rows[0]
        self.assertTrue(file_row.cells[0].copyable)
        self.assertEqual(file_row.cells[1].value, 2048)
        self.assertEqual(file_row.cells[1].display, "2.0KB")
        self.assertEqual(file_row.cells[4].action.action_id, "hf-download-file")
        self.assertEqual(options.selection.selected_ids, [row.id])

    def test_hf_table_events_select_and_lazy_inspect_by_stable_row_id(self) -> None:
        original_ctx = get_ctx()
        original_results = main.STATE.hf.search_results
        original_details = main.STATE.hf.selected_details
        original_detail_cache = dict(main.STATE.hf.repo_details)
        original_expanded = list(main.STATE.hf.expanded_result_ids)
        original_repo_id = main.STATE.hf.last_repo_id
        original_repo_type = main.STATE.hf.last_repo_type
        result = SearchResult(repo_id="example/model", repo_type="model")
        row_id = main._hf_result_row_id(result)
        try:
            main.STATE.hf.search_results = [result]
            main.STATE.hf.selected_details = None
            main.STATE.hf.repo_details = {}
            main.STATE.hf.expanded_result_ids = []
            set_ctx(
                _LCARSContext(
                    mode=Mode.HANDLE,
                    session_id="table-selection",
                    active_action_id=main.HF_RESULTS_TABLE_ID,
                    active_action_value={
                        "kind": "selection",
                        "state": {
                            "selected_ids": [row_id],
                            "expanded_ids": [],
                        },
                    },
                )
            )
            with patch.object(main, "_update_hf_widgets"):
                main._handle_hf_table_action()
            self.assertEqual(main.STATE.hf.last_repo_id, result.repo_id)
            self.assertEqual(main.STATE.hf.last_repo_type, result.repo_type)

            set_ctx(
                _LCARSContext(
                    mode=Mode.HANDLE,
                    session_id="table-expansion",
                    active_action_id=main.HF_RESULTS_TABLE_ID,
                    active_action_value={
                        "kind": "expansion",
                        "state": {
                            "selected_ids": [row_id],
                            "expanded_ids": [row_id],
                        },
                    },
                )
            )
            with patch.object(main, "_hf_inspect_action") as inspect:
                main._handle_hf_table_action()
            inspect.assert_called_once_with(result.repo_id, result.repo_type, "")
            self.assertEqual(main.STATE.hf.expanded_result_ids, [row_id])
        finally:
            set_ctx(original_ctx)
            main.STATE.hf.search_results = original_results
            main.STATE.hf.selected_details = original_details
            main.STATE.hf.repo_details = original_detail_cache
            main.STATE.hf.expanded_result_ids = original_expanded
            main.STATE.hf.last_repo_id = original_repo_id
            main.STATE.hf.last_repo_type = original_repo_type

    def test_hf_table_page_event_hydrates_the_new_visible_slice(self) -> None:
        original_ctx = get_ctx()
        original_results = main.STATE.hf.search_results
        original_expanded = list(main.STATE.hf.expanded_result_ids)
        results = [
            SearchResult(repo_id=f"example/dataset-{index}", repo_type="dataset")
            for index in range(15)
        ]
        try:
            main.STATE.hf.search_results = results
            main.STATE.hf.expanded_result_ids = []
            set_ctx(
                _LCARSContext(
                    mode=Mode.HANDLE,
                    session_id="table-page-hydration",
                    active_action_id=main.HF_RESULTS_TABLE_ID,
                    active_action_value={
                        "kind": "page",
                        "state": {
                            "page": 2,
                            "page_size": 10,
                            "expanded_ids": [],
                        },
                    },
                )
            )
            with (
                patch.object(main.STATE.hf, "hydrate_results", return_value=5) as hydrate,
                patch.object(main, "_update_hf_widgets") as update,
                patch.object(main, "_append_hf_logs"),
            ):
                main._handle_hf_table_action()

            hydrate.assert_called_once_with(results[10:15], limit=10)
            update.assert_called_once_with()
        finally:
            set_ctx(original_ctx)
            main.STATE.hf.search_results = original_results
            main.STATE.hf.expanded_result_ids = original_expanded

    def test_empty_dataset_search_does_not_reset_repo_type_to_model(self) -> None:
        original_ctx = get_ctx()
        original_all_results = main.STATE.hf.all_search_results
        original_results = main.STATE.hf.search_results
        original_related_results = main.STATE.hf.related_results
        original_related_repo_id = main.STATE.hf.related_repo_id
        original_details = main.STATE.hf.selected_details
        original_expanded = list(main.STATE.hf.expanded_result_ids)
        original_repo_id = main.STATE.hf.last_repo_id
        original_repo_type = main.STATE.hf.last_repo_type
        stale = SearchResult(repo_id="example/model", repo_type="model")
        session_id = "empty-dataset-search"
        try:
            main.STATE.hf.all_search_results = [stale]
            main.STATE.hf.search_results = [stale]
            main.STATE.hf.selected_details = None
            main.STATE.hf.expanded_result_ids = []
            main.STATE.hf.last_repo_id = stale.repo_id
            main.STATE.hf.last_repo_type = stale.repo_type
            clear_session_state(session_id)
            set_ctx(
                _LCARSContext(
                    mode=Mode.HANDLE,
                    session_id=session_id,
                    active_action_id="hf-search",
                )
            )
            with patch.object(main.STATE.hf, "_list_datasets", return_value=[]):
                main._hf_search_action("no dataset matches", "dataset")

            self.assertEqual(main.STATE.hf.last_repo_type, "dataset")
            self.assertEqual(main.STATE.hf.last_repo_id, "")
            self.assertEqual(
                get_session_state(session_id)["hf-repo-type"],
                "dataset",
            )
            self.assertEqual(
                get_session_state(session_id)["hf-search-repo-type"],
                "dataset",
            )
            self.assertEqual(get_session_state(session_id)["hf-repo-id"], "")
        finally:
            clear_session_state(session_id)
            set_ctx(original_ctx)
            main.STATE.hf.all_search_results = original_all_results
            main.STATE.hf.search_results = original_results
            main.STATE.hf.related_results = original_related_results
            main.STATE.hf.related_repo_id = original_related_repo_id
            main.STATE.hf.selected_details = original_details
            main.STATE.hf.expanded_result_ids = original_expanded
            main.STATE.hf.last_repo_id = original_repo_id
            main.STATE.hf.last_repo_type = original_repo_type

    def test_selected_repo_uses_native_copy_and_link_options(self) -> None:
        options = main._hf_selected_text_options("example/model", "model")

        self.assertTrue(options.copyable)
        self.assertTrue(options.selectable)
        self.assertEqual(options.link.href, "https://huggingface.co/example/model")


if __name__ == "__main__":
    unittest.main()

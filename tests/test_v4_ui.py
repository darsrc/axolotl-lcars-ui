from __future__ import annotations

import asyncio
import unittest
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


class V42UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = main._build_manifest(main.build_ui, get_ctx().config)
        cls.widgets = _manifest_widgets(cls.manifest)

    def test_project_builds_with_lcars_v42(self) -> None:
        self.assertEqual(lcars.__version__, "4.2.0")
        self.assertEqual(len(self.manifest.pages), 12)

    def test_manifest_uses_v42_capabilities(self) -> None:
        hub = self.manifest.pages["hub"]
        self.assertEqual(hub.archetype, "grid")
        self.assertFalse(hub.fillers)
        results_panel = self.widgets["hf-results-panel"]
        self.assertEqual(results_panel.weight, 12)
        self.assertEqual(results_panel.aspect, "wide")
        self.assertEqual(results_panel.group, "hf-discovery")
        self.assertEqual(results_panel.span, (4, 5))
        self.assertEqual(self.widgets["hf-search-panel"].group, "hf-discovery")
        self.assertEqual(self.widgets["hf-search-panel"].span, (2, 6))
        self.assertEqual(self.widgets["hf-filter-panel"].group, "hf-discovery")
        self.assertEqual(self.widgets["hf-filter-panel"].span, (2, 6))
        self.assertEqual(self.widgets["hf-target-panel"].group, "hf-selection")
        self.assertEqual(self.widgets["hf-target-panel"].span, (2, 4))
        self.assertEqual(self.widgets["hf-workflow-panel"].group, "hf-selection")
        self.assertEqual(self.widgets["hf-workflow-panel"].span, (2, 5))
        self.assertEqual(self.widgets["hf-transfers-panel"].group, "hf-transfers")
        self.assertEqual(self.widgets["hf-transfers-panel"].span, (4, 3))
        self.assertEqual(self.widgets["hf-activity-panel"].group, "hf-transfers")

        results = self.widgets["hf-results-table"]
        self.assertTrue(results.options.expandable)
        self.assertTrue(results.options.sticky_header)
        self.assertEqual(results.options.data_mode, "client")
        self.assertTrue(results.options.emit_state_changes)
        self.assertTrue(results.options.row_click_select)
        self.assertEqual(results.options.selection.mode, "single")
        self.assertEqual(results.options.interaction.action_id, main.HF_RESULTS_TABLE_ID)
        self.assertTrue(all(column.sortable for column in results.options.columns))
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
                "hf-search-repo-type",
                "hf-sort",
                "hf-compatibility",
                "hf-limit",
            },
        )
        self.assertNotEqual(
            self.widgets["hf-search-repo-type"].id,
            self.widgets["hf-repo-type"].id,
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
        spec = next(
            item
            for item in main.FIELD_SPECS
            if item.key == "attn_implementation"
        )

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

    def test_hf_search_form_submits_visible_values_atomically(self) -> None:
        original_ctx = get_ctx()
        original_vram = main.STATE.hf.vram_limit_gb
        session_id = "hf-atomic-form"
        payload = {
            "hf-query": "atomic dataset query",
            "hf-search-repo-type": "dataset",
            "hf-sort": "likes",
            "hf-compatibility": "include warnings and blocked",
            "hf-limit": "25",
        }
        try:
            clear_session_state(session_id)
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
        self.assertEqual(row.cells[5], 67)
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

        self.assertIn("CONFIGURED", row.cells[0].display)
        self.assertIn("MANIFEST", row.cells[0].display)
        self.assertEqual(row.cells[0].status, "ok")
        self.assertFalse(row.loading)
        detail_table = next(
            item
            for item in row.expanded_content
            if isinstance(item, lcars.TableDetailTable)
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

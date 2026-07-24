from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from axolotl_lcars_ui.hf_manager import HuggingFaceManager, SearchResult


class HuggingFaceManagerV41Tests(unittest.TestCase):
    def test_inspection_is_cached_and_reconciles_search_metadata(self) -> None:
        manager = HuggingFaceManager()
        lightweight = SearchResult(repo_id="example/model", repo_type="model")
        manager.all_search_results = [lightweight]
        manager.search_results = [lightweight]
        manager.api.model_info = Mock(
            return_value=SimpleNamespace(
                id="example/model",
                downloads=120,
                likes=7,
                tags=["transformers"],
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=512),
                    SimpleNamespace(rfilename="model.safetensors", size=2048),
                ],
                pipeline_tag="text-generation",
                library_name="transformers",
                last_modified="2026-07-24",
                gated=False,
                sha="abcdef1234567890",
            )
        )

        details = manager.inspect_repo("example/model", "model")

        self.assertIsNotNone(details)
        assert details is not None
        self.assertIs(manager.details_for("example/model", "model"), details)
        self.assertIs(manager.search_results[0], details.result)
        self.assertEqual(details.result.file_count, 2)
        self.assertEqual(details.result.weight_bytes, 2048)
        self.assertEqual(manager.inspection_error_for("example/model", "model"), "")

    def test_inspection_error_is_available_to_expandable_row_retry_ui(self) -> None:
        manager = HuggingFaceManager()
        manager.api.model_info = Mock(side_effect=RuntimeError("repository unavailable"))

        details = manager.inspect_repo("example/missing", "model")

        self.assertIsNone(details)
        self.assertEqual(
            manager.inspection_error_for("example/missing", "model"),
            "repository unavailable",
        )


if __name__ == "__main__":
    unittest.main()

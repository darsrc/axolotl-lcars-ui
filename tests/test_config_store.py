from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from axolotl_lcars_ui.config_store import FIELD_SPECS, ConfigStore


def _spec(key: str):
    return next(spec for spec in FIELD_SPECS if spec.key == key)


class ConfigControlValueTests(unittest.TestCase):
    def test_control_values_have_stable_runtime_types_and_safe_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConfigStore(Path(temp_dir))
            cfg = {
                "load_in_8bit": "false",
                "fp16": "yes",
                "sequence_len": "not-a-number",
                "dataset_processes": 8,
                "adam_epsilon": 0.00000001,
                "attn_implementation": "future_attention_backend",
            }

            self.assertIs(
                store.control_value(_spec("load_in_8bit"), cfg),
                False,
            )
            self.assertEqual(
                store.control_value(_spec("fp16"), cfg),
                "true",
            )
            self.assertEqual(
                store.control_value(_spec("sequence_len"), cfg),
                2048.0,
            )
            self.assertEqual(
                store.control_value(_spec("dataset_processes"), cfg),
                "8",
            )
            self.assertEqual(
                store.control_value(_spec("adam_epsilon"), cfg),
                "0.00000001",
            )
            self.assertEqual(
                store.control_value(_spec("attn_implementation"), cfg),
                "future_attention_backend",
            )

    def test_missing_config_values_use_declared_control_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConfigStore(Path(temp_dir))
            cfg: dict[str, object] = {}

            self.assertEqual(
                store.control_value(_spec("model_type"), cfg),
                "AutoModelForCausalLM",
            )
            self.assertEqual(
                store.control_value(_spec("learning_rate"), cfg),
                0.0001,
            )
            self.assertEqual(
                store.control_value(_spec("dataset_processes"), cfg),
                "",
            )


if __name__ == "__main__":
    unittest.main()

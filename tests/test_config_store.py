from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from axolotl_lcars_ui.config_store import FIELD_SPECS, ConfigStore
from axolotl_lcars_ui.lora_studio import (
    beginner_config_updates,
    downloaded_dataset_config_updates,
)


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

    def test_switching_model_templates_clears_stale_architecture_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConfigStore(Path(temp_dir))

            store.apply_updates(
                beginner_config_updates(
                    "switchable",
                    base_model="google/gemma-4-E2B-it",
                    preset="balanced",
                )
            )
            gemma = store.load()
            self.assertNotIn("model_type", gemma)
            self.assertNotIn("tokenizer_type", gemma)
            self.assertEqual(gemma["chat_template"], "gemma4")
            self.assertEqual(gemma["eot_tokens"], ["<turn|>"])
            self.assertIsInstance(gemma["lora_target_modules"], str)
            self.assertNotIn("chat_template", gemma["datasets"][0])

            store.apply_updates(
                beginner_config_updates(
                    "switchable",
                    base_model="Qwen/Qwen3.5-4B",
                    preset="balanced",
                )
            )
            qwen = store.load()
            self.assertEqual(qwen["chat_template"], "qwen3_5")
            self.assertNotIn("eot_tokens", qwen)
            self.assertIsInstance(qwen["lora_target_modules"], list)
            self.assertIn("linear_attn.in_proj_z", qwen["lora_target_modules"])

            store.apply_updates(
                beginner_config_updates(
                    "switchable",
                    base_model="unsloth/Llama-3.2-1B-Instruct",
                    preset="balanced",
                )
            )
            legacy = store.load()
            self.assertEqual(legacy["model_type"], "AutoModelForCausalLM")
            self.assertEqual(legacy["tokenizer_type"], "AutoTokenizer")
            self.assertNotIn("chat_template", legacy)
            self.assertNotIn("eot_tokens", legacy)
            self.assertNotIn("lora_target_modules", legacy)
            self.assertTrue(legacy["lora_target_linear"])
            self.assertEqual(legacy["datasets"][0]["chat_template"], "tokenizer_default")

    def test_downloaded_dataset_shape_replaces_stale_local_jsonl_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConfigStore(Path(temp_dir))
            store.apply_updates(
                {
                    "datasets.0.path": "./data/old.jsonl",
                    "datasets.0.type": "completion",
                    "datasets.0.ds_type": "json",
                    "datasets.0.field": "old_text",
                    "datasets.0.field_messages": "old_messages",
                    "datasets.0.data_files": ["old.jsonl"],
                }
            )

            store.apply_updates(
                downloaded_dataset_config_updates(
                    "example/downloaded-chat",
                    "sharegpt",
                    subset="cleaned",
                )
            )
            dataset = store.load()["datasets"][0]

            self.assertEqual(dataset["path"], "example/downloaded-chat")
            self.assertEqual(dataset["type"], "chat_template")
            self.assertEqual(dataset["split"], "train")
            self.assertEqual(dataset["name"], "cleaned")
            self.assertEqual(dataset["field_messages"], "conversations")
            self.assertEqual(
                dataset["message_property_mappings"],
                {"role": "from", "content": "value"},
            )
            self.assertEqual(dataset["roles_to_train"], ["assistant"])
            self.assertNotIn("ds_type", dataset)
            self.assertNotIn("field", dataset)
            self.assertNotIn("data_files", dataset)


if __name__ == "__main__":
    unittest.main()

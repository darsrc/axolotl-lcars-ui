from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from axolotl_lcars_ui.lora_studio import (
    LORA_GOALS,
    LORA_MEMORY_PROFILES,
    LoraStudioError,
    beginner_config_updates,
    discover_adapter_artifacts,
    inspect_configured_dataset,
    inspect_jsonl_text,
    normalize_project_name,
    save_chat_jsonl,
    starter_dataset_template,
    suggested_ollama_model,
)


def _chat_line(user: str = "Hello", assistant: str = "Hello there.") -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }
    )


class LoraStudioTests(unittest.TestCase):
    def test_beginner_profiles_translate_to_typed_axolotl_updates(self) -> None:
        balanced = beginner_config_updates(
            "Helpful Captain",
            base_model="unsloth/Llama-3.2-1B-Instruct",
            memory_profile=LORA_MEMORY_PROFILES[0],
        )
        self.assertEqual(normalize_project_name("Helpful Captain"), "helpful-captain")
        self.assertEqual(balanced["adapter"], "lora")
        self.assertTrue(balanced["load_in_8bit"])
        self.assertFalse(balanced["load_in_4bit"])
        self.assertEqual(balanced["datasets.0.type"], "chat_template")
        self.assertEqual(balanced["output_dir"], "./outputs/helpful-captain")

        qlora = beginner_config_updates(
            "helpful-captain",
            base_model="unsloth/Llama-3.2-1B-Instruct",
            memory_profile=LORA_MEMORY_PROFILES[1],
        )
        self.assertEqual(qlora["adapter"], "qlora")
        self.assertFalse(qlora["load_in_8bit"])
        self.assertTrue(qlora["load_in_4bit"])
        self.assertEqual(qlora["micro_batch_size"], 1)

    def test_starter_template_is_valid_jsonl_but_remains_an_explicit_draft(self) -> None:
        text = starter_dataset_template(LORA_GOALS[1], "Pathfinder")
        report = inspect_jsonl_text(text)

        self.assertEqual(report.example_count, 6)
        self.assertEqual(report.message_count, 18)
        self.assertGreater(report.placeholder_count, 0)
        self.assertFalse(report.ready)
        self.assertEqual(report.status, "DRAFT")

    def test_jsonl_shape_errors_are_line_specific(self) -> None:
        report = inspect_jsonl_text(
            _chat_line() + "\n" + "{not json}",
        )

        self.assertFalse(report.ready)
        self.assertIn("Line 2", report.errors[0])

    def test_dataset_save_is_confined_and_preserves_a_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target, report = save_chat_jsonl(root, "persona.jsonl", _chat_line())
            self.assertTrue(report.ready)
            self.assertEqual(target, root / "data" / "persona.jsonl")

            replacement = _chat_line("New prompt", "New ideal response")
            save_chat_jsonl(root, "persona.jsonl", replacement)
            self.assertEqual(target.read_text(encoding="utf-8").strip(), replacement)
            self.assertTrue((root / "data" / "persona.jsonl.bak").is_file())

            with self.assertRaises(LoraStudioError):
                save_chat_jsonl(root, "../escape.jsonl", _chat_line())

    def test_configured_dataset_and_adapter_artifacts_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path, _ = save_chat_jsonl(root, "persona.jsonl", _chat_line())
            output = root / "outputs" / "persona"
            output.mkdir(parents=True)
            (output / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": (
                            "unsloth/Llama-3.2-1B-Instruct"
                        )
                    }
                ),
                encoding="utf-8",
            )
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            cfg = {
                "datasets": [
                    {
                        "path": f"./data/{dataset_path.name}",
                        "type": "chat_template",
                        "field_messages": "messages",
                    }
                ],
                "output_dir": "./outputs/persona",
            }

            report = inspect_configured_dataset(root, cfg)
            artifacts = discover_adapter_artifacts(root, cfg)

            self.assertTrue(report.ready)
            self.assertEqual(report.example_count, 1)
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(
                artifacts[0].base_model,
                "unsloth/Llama-3.2-1B-Instruct",
            )

    def test_ollama_suggestion_prefers_the_training_family(self) -> None:
        selected = suggested_ollama_model(
            "unsloth/Llama-3.2-1B-Instruct",
            ["gemma2:2b", "llama3.2:1b"],
        )
        self.assertEqual(selected, "llama3.2:1b")
        self.assertEqual(
            suggested_ollama_model(
                "unsloth/Llama-3.2-1B-Instruct",
                ["llama-guard3:8b", "llama3.2:3b"],
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()

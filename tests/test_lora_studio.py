from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from axolotl_lcars_ui.lora_studio import (
    DEFAULT_LORA_PRESET,
    LORA_GOALS,
    LORA_MEMORY_PROFILES,
    LORA_MODEL_TEMPLATES,
    LORA_PRESETS,
    LORA_TUNING_HINTS,
    LoraStudioError,
    beginner_config_updates,
    chat_example_line,
    discover_adapter_artifacts,
    get_lora_model_template,
    infer_lora_preset,
    inspect_configured_dataset,
    inspect_jsonl_text,
    recommend_lora_preset,
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

    def test_smart_presets_are_coherent_and_include_beginner_safety_defaults(self) -> None:
        for preset in LORA_PRESETS:
            with self.subTest(preset=preset.key):
                updates = beginner_config_updates(
                    "smart-project",
                    base_model="unsloth/Llama-3.2-1B-Instruct",
                    preset=preset.key,
                )
                self.assertNotEqual(updates["load_in_8bit"], updates["load_in_4bit"])
                self.assertEqual(
                    updates["adapter"] == "qlora",
                    updates["load_in_4bit"],
                )
                self.assertEqual(updates["lora_alpha"], updates["lora_r"] * 2)
                self.assertEqual(updates["datasets.0.roles_to_train"], "assistant")
                self.assertTrue(updates["lora_target_linear"])
                self.assertFalse(updates["train_on_inputs"])
                self.assertEqual(updates["attn_implementation"], "sdpa")

    def test_hardware_recommendation_and_existing_config_inference(self) -> None:
        self.assertEqual(
            recommend_lora_preset(8, "meta-llama/Llama-3.2-3B-Instruct"),
            "low-vram",
        )
        self.assertEqual(
            recommend_lora_preset(16, "meta-llama/Llama-3.2-3B-Instruct"),
            DEFAULT_LORA_PRESET,
        )
        self.assertEqual(
            recommend_lora_preset(24, "meta-llama/Llama-3.2-3B-Instruct"),
            "high-detail",
        )
        self.assertEqual(
            recommend_lora_preset(16, "mistralai/Mistral-7B-Instruct-v0.3"),
            "low-vram",
        )
        self.assertEqual(
            infer_lora_preset(
                {
                    "adapter": "lora",
                    "lora_r": 8,
                    "sequence_len": 1024,
                    "num_epochs": 1,
                }
            ),
            "quick-check",
        )
        self.assertGreaterEqual(len(LORA_TUNING_HINTS), 8)

    def test_requested_qwen_and_gemma_templates_use_official_checkpoint_ids(self) -> None:
        expected = {
            "Qwen/Qwen3.5-2B",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.6-27B",
            "Qwen/Qwen3.6-35B-A3B",
            "google/gemma-4-E2B-it",
            "google/gemma-4-E4B-it",
        }

        self.assertEqual({template.model_id for template in LORA_MODEL_TEMPLATES}, expected)
        self.assertEqual(len(LORA_MODEL_TEMPLATES), 7)
        self.assertEqual(
            get_lora_model_template(" qwen/qwen3.6-35b-a3b ").architecture,
            "35B total / 3B active MoE",
        )

    def test_model_templates_apply_architecture_specific_text_lora_defaults(self) -> None:
        for template in LORA_MODEL_TEMPLATES:
            with self.subTest(model=template.model_id):
                updates = beginner_config_updates(
                    "model-template",
                    base_model=template.model_id,
                    preset=template.default_preset,
                )
                self.assertIsNone(updates["model_type"])
                self.assertIsNone(updates["tokenizer_type"])
                self.assertIsNone(updates["datasets.0.chat_template"])
                self.assertFalse(updates["lora_target_linear"])
                self.assertFalse(updates["pad_to_sequence_len"])
                self.assertEqual(
                    updates["gradient_checkpointing_kwargs"],
                    {"use_reentrant": False},
                )

                if template.family == "Gemma 4":
                    self.assertEqual(updates["chat_template"], "gemma4")
                    self.assertEqual(updates["eot_tokens"], ("<turn|>",))
                    self.assertIsInstance(updates["lora_target_modules"], str)
                    self.assertIn("model.language_model.layers", updates["lora_target_modules"])
                else:
                    self.assertEqual(updates["chat_template"], "qwen3_5")
                    self.assertFalse(updates["sample_packing"])
                    self.assertIn(
                        "linear_attn.in_proj_qkv",
                        updates["lora_target_modules"],
                    )

        moe = beginner_config_updates(
            "moe-template",
            base_model="Qwen/Qwen3.6-35B-A3B",
            preset="low-vram",
        )
        self.assertTrue(moe["quantize_moe_experts"])
        self.assertNotIn("gate_up_proj", moe["lora_target_modules"])

    def test_each_known_model_recommends_its_safe_default_without_gpu_data(self) -> None:
        for template in LORA_MODEL_TEMPLATES:
            with self.subTest(model=template.model_id):
                self.assertEqual(
                    recommend_lora_preset(None, template.model_id),
                    template.default_preset,
                )

    def test_starter_template_is_valid_jsonl_but_remains_an_explicit_draft(self) -> None:
        text = starter_dataset_template(LORA_GOALS[1], "Pathfinder")
        report = inspect_jsonl_text(text)

        self.assertEqual(report.example_count, 6)
        self.assertEqual(report.message_count, 18)
        self.assertGreater(report.placeholder_count, 0)
        self.assertFalse(report.ready)
        self.assertEqual(report.status, "DRAFT")

    def test_easy_builder_creates_one_valid_openai_messages_line(self) -> None:
        line = chat_example_line(
            "Please help me recover this failed task.",
            "I’ll inspect the error first, then retry the smallest safe step.",
            system_prompt="Be calm and concise.",
        )
        payload = json.loads(line)
        self.assertEqual(
            [message["role"] for message in payload["messages"]],
            ["system", "user", "assistant"],
        )
        self.assertTrue(inspect_jsonl_text(line).ready)
        with self.assertRaises(LoraStudioError):
            chat_example_line("A real prompt", "")

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

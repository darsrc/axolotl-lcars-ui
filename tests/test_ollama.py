from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from axolotl_lcars_ui.ollama import OllamaManager, OllamaModel


class OllamaTrainingTests(unittest.TestCase):
    def test_chat_uses_a_non_streaming_local_request(self) -> None:
        manager = OllamaManager()
        with patch.object(
            manager,
            "_json",
            return_value={
                "message": {"role": "assistant", "content": "Tuned response"},
                "eval_count": 20,
                "eval_duration": 2_000_000_000,
                "total_duration": 2_000_000_000,
            },
        ) as request:
            result = manager.chat(
                "persona:latest",
                "Hello",
                system_prompt="Be concise.",
                temperature=0.2,
            )

        self.assertEqual(result.content, "Tuned response")
        self.assertEqual(result.tokens_per_second, 10.0)
        payload = request.call_args.args[2]
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["content"], "Hello")

    def test_create_adapter_model_writes_a_managed_modelfile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = root / "outputs" / "persona"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "example/base"}),
                encoding="utf-8",
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            manager = OllamaManager()
            manager.models = [OllamaModel(name="llama3.2:1b", size=1)]

            def refresh() -> list[OllamaModel]:
                manager.models = [
                    OllamaModel(name="llama3.2:1b", size=1),
                    OllamaModel(name="persona:latest", size=2),
                ]
                return manager.models

            with (
                patch("axolotl_lcars_ui.ollama.shutil.which", return_value="/usr/bin/ollama"),
                patch(
                    "axolotl_lcars_ui.ollama.subprocess.run",
                    return_value=subprocess.CompletedProcess(
                        args=[],
                        returncode=0,
                        stdout="success\n",
                        stderr="",
                    ),
                ) as run,
                patch.object(manager, "refresh", side_effect=refresh),
            ):
                created = manager.create_adapter_model(
                    project_root=root,
                    model_name="persona",
                    base_model="llama3.2:1b",
                    adapter_path=str(adapter),
                )

            self.assertEqual(created.name, "persona:latest")
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["/usr/bin/ollama", "create", "persona"])
            modelfile = root / ".lora-studio" / "ollama" / "persona" / "Modelfile"
            content = modelfile.read_text(encoding="utf-8")
            self.assertIn("FROM llama3.2:1b", content)
            self.assertIn(f"ADAPTER {json.dumps(str(adapter))}", content)

    def test_create_adapter_model_rejects_missing_adapter_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = root / "empty-adapter"
            adapter.mkdir()
            manager = OllamaManager()
            manager.models = [OllamaModel(name="llama3.2:1b", size=1)]

            with self.assertRaisesRegex(RuntimeError, "adapter_config.json"):
                manager.create_adapter_model(
                    project_root=root,
                    model_name="persona",
                    base_model="llama3.2:1b",
                    adapter_path=str(adapter),
                )


if __name__ == "__main__":
    unittest.main()

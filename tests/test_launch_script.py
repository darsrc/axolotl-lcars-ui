from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SCRIPT = PROJECT_ROOT / "launch.sh"


class LaunchScriptTests(unittest.TestCase):
    def _project_copy(self, parent: Path) -> Path:
        root = parent / "project"
        root.mkdir()
        shutil.copy2(LAUNCH_SCRIPT, root / "launch.sh")
        (root / "requirements.txt").write_text("", encoding="utf-8")
        return root

    def _fake_venv(self, root: Path, name: str) -> Path:
        venv = root / name
        bin_dir = venv / "bin"
        bin_dir.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
        python = bin_dir / "python"
        python.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                printf 'TEST_VIRTUAL_ENV=%s\\n' "${VIRTUAL_ENV:-}"
                printf 'TEST_PATH=%s\\n' "${PATH:-}"
                printf 'TEST_ARGS=%s\\n' "$*"
                """
            ),
            encoding="utf-8",
        )
        python.chmod(0o755)
        axolotl = bin_dir / "axolotl"
        axolotl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        axolotl.chmod(0o755)
        return venv

    def _run(self, root: Path, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(root / "launch.sh"), "--port", "8123"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    def test_project_venv_is_activated_for_python_and_axolotl(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self._project_copy(Path(temp))
            venv = self._fake_venv(root, ".venv")

            result = self._run(root, env=os.environ.copy())

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"Using virtual environment: {venv}", result.stdout)
            self.assertIn(f"Axolotl CLI: {venv / 'bin/axolotl'}", result.stdout)
            self.assertIn(f"TEST_VIRTUAL_ENV={venv}", result.stdout)
            self.assertIn(f"TEST_PATH={venv / 'bin'}:", result.stdout)
            self.assertIn(
                "TEST_ARGS=-m axolotl_lcars_ui.main --host 127.0.0.1 --port 8123",
                result.stdout,
            )

    def test_single_named_project_venv_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self._project_copy(Path(temp))
            venv = self._fake_venv(root, "axolotl-training")

            result = self._run(root, env=os.environ.copy())

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"Using virtual environment: {venv}", result.stdout)
            self.assertIn(f"TEST_VIRTUAL_ENV={venv}", result.stdout)

    def test_missing_venv_is_created_and_populated_with_uv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            root = self._project_copy(temp_path)
            tools = temp_path / "tools"
            tools.mkdir()
            uv_log = temp_path / "uv.log"
            uv = tools / "uv"
            uv.write_text(
                textwrap.dedent(
                    f"""\
                    #!{sys.executable}
                    import os
                    from pathlib import Path
                    import sys

                    args = sys.argv[1:]
                    with Path(os.environ["UV_TEST_LOG"]).open("a", encoding="utf-8") as handle:
                        handle.write(" ".join(args) + "\\n")
                    if args[0] == "venv":
                        target = Path(args[-1])
                        (target / "bin").mkdir(parents=True)
                        (target / "pyvenv.cfg").write_text("home = fake uv\\n", encoding="utf-8")
                        python = target / "bin" / "python"
                        python.write_text(
                            "#!/usr/bin/env bash\\n"
                            "printf 'TEST_VIRTUAL_ENV=%s\\\\n' \\"${{VIRTUAL_ENV:-}}\\"\\n"
                            "printf 'TEST_PATH=%s\\\\n' \\"${{PATH:-}}\\"\\n",
                            encoding="utf-8",
                        )
                        python.chmod(0o755)
                    """
                ),
                encoding="utf-8",
            )
            uv.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{tools}:{env['PATH']}"
            env["UV_TEST_LOG"] = str(uv_log)

            result = self._run(root, env=env)

            venv = root / ".venv"
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No project virtualenv found; creating .venv with uv", result.stdout)
            self.assertIn("Installing the UI requirements into .venv", result.stdout)
            self.assertIn(f"TEST_VIRTUAL_ENV={venv}", result.stdout)
            self.assertIn(f"TEST_PATH={venv / 'bin'}:", result.stdout)
            calls = uv_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(calls[0], f"venv --python 3.11 {venv}")
            self.assertEqual(
                calls[1],
                f"pip install --python {venv / 'bin/python'} -r {root / 'requirements.txt'}",
            )


if __name__ == "__main__":
    unittest.main()

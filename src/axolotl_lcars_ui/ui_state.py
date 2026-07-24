"""Durable UI state so control selections survive reloads and restarts.

Widget values live in per-session memory inside the LCARS runtime, and that memory is
dropped when a browser tab disconnects. Anything the operator should not have to pick
twice is mirrored here and written to disk, then replayed into every new session.

Structured Axolotl config fields are deliberately *not* stored here: the active YAML
file is their source of truth, and mirroring them would let a stale copy win.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping


STATE_FILE_NAME = ".lcars-ui-state.json"

# Control widgets whose value is an operator preference rather than config data.
PERSISTED_WIDGET_IDS = frozenset(
    {
        # Config manager
        "active-config-select",
        "new-config-name",
        # Setup helpers
        "setup-recipe",
        "setup-model-preset",
        "setup-dataset-preset",
        # Run controls
        "run-action",
        "run-launcher",
        "run-cli-args",
        "run-launcher-args",
        # Hugging Face search controls
        "hf-query",
        "hf-repo-type",
        "hf-sort",
        "hf-compatibility",
        "hf-limit",
        "hf-vram-limit",
        "hf-sift",
        "hf-artifact-filter",
        "hf-quant-filter",
        "hf-fit-filter",
        "hf-repo-id",
        "hf-revision",
        # Content manager
        "delete-repo-id",
        "delete-repo-type",
        # Ollama
        "ollama-model-name",
    }
)


class UiStateStore:
    """JSON-backed store for widget values and cross-page selections."""

    def __init__(self, project_root: Path, *, file_name: str = STATE_FILE_NAME) -> None:
        self.path = project_root / file_name
        self._lock = threading.Lock()
        self.widgets: dict[str, Any] = {}
        self.app: dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------ io

    def load(self) -> None:
        """Read persisted state; a missing or damaged file falls back to empty."""

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        widgets = payload.get("widgets")
        app = payload.get("app")
        with self._lock:
            if isinstance(widgets, dict):
                self.widgets = {str(key): value for key, value in widgets.items()}
            if isinstance(app, dict):
                self.app = {str(key): value for key, value in app.items()}

    def save(self) -> None:
        """Persist state atomically so a crash mid-write cannot truncate the file."""

        with self._lock:
            payload = {"widgets": dict(self.widgets), "app": dict(self.app)}
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle, temp_name = tempfile.mkstemp(dir=self.path.parent, prefix=".lcars-state-", suffix=".tmp")
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    stream.write(text)
                os.replace(temp_name, self.path)
            except BaseException:
                Path(temp_name).unlink(missing_ok=True)
                raise
        except OSError:
            # Persistence is a convenience; a read-only checkout must not break the UI.
            return

    # --------------------------------------------------------------- access

    def widget_values(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.widgets)

    def remember_widgets(self, values: Mapping[str, Any], *, keys: Iterable[str] | None = None) -> bool:
        """Store persistable widget values. Returns True when anything changed."""

        allowed = PERSISTED_WIDGET_IDS if keys is None else frozenset(keys)
        changed = False
        with self._lock:
            for widget_id, value in values.items():
                if widget_id not in allowed:
                    continue
                if not isinstance(value, (str, int, float, bool)) and value is not None:
                    continue
                if self.widgets.get(widget_id) != value:
                    self.widgets[widget_id] = value
                    changed = True
        return changed

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.app.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        """Store a cross-page value. Returns True when the value changed."""

        with self._lock:
            if self.app.get(key) == value:
                return False
            self.app[key] = value
        return True

    def set_many(self, values: Mapping[str, Any]) -> bool:
        changed = False
        for key, value in values.items():
            changed = self.set(key, value) or changed
        return changed


__all__ = ["PERSISTED_WIDGET_IDS", "STATE_FILE_NAME", "UiStateStore"]

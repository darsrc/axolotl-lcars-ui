#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="127.0.0.1"
PORT="8000"
OPEN_BROWSER="0"
PYTHON_VERSION="3.11"
VENV_DIR=""

usage() {
  cat <<'EOF'
Axolotl LCARS UI launcher

Usage:
  ./launch.sh
  ./launch.sh help
  ./launch.sh port 8080
  ./launch.sh ip 0.0.0.0 port 8080
  ./launch.sh --ip 0.0.0.0 --port 8080 --open

Args:
  help, -h, --help       Show this help
  ip, --ip, host, --host Bind address (default: 127.0.0.1)
  port, --port           Bind port (default: 8000)
  open, --open           Open the app in the default browser

Environment:
  AXOLOTL_LCARS_VENV      Project venv path to use instead of auto-detection
EOF
}

is_venv() {
  [[ -f "$1/pyvenv.cfg" && -x "$1/bin/python" ]]
}

select_venv() {
  local candidate=""
  local config=""
  local -a discovered=()

  if [[ -n "${AXOLOTL_LCARS_VENV:-}" ]]; then
    candidate="$AXOLOTL_LCARS_VENV"
    if [[ "$candidate" != /* ]]; then
      candidate="$ROOT_DIR/$candidate"
    fi
    if ! is_venv "$candidate"; then
      echo "AXOLOTL_LCARS_VENV is not a usable virtualenv: $candidate" >&2
      return 1
    fi
    VENV_DIR="$candidate"
    return 0
  fi

  for candidate in "$ROOT_DIR/.venv" "$ROOT_DIR/venv"; do
    if is_venv "$candidate"; then
      VENV_DIR="$candidate"
      return 0
    fi
  done

  shopt -s nullglob dotglob
  for config in "$ROOT_DIR"/*/pyvenv.cfg; do
    candidate="${config%/pyvenv.cfg}"
    if is_venv "$candidate"; then
      discovered+=("$candidate")
    fi
  done
  shopt -u nullglob dotglob

  if [[ "${#discovered[@]}" -eq 1 ]]; then
    VENV_DIR="${discovered[0]}"
    return 0
  fi
  if [[ "${#discovered[@]}" -gt 1 ]]; then
    echo "Multiple project virtualenvs were found:" >&2
    printf '  %s\n' "${discovered[@]}" >&2
    echo "Set AXOLOTL_LCARS_VENV to choose one." >&2
    return 1
  fi

  return 2
}

create_venv() {
  local uv_path=""

  uv_path="$(command -v uv || true)"
  if [[ -z "$uv_path" ]]; then
    echo "No project virtualenv was found and uv is not installed." >&2
    echo "Install uv, or create .venv manually with: python3 -m venv .venv" >&2
    return 1
  fi
  if [[ -e "$ROOT_DIR/.venv" ]]; then
    echo "$ROOT_DIR/.venv exists but is not a usable virtualenv." >&2
    echo "Repair it, move it aside, or set AXOLOTL_LCARS_VENV to another environment." >&2
    return 1
  fi

  VENV_DIR="$ROOT_DIR/.venv"
  echo "No project virtualenv found; creating .venv with uv (Python $PYTHON_VERSION)..."
  "$uv_path" venv --python "$PYTHON_VERSION" "$VENV_DIR"
  echo "Installing the UI requirements into .venv..."
  "$uv_path" pip install --python "$VENV_DIR/bin/python" -r "$ROOT_DIR/requirements.txt"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    help|-h|--help)
      usage
      exit 0
      ;;
    ip|--ip|host|--host)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      HOST="$2"
      shift 2
      ;;
    ip=*|--ip=*|host=*|--host=*)
      HOST="${1#*=}"
      shift
      ;;
    port|--port)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      PORT="$2"
      shift 2
      ;;
    port=*|--port=*)
      PORT="${1#*=}"
      shift
      ;;
    open|--open)
      OPEN_BROWSER="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Port must be numeric: $PORT" >&2
  exit 2
fi

if select_venv; then
  :
else
  select_status="$?"
  if [[ "$select_status" -ne 2 ]]; then
    exit "$select_status"
  fi
  create_venv
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin${PATH:+:$PATH}"

child_pid=""

cleanup() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    echo
    echo "Stopping Axolotl LCARS UI..."
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}

trap cleanup INT TERM

args=(--host "$HOST" --port "$PORT")
if [[ "$OPEN_BROWSER" == "1" ]]; then
  args+=(--open)
fi

echo "Using virtual environment: $VENV_DIR"
if command -v axolotl >/dev/null 2>&1; then
  echo "Axolotl CLI: $(command -v axolotl)"
else
  echo "Axolotl CLI: not installed in $VENV_DIR (the UI will still start)"
fi
echo "Starting Axolotl LCARS UI at http://$HOST:$PORT/"
PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$VENV_DIR/bin/python" -m axolotl_lcars_ui.main "${args[@]}" &
child_pid="$!"
wait "$child_pid"

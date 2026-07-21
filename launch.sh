#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="127.0.0.1"
PORT="8000"
OPEN_BROWSER="0"

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
EOF
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

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing venv at $ROOT_DIR/.venv. Create it with: python3 -m venv .venv" >&2
  exit 1
fi

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

echo "Starting Axolotl LCARS UI at http://$HOST:$PORT/"
PYTHONPATH="$ROOT_DIR/src" "$ROOT_DIR/.venv/bin/python" -m axolotl_lcars_ui.main "${args[@]}" &
child_pid="$!"
wait "$child_pid"

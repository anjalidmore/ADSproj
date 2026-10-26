#!/usr/bin/env bash
# Start the whole project locally: regenerate results (backend), serve docs/
# (team viva guide) and launch the Streamlit dashboard (frontend).
#
# macOS / Linux:  ./run.sh
# Windows:        run in Git Bash (bash run.sh), or double-click run.bat
#
# Options:
#   --skip-export   don't re-run scripts/export_results.py
#   --open          open both links in the default browser
# Ports can be overridden: APP_PORT=8502 DOCS_PORT=8001 ./run.sh

set -euo pipefail
cd "$(dirname "$0")"

SKIP_EXPORT=0
OPEN_BROWSER=0
for arg in "$@"; do
  case "$arg" in
    --skip-export) SKIP_EXPORT=1 ;;
    --open) OPEN_BROWSER=1 ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1; VENV_PY=".venv/Scripts/python.exe" ;;
  *)                    IS_WINDOWS=0; VENV_PY=".venv/bin/python" ;;
esac

log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. Python environment -------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  log "Syncing environment with uv"
  uv sync --quiet
elif [ ! -x "$VENV_PY" ]; then
  SYS_PY=""
  for cand in python3.12 python3 python py; do
    if command -v "$cand" >/dev/null 2>&1 \
       && "$cand" -c 'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] < (3, 14))' 2>/dev/null; then
      SYS_PY="$cand"; break
    fi
  done
  [ -n "$SYS_PY" ] || die "Need uv (https://docs.astral.sh/uv/) or Python 3.11-3.13 on PATH."
  log "Creating .venv with $SYS_PY (uv not found)"
  "$SYS_PY" -m venv .venv
  "$VENV_PY" -m pip install --quiet --upgrade pip
  "$VENV_PY" -m pip install --quiet -e . pytest
fi
[ -x "$VENV_PY" ] || die "$VENV_PY not found after setup."

# Run Python from .venv directly (not via `uv run`) so the PIDs we kill on
# exit are the real servers, which matters on Windows.
PY="$VENV_PY"
export PYTHONUTF8=1   # Windows consoles otherwise choke on CO₂ / ≈ in output

# --- 2. Backend: regenerate results.json, figures, concept matrix ----------
if [ "$SKIP_EXPORT" -eq 0 ]; then
  log "Running analysis pipeline (scripts/export_results.py)"
  "$PY" scripts/export_results.py
else
  log "Skipping analysis pipeline"
fi

# --- 3. Pick free ports ----------------------------------------------------
free_port() {
  "$PY" - "$1" <<'EOF'
import socket, sys
port = int(sys.argv[1])
while True:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            break
        except OSError:
            port += 1
print(port)
EOF
}
APP_PORT="$(free_port "${APP_PORT:-8501}")"
DOCS_PORT="$(free_port "${DOCS_PORT:-8000}")"

# --- 4. Start servers ------------------------------------------------------
PIDS=()
cleanup() {
  trap - INT TERM EXIT
  echo
  log "Stopping servers"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
    if [ "$IS_WINDOWS" -eq 1 ]; then
      taskkill //F //T //PID "$(cat /proc/"$pid"/winpid 2>/dev/null || echo "$pid")" >/dev/null 2>&1 || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

log "Serving docs/ on port $DOCS_PORT"
"$PY" -m http.server "$DOCS_PORT" --bind 127.0.0.1 --directory docs >/dev/null 2>&1 &
PIDS+=($!)

log "Starting Streamlit on port $APP_PORT"
"$PY" -m streamlit run app/Home.py \
  --server.port "$APP_PORT" --server.address 127.0.0.1 --server.headless true &
PIDS+=($!)

# Wait for Streamlit's health endpoint before printing links.
"$PY" - "$APP_PORT" <<'EOF' || die "Streamlit did not start (see output above)."
import sys, time, urllib.request
url = f"http://127.0.0.1:{sys.argv[1]}/_stcore/health"
for _ in range(120):
    try:
        if urllib.request.urlopen(url, timeout=1).status == 200:
            sys.exit(0)
    except Exception:
        time.sleep(0.5)
sys.exit(1)
EOF

APP_URL="http://localhost:$APP_PORT"
TEAM_URL="http://localhost:$DOCS_PORT/viva_guide.html"

cat <<EOF

  ┌──────────────────────────────────────────────────────────────┐
    Frontend dashboard (Streamlit):  $APP_URL
    Team dashboard (viva guide):     $TEAM_URL
    Docs folder (figures, results):  http://localhost:$DOCS_PORT/
  └──────────────────────────────────────────────────────────────┘
    Press Ctrl+C to stop both servers.

EOF

if [ "$OPEN_BROWSER" -eq 1 ]; then
  for url in "$APP_URL" "$TEAM_URL"; do
    if [ "$IS_WINDOWS" -eq 1 ]; then cmd.exe //c start "" "$url"
    elif command -v open >/dev/null 2>&1; then open "$url"
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url" >/dev/null 2>&1
    fi
  done
fi

# Exit (and clean up) as soon as either server dies.
while true; do
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null || { log "A server exited"; exit 1; }
  done
  sleep 2
done

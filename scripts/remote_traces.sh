#!/usr/bin/env bash
# Re-execute the prepared trajectories on a plain x86 Linux machine with Docker, then copy the
# frozen trace back. The GPU machine is not needed for this: the job is CPU and Docker only.
#
#   scripts/remote_traces.sh user@host -i key [--start | --status | --fetch] [--dry-run]
#
#   --start    sync this tree, install Docker if missing, set a shutdown timer, and start
#              `make traces` detached on the machine (the default)
#   --status   print the end of the machine's log
#   --fetch    copy traces/ and the log back to results/traces-<host>/
#
# The machine shuts itself down DEADMAN_MINUTES (default 240) after --start, whatever happens.
# Durations are the machine's own: its instance type is recorded in every trace.
set -u -o pipefail

HOST="" KEY="" ACTION=start DRY_RUN=0
DEADMAN_MINUTES=${DEADMAN_MINUTES:-240}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_DIR=layover-traces

while [ $# -gt 0 ]; do
    case "$1" in
        -i) KEY=$2; shift 2 ;;
        --start) ACTION=start; shift ;;
        --status) ACTION=status; shift ;;
        --fetch) ACTION=fetch; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) HOST=$1; shift ;;
    esac
done
[ -n "$HOST" ] || { echo "no host given" >&2; exit 2; }
[ -f "$ROOT/traces/prepared/manifest.json" ] || { echo "traces/prepared/manifest.json is missing: run make traces-prepare first" >&2; exit 2; }

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
[ -z "$KEY" ] || SSH+=(-i "$KEY")
RSYNC=(rsync -rlptz --timeout=120 -e "${SSH[*]}")
# The same rules as remote_gpu.sh: no hidden directories (.git, the venvs, private notes), no
# results or build output, no secrets, and nothing this clone lists in .git/info/exclude.
EXCLUDES=(
    --exclude='/.*/' --exclude=/results/ --exclude=/native/build/ --exclude=/build/
    --exclude=__pycache__/ --exclude=.hypothesis/ --exclude=.pytest_cache/ --exclude='*.egg-info/'
    --exclude='*.pyc' --exclude='*.o' --exclude='*.so' --exclude='*.dylib' --exclude=.DS_Store
    --exclude=.env --exclude='.env.*' --exclude=.netrc --exclude='*.pem' --exclude='*.key'
    --exclude='id_rsa*' --exclude='id_ecdsa*' --exclude='id_ed25519*'
    --exclude-from="$ROOT/.git/info/exclude"
)
[ -f "$ROOT/.git/info/exclude" ] || { echo "no .git/info/exclude in $ROOT: it lists the private files that must not be synced" >&2; exit 2; }

run() {
    if [ "$DRY_RUN" = 1 ]; then printf '%q ' "$@"; echo; else "$@"; fi
}

REMOTE_START='
set -e
cd ~/'"$REMOTE_DIR"'
if ! command -v docker >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
    sudo -n apt-get update -qq && sudo -n apt-get install -y -qq docker.io make >/dev/null
    sudo -n usermod -aG docker "$(id -un)"
fi
sudo -n shutdown -h +'"$DEADMAN_MINUTES"' "Layover traces dead-man" >/dev/null 2>&1 || echo "WARNING: no shutdown timer"
# The docker group is new to this login, so the job runs through sg to have it.
nohup sg docker -c "make traces PYTHON=python3" > traces.log 2>&1 < /dev/null &
echo started
'

case "$ACTION" in
    start)
        run "${RSYNC[@]}" "${EXCLUDES[@]}" "$ROOT/" "$HOST:$REMOTE_DIR/"
        run "${SSH[@]}" "$HOST" "$REMOTE_START"
        ;;
    status)
        run "${SSH[@]}" "$HOST" "tail -n 20 ~/$REMOTE_DIR/traces.log; ls ~/$REMOTE_DIR/traces/swebench_30.jsonl 2>/dev/null"
        ;;
    fetch)
        dest="$ROOT/results/traces-${HOST#*@}"
        run mkdir -p "$dest"
        run "${RSYNC[@]}" "$HOST:$REMOTE_DIR/traces/" "$dest/traces/"
        run "${RSYNC[@]}" "$HOST:$REMOTE_DIR/traces.log" "$dest/"
        echo "copied to $dest"
        ;;
esac

#!/usr/bin/env bash
# Create one virtual environment per baseline, at the versions the benchmark was checked
# against. vLLM 0.33.0 pins torch==2.14.0 and SGLang 0.5.23 pins torch==2.14.1, so they
# cannot share one.
#
#   scripts/setup_baselines.sh            create .venvs/vllm and .venvs/sglang
#   scripts/setup_baselines.sh vllm       only one of them
#
# Each environment is several gigabytes. Both ask for CUDA 13 builds of torch, so the
# driver must support CUDA 13; the first check below says if it does not.
set -euo pipefail

VLLM_VERSION=0.33.0
SGLANG_VERSION=0.5.23

cd "$(dirname "${BASH_SOURCE[0]}")/.."

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

PYTHON=""
for candidate in python3.12 python3.13 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON=$candidate
        break
    fi
done
[ -n "$PYTHON" ] || die "Python 3.11 to 3.13 is required; both baselines publish wheels for those"

# uv seeds pip by itself, while python -m venv needs ensurepip, which Ubuntu 24.04's Python lacks
# without python3-venv. The main environment is built on that Python there. setup_gpu.sh leaves a
# pinned uv in this cache.
UV=""
for candidate in "${XDG_CACHE_HOME:-$HOME/.cache}"/layover/uv-*/uv; do
    [ -x "$candidate" ] && UV=$candidate
done
[ -n "$UV" ] || UV=$(command -v uv 2>/dev/null || true)

create_venv() {
    local venv=$1
    if [ -z "$UV" ]; then
        "$PYTHON" -m venv "$venv" || die "no uv, and $PYTHON cannot make an environment: install python3-venv, or run scripts/setup_gpu.sh first"
    elif [ ! -x "$venv/bin/pip" ]; then
        # A stage re-run keeps a finished environment, which took gigabytes; a partial one has no pip yet.
        "$UV" venv --clear --seed --python "$PYTHON" "$venv"
    fi
}

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | grep -E "CUDA Version" || true
fi

install_baseline() {
    local name=$1 package=$2 version=$3 venv=.venvs/$1
    say "Creating $venv with $package==$version"
    create_venv "$venv"
    "$venv/bin/pip" install --quiet --upgrade pip
    "$venv/bin/pip" install --quiet "$package==$version"
    "$venv/bin/python" - "$package" "$version" <<'PY'
import importlib.metadata as metadata
import sys

package, version = sys.argv[1:]
installed = metadata.version(package)
if installed != version:
    raise SystemExit(f"{package} is {installed}, expected {version}")
print(package, installed)
PY
    "$venv/bin/python" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
    "$venv/bin/python" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' ||
        die "$name: torch cannot use the GPU. Compare 'nvidia-smi' with the CUDA version above."
    "$venv/bin/pip" freeze > "$venv/freeze.txt"
}

wanted=("$@")
[ ${#wanted[@]} -gt 0 ] || wanted=(vllm sglang)
for name in "${wanted[@]}"; do
    case "$name" in
        vllm) install_baseline vllm vllm "$VLLM_VERSION" ;;
        sglang) install_baseline sglang sglang "$SGLANG_VERSION" ;;
        *) die "unknown baseline: $name" ;;
    esac
done
say "Done"

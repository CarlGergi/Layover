#!/usr/bin/env bash
# Set up a fresh Linux machine with an NVIDIA GPU, then run the GPU tests.
#
#   scripts/setup_gpu.sh                check, install, download models, run GPU tests
#   scripts/setup_gpu.sh --check-only   only report whether the machine is suitable
#   scripts/setup_gpu.sh --reproduce    also run every benchmark afterwards
#   scripts/setup_gpu.sh --no-toolkit   skip the CUDA toolkit (nvcc, compute-sanitizer)
#
# Run it from a clone of this repository. It installs system packages with sudo only
# for what is missing, and never installs or changes the NVIDIA driver.
set -euo pipefail

CHECK_ONLY=0
REPRODUCE=0
TOOLKIT=1
MIN_GPU_MIB=22000
GPU_MODEL=${GPU_MODEL:-Qwen/Qwen3-4B}
CPU_MODEL=${CPU_MODEL:-Qwen/Qwen3-0.6B}
VENV=${VENV:-.venv}

for arg in "$@"; do
    case "$arg" in
        --check-only) CHECK_ONLY=1 ;;
        --reproduce) REPRODUCE=1 ;;
        --no-toolkit) TOOLKIT=0 ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

say "Checking the machine"
[ "$(uname -s)" = "Linux" ] || die "this needs Linux; found $(uname -s). Apple Silicon cannot run CUDA."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found: no NVIDIA driver is installed.
Install a driver for your distribution first, for example on Ubuntu:
    sudo ubuntu-drivers install && sudo reboot
Then run this script again."
fi
if ! GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>&1); then
    die "nvidia-smi failed, so the driver is not working: $GPU_INFO"
fi
[ -n "$GPU_INFO" ] || die "nvidia-smi lists no GPUs"
echo "$GPU_INFO"

GPU_MIB=$(echo "$GPU_INFO" | head -n1 | cut -d, -f2 | tr -d ' ')
if [ "$GPU_MIB" -lt "$MIN_GPU_MIB" ]; then
    echo "WARNING: first GPU has ${GPU_MIB} MiB. The benchmarks target 24 GB (L4 or RTX 4090 class)."
fi

PYTHON=""
for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
        PYTHON=$candidate
        break
    fi
done
[ -n "$PYTHON" ] && echo "Python: $("$PYTHON" --version)" || echo "Python: 3.11 or newer not found"

for tool in nvcc compute-sanitizer; do
    if command -v "$tool" >/dev/null 2>&1; then echo "$tool: $(command -v "$tool")"; else echo "$tool: not found"; fi
done

if [ "$CHECK_ONLY" = 1 ]; then
    [ -n "$PYTHON" ] || die "Python 3.11 or newer is required"
    say "Machine looks suitable"
    exit 0
fi

need_sudo() {
    if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}

if [ -z "$PYTHON" ] || ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
    command -v apt-get >/dev/null 2>&1 || die "install Python 3.11+ with venv support, then run this again"
    say "Installing Python"
    need_sudo apt-get update
    need_sudo apt-get install -y python3 python3-venv
    PYTHON=python3
    "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' ||
        die "this distribution's Python is older than 3.11; install a newer one and run this again"
fi

# The toolkit is for the native transfer kernel and compute-sanitizer. PyTorch brings
# its own CUDA runtime, so the model and Triton paths work without it.
if [ "$TOOLKIT" = 1 ] && ! command -v nvcc >/dev/null 2>&1; then
    say "Installing the CUDA toolkit"
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        echo "Found /usr/local/cuda/bin/nvcc; add /usr/local/cuda/bin to PATH."
    elif command -v apt-get >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        . /etc/os-release
        arch=$(uname -m)
        [ "$arch" = "aarch64" ] && arch=sbsa
        repo="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/${arch}"
        need_sudo apt-get update
        need_sudo apt-get install -y ca-certificates curl g++ make
        curl -fsSL "${repo}/cudatools.pub" | need_sudo tee /etc/apt/trusted.gpg.d/nvidia-cuda.asc >/dev/null
        echo "deb ${repo}/ /" | need_sudo tee /etc/apt/sources.list.d/nvidia-cuda.list >/dev/null
        need_sudo apt-get update
        need_sudo apt-get install -y cuda-toolkit
        export PATH="/usr/local/cuda/bin:$PATH"
    else
        die "install the CUDA toolkit for your distribution, or pass --no-toolkit"
    fi
fi

say "Creating the Python environment in $VENV"
[ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -e ".[dev,gpu]"

say "Checking that PyTorch can use the GPU"
if ! "$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
print("device:", torch.cuda.get_device_name(0))
x = torch.ones(1024, 1024, dtype=torch.bfloat16, device="cuda")
assert (x @ x).float().sum().item() > 0
print("cuBLAS matmul: ok")
import triton
print("triton", triton.__version__)
PY
then
    die "PyTorch cannot use the GPU. The usual cause is a driver older than the CUDA build of
torch: compare 'nvidia-smi' with the version above, then upgrade the driver or install a
matching torch from https://download.pytorch.org/whl/."
fi

say "Downloading model weights (safetensors only)"
"$VENV/bin/python" - "$CPU_MODEL" "$GPU_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
for name in sys.argv[1:]:
    print(name, snapshot_download(name, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"]))
PY

say "Running the CPU suite"
"$VENV/bin/python" -m pytest -q

say "Running the GPU tests on $GPU_MODEL"
"$VENV/bin/python" -m pytest -q --gpu -m gpu --checkpoint "$GPU_MODEL"

if [ "$REPRODUCE" = 1 ]; then
    say "Reproducing every benchmark"
    make reproduce PYTHON="$VENV/bin/python" GPU_MODEL="$GPU_MODEL"
fi

say "Done"

#!/usr/bin/env bash
# Set up an x86_64 Linux machine with an NVIDIA GPU, to the point where every stage can run.
#
#   scripts/setup_gpu.sh                check, install, build, download models
#   scripts/setup_gpu.sh --check-only   only report whether the machine is suitable
#
# Run it from a clone of this repository, as often as you like: what is already there is
# kept. It uses non-interactive sudo for system packages it finds missing, it never installs
# or changes the NVIDIA driver, and it needs no key or model account of any kind.
#
# LAYOVER_NO_DOCKER=1 leaves Docker alone, for a run without the traces stage, which is the only
# one that runs containers. A rented container can then run every other stage.
set -euo pipefail

CHECK_ONLY=0
MIN_GPU_MIB=22000
GPU_MODEL=${GPU_MODEL:-Qwen/Qwen3-4B}
CPU_MODEL=${CPU_MODEL:-Qwen/Qwen3-0.6B}
VENV=${VENV:-.venv}
OS_RELEASE=${OS_RELEASE:-/etc/os-release}
APT_DIR=${APT_DIR:-/etc/apt}

# Table 2 of the CUDA 13.2 release notes: any 13.x runtime starts on a driver of at least 580.
MIN_DRIVER_CUDA13=580
# Table 3 of the same notes: the driver matching 13.2 Update 2, which torch's wheels pin
# through cuda-toolkit==13.2.2. Older ones run under minor version compatibility only.
TOOLKIT_DRIVER_CUDA132=595.71.05

# A pinned, checksummed release rather than an installer script piped into a shell. It also
# brings Python 3.12 to Ubuntu 22.04, whose own Python is too old.
UV_VERSION=0.13.1
UV_SHA256=eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220

for arg in "$@"; do
    case "$arg" in
        --check-only) CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."

say() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# True when dotted version $1 is at least $2.
version_ge() {
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo -n "$@"
    else
        die "this step needs root and passwordless sudo is not available:
    $*
Run that yourself, or configure passwordless sudo, then run this script again."
    fi
}

# Fresh cloud images run unattended upgrades at boot, so wait for the dpkg lock.
apt_get() {
    as_root env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 "$@"
}

docker_runs() {
    docker run --rm hello-world > /dev/null 2>&1
}

# Trace re-execution runs the task containers, so a failure here must come before any large
# download. Only a plain VM gets Docker installed: on a machine that is itself a container an
# install cannot give it a working daemon.
ensure_docker() {
    local me
    me=$(id -un)
    if command -v docker >/dev/null 2>&1; then
        docker_runs && { echo "Docker runs containers"; return 0; }
        # Installed, but this user may simply not be allowed to use it.
        if [ "$(id -u)" -ne 0 ] && sudo -n docker run --rm hello-world > /dev/null 2>&1; then
            as_root usermod -aG docker "$me"
            echo "Added $me to the docker group; the stages pick that up without a reconnect"
            return 0
        fi
        die "docker is installed but 'docker run --rm hello-world' fails. Trace re-execution needs a
Docker daemon that can run containers, and rental offerings that are themselves containers
commonly cannot do that. Rent a VM or bare metal instead."
    fi
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
        die "Docker is missing and this machine is itself a container, so installing it would not give
it a working daemon. Trace re-execution needs one. Rent a VM or bare metal instead."
    fi
    command -v apt-get >/dev/null 2>&1 || die "Docker is missing; install it for your distribution, then run this again"
    say "Docker is missing on this plain VM: installing Ubuntu's docker.io"
    apt_get update
    apt_get install -y --no-install-recommends docker.io
    as_root docker run --rm hello-world > /dev/null ||
        die "docker.io installed, but 'docker run --rm hello-world' fails, so trace re-execution cannot run"
    [ "$(id -u)" -eq 0 ] || as_root usermod -aG docker "$me"
    echo "Docker runs containers"
}

say "Checking the machine"
[ "$(uname -s)" = "Linux" ] || die "this needs Linux; found $(uname -s). Apple Silicon cannot run CUDA."
[ "$(uname -m)" = "x86_64" ] || die "this needs x86_64; found $(uname -m). The task containers for trace re-execution are x86_64 images."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found: no NVIDIA driver is installed.
Plain cloud images often ship without one; on AWS g6, for example, it depends on the AMI.
Pick an image with a driver of at least $TOOLKIT_DRIVER_CUDA132, or install one following
https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html
which on Ubuntu, once its repository is set up, comes down to:
    sudo apt install linux-headers-\$(uname -r) nvidia-open && sudo reboot
Then run this script again. It will not install a driver itself."
fi
if ! GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>&1); then
    die "nvidia-smi failed, so the driver is not working: $GPU_INFO"
fi
[ -n "$GPU_INFO" ] || die "nvidia-smi lists no GPUs"
echo "$GPU_INFO"

GPU_MIB=$(echo "$GPU_INFO" | head -n1 | cut -d, -f2 | tr -d ' ')
if [ "$GPU_MIB" -lt "$MIN_GPU_MIB" ]; then
    warn "first GPU has ${GPU_MIB} MiB. The benchmarks target 24 GB (L4 or RTX 4090 class)."
fi

DRIVER=$(echo "$GPU_INFO" | head -n1 | cut -d, -f3 | tr -d ' ')
version_ge "$DRIVER" "$MIN_DRIVER_CUDA13" || die "driver $DRIVER is too old: torch's CUDA 13 builds need at least \
$MIN_DRIVER_CUDA13, and $TOOLKIT_DRIVER_CUDA132 for full CUDA 13.2 support. Upgrade the driver, then run this again.
See https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html"
echo "Driver $DRIVER is at least $MIN_DRIVER_CUDA13"

# A toolkit must match torch's CUDA version, because the native library shares a process
# with torch's runtime. compute-sanitizer is required: every custom kernel runs under it.
find_toolkit() {
    local want=$1 dir nvcc_path
    nvcc_path=$(command -v nvcc 2>/dev/null || true)
    for dir in "/usr/local/cuda-$want/bin" "${nvcc_path%/nvcc}" /usr/local/cuda/bin; do
        [ -x "$dir/nvcc" ] && [ -x "$dir/compute-sanitizer" ] || continue
        case "$("$dir/nvcc" --version 2>/dev/null)" in
            *"cuda_$want"*) echo "$dir"; return 0 ;;
        esac
    done
    return 1
}

if [ "$CHECK_ONLY" = 1 ]; then
    if command -v docker >/dev/null 2>&1; then
        if docker_runs; then echo "docker: runs containers"; else warn "docker cannot run containers here"; fi
    else
        echo "docker: not found, will be installed on a plain VM"
    fi
    for tool in nvcc compute-sanitizer; do
        if command -v "$tool" >/dev/null 2>&1; then echo "$tool: $(command -v "$tool")"
        elif [ -x "/usr/local/cuda/bin/$tool" ]; then echo "$tool: /usr/local/cuda/bin/$tool"
        else echo "$tool: not found, will be installed"; fi
    done
    if [ "$(id -u)" -ne 0 ] && ! sudo -n true 2>/dev/null; then
        warn "no passwordless sudo, so anything missing has to be installed by hand"
    fi
    say "Machine looks suitable"
    exit 0
fi

if [ "${LAYOVER_NO_DOCKER:-}" = 1 ]; then
    echo "Docker: left alone (LAYOVER_NO_DOCKER=1); only the traces stage needs it"
else
    ensure_docker
fi

MISSING=""
for pair in curl:curl g++:g++ make:make tar:tar sha256sum:coreutils; do
    command -v "${pair%%:*}" >/dev/null 2>&1 || MISSING="$MISSING ${pair##*:}"
done
if [ -n "$MISSING" ]; then
    command -v apt-get >/dev/null 2>&1 || die "install these, then run this again:$MISSING"
    say "Installing system packages:$MISSING"
    apt_get update
    # shellcheck disable=SC2086
    apt_get install -y --no-install-recommends ca-certificates $MISSING
fi

UV_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/layover/uv-$UV_VERSION"
UV="$UV_DIR/uv"
if [ ! -x "$UV" ]; then
    say "Fetching uv $UV_VERSION"
    mkdir -p "$UV_DIR"
    curl -fsSL -o "$UV_DIR/uv.tar.gz" \
        "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz"
    echo "$UV_SHA256  $UV_DIR/uv.tar.gz" | sha256sum -c --quiet - || die "uv download failed its checksum"
    # As root, tar keeps the archive's owners, which a network volume refuses to set.
    tar --no-same-owner -xzf "$UV_DIR/uv.tar.gz" -C "$UV_DIR" --strip-components=1
fi

say "Creating the Python environment in $VENV"
[ -x "$VENV/bin/python" ] || "$UV" venv --python 3.12 "$VENV"
"$VENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' ||
    die "$VENV holds a Python older than 3.11; move it away and run this again"
"$UV" pip install --python "$VENV/bin/python" -e ".[dev,gpu]"

TORCH_CUDA=$("$VENV/bin/python" -c 'import torch; print(torch.version.cuda or "")')
[ -n "$TORCH_CUDA" ] || die "pip resolved a CPU-only torch ($("$VENV/bin/python" -c 'import torch; print(torch.__version__)'))
This needs a CUDA build, which on x86_64 Linux is the PyPI default."
echo "torch is built for CUDA $TORCH_CUDA"

case "$TORCH_CUDA" in
    13.2)
        if ! version_ge "$DRIVER" "$TOOLKIT_DRIVER_CUDA132"; then
            warn "driver $DRIVER is older than $TOOLKIT_DRIVER_CUDA132, the one matching torch's CUDA 13.2.
It runs under minor version compatibility, which has a limited feature set, and device code
loaded as PTX will not work. Upgrade the driver if anything below reports
cudaErrorCallRequiresNewerDriver. nvcc needs an explicit -arch under this mode."
        fi
        ;;
    13.*)
        warn "torch moved to CUDA $TORCH_CUDA. The driver floor checked above covers all of 13.x, but look
up its exact driver in that release's notes:
https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html"
        ;;
    *)
        warn "torch is built for CUDA $TORCH_CUDA and this script's driver check covers 13.x only."
        ;;
esac

if ! TOOLKIT_BIN=$(find_toolkit "$TORCH_CUDA"); then
    say "Installing CUDA toolkit $TORCH_CUDA"
    command -v apt-get >/dev/null 2>&1 ||
        die "install CUDA toolkit $TORCH_CUDA with nvcc and compute-sanitizer for your distribution, then run this again"
    # shellcheck disable=SC1090
    . "$OS_RELEASE"
    [ "${ID:-}" = "ubuntu" ] || warn "NVIDIA's repository scheme is checked for Ubuntu; this is ${ID:-unknown}"
    # NVIDIA's network repository install: the keyring package holds the signing key, the
    # sources.list entry and an apt pin. Paths are ubuntu2204 and ubuntu2404, and x86_64 is what
    # the keyring's own sources.list uses; the guide's amd64 path answers 404.
    repo="https://developer.download.nvidia.com/compute/cuda/repos/${ID:-ubuntu}${VERSION_ID//./}/x86_64"
    # CUDA base images configure the repository with a key of their own, and a second entry for
    # it with another key makes apt-get update fail.
    if grep -qs "developer.download.nvidia.com/compute/cuda/repos" "$APT_DIR/sources.list" "$APT_DIR"/sources.list.d/*; then
        echo "NVIDIA's repository is already configured"
    elif ! dpkg -s cuda-keyring >/dev/null 2>&1; then
        keyring_dir=$(mktemp -d)
        curl -fsSL -o "$keyring_dir/cuda-keyring.deb" "$repo/cuda-keyring_1.1-1_all.deb" ||
            die "no NVIDIA repository at $repo. See https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
        as_root dpkg -i "$keyring_dir/cuda-keyring.deb"
    fi
    apt_get update
    # Versioned, because plain cuda-toolkit tracks the newest release instead of torch's.
    package="cuda-toolkit-${TORCH_CUDA//./-}"
    plan=$(apt-get -s install --no-install-recommends "$package" 2>&1) ||
        die "apt cannot install $package from $repo:
$plan"
    if echo "$plan" | grep -E '^Inst (nvidia-(driver|dkms|open|kernel)|cuda-drivers|libnvidia-compute)'; then
        die "$package would change the NVIDIA driver (packages above). This script never does that."
    fi
    apt_get install -y --no-install-recommends "$package"
    TOOLKIT_BIN=$(find_toolkit "$TORCH_CUDA") ||
        die "$package installed, but no nvcc $TORCH_CUDA with compute-sanitizer under /usr/local/cuda-$TORCH_CUDA/bin"
fi
export PATH="$TOOLKIT_BIN:$PATH"
# native/Makefile looks here first, and /usr/local/cuda may point at some other version.
export CUDA_HOME="${TOOLKIT_BIN%/bin}"
# Non-login shells do not read the toolkit's profile, so later steps source this.
printf 'export PATH="%s:$PATH"\nexport CUDA_HOME="%s"\n' "$TOOLKIT_BIN" "$CUDA_HOME" > "$VENV/cuda_env.sh"
echo "Toolkit: $TOOLKIT_BIN"

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
    die "PyTorch cannot use the GPU, with driver $DRIVER and torch's CUDA $TORCH_CUDA. The usual cause is a
driver older than the CUDA build of torch: upgrade the driver or install a matching torch from
https://download.pytorch.org/whl/."
fi

say "Building the native CUDA library"
[ -f native/Makefile ] || die "native/Makefile is missing, so the transfer kernel cannot be built"
make -C native cuda

say "Downloading model weights (safetensors only)"
"$VENV/bin/python" - "$CPU_MODEL" "$GPU_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
for name in sys.argv[1:]:
    print(name, snapshot_download(name, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"]))
PY

# Results come back from the machine, so the versions behind them come back too.
say "Versions"
mkdir -p results
{
    # shellcheck disable=SC1090
    echo "os: $(. "$OS_RELEASE" && echo "${PRETTY_NAME:-unknown}"), kernel $(uname -r), $(uname -m)"
    echo "gpu: $GPU_INFO"
    echo "python: $("$VENV/bin/python" --version 2>&1)"
    "$VENV/bin/python" -c 'import torch, triton, transformers
print("torch:", torch.__version__, "cuda", torch.version.cuda)
print("triton:", triton.__version__)
print("transformers:", transformers.__version__)'
    echo "nvcc: $(nvcc --version | grep -E 'cuda_[0-9]' | head -n1)"
    echo "compute-sanitizer: $(compute-sanitizer --version | grep -E '^Version' | head -n1)"
    echo "g++: $(g++ --version | head -n1)"
    echo "docker: $(docker --version 2>&1)"
    echo "make: $(make --version | head -n1)"
} | tee results/setup_versions.txt

say "Done"

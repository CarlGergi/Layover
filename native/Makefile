# make host        the transfer library without CUDA, for tests on any machine
# make cuda        the transfer library with nvcc, needs the CUDA toolkit
# make cuda-debug  the same with host symbols, for compute-sanitizer backtraces
# make check       the C++ checks of the host build under the address and undefined sanitizers
#
# Host and CUDA builds and different machines can share one tree (a Mac and its Linux Docker
# container mount the same directory), so each gets a directory that layover/kv/native.py
# can find. CUDA_HOME's nvcc wins over one found on PATH.
PLATFORM := $(shell uname -s)-$(shell uname -m)
HOST_DIR := build/host-$(PLATFORM)
CUDA_DIR := build/cuda-$(PLATFORM)
DEBUG_DIR := build/cuda-debug-$(PLATFORM)
LIBRARY := liblayover_transfer.so

CXX ?= c++
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(shell test -x $(CUDA_HOME)/bin/nvcc && echo $(CUDA_HOME)/bin/nvcc || echo nvcc)
# L4 and RTX 4090 are both sm_89.
CUDA_ARCH ?= sm_89

COMMON := -std=c++17 -fPIC -fvisibility=hidden
HEADERS := layover_transfer.h pack.h transfer_core.h
# -lineinfo costs nothing at runtime and lets compute-sanitizer name source lines.
# Torch's libcudart is in the process's global scope, because torch_global_deps links it and
# loads with RTLD_GLOBAL. Lookup searches that scope first, so a static runtime with exported
# symbols would bind to torch's runtime instead, whatever its version. --exclude-libs keeps
# ours private, and the check after linking refuses a library where that did not hold.
NVCC_FLAGS := -std=c++17 -Xcompiler=-fPIC,-fvisibility=hidden -O3 -arch=$(CUDA_ARCH) -lineinfo -cudart static -shared \
	-Xlinker=--exclude-libs,ALL
define refuse_exported_runtime
	@if nm -D --defined-only $@ | grep -E ' (__)?cuda'; then rm -f $@; echo "$@ exports the CUDA runtime" >&2; exit 1; fi
endef

.PHONY: host cuda cuda-debug check clean

host: $(HOST_DIR)/$(LIBRARY)
cuda: $(CUDA_DIR)/$(LIBRARY)
cuda-debug: $(DEBUG_DIR)/$(LIBRARY)

$(HOST_DIR)/$(LIBRARY): transfer_host.cc host_device.h $(HEADERS)
	mkdir -p $(@D)
	$(CXX) $(COMMON) -O2 -Wall -Wextra -pthread -shared -o $@ transfer_host.cc

$(CUDA_DIR)/$(LIBRARY): transfer_cuda.cu $(HEADERS) Makefile
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -o $@ transfer_cuda.cu
	$(refuse_exported_runtime)

$(DEBUG_DIR)/$(LIBRARY): transfer_cuda.cu $(HEADERS) Makefile
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -g -Xlinker=--export-dynamic -o $@ transfer_cuda.cu
	$(refuse_exported_runtime)

$(HOST_DIR)/host_check: host_check.cc transfer_host.cc host_device.h $(HEADERS)
	mkdir -p $(@D)
	$(CXX) -std=c++17 -g -O1 -Wall -Wextra -pthread -fsanitize=address,undefined -fno-sanitize-recover=undefined \
		-o $@ host_check.cc transfer_host.cc

check: $(HOST_DIR)/host_check
	$(HOST_DIR)/host_check

clean:
	rm -rf build

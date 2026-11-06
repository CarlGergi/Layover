"""ctypes binding of the native transfer library. See native/layover_transfer.h."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from ctypes import POINTER, byref, c_char_p, c_int, c_uint32, c_uint64, c_void_p
from pathlib import Path

NATIVE_DIR = Path(__file__).resolve().parents[2] / "native"
ABI_VERSION = 2
LIBRARY_NAME = "liblayover_transfer.so"
SLOT_FILLED, SLOT_RELEASED = 0, 1
_ERR_NOMEM = 4


class NativeError(RuntimeError):
    """The transfer library reported a failure, with its message."""


class HostMemoryError(NativeError):
    """Host memory could not be pinned. Nothing was enqueued and the engine stays usable."""


class NativeUnavailable(RuntimeError):
    """The transfer library has not been built for this machine."""


class Stats(ctypes.Structure):
    _fields_ = [
        ("kernels", c_uint64),
        ("memcpys", c_uint64),
        ("ids_uploads", c_uint64),
        ("stream_waits", c_uint64),
        ("bytes_to_host", c_uint64),
        ("bytes_to_device", c_uint64),
        ("transfers", c_uint64),
        ("host_bytes_live", c_uint64),
        ("host_bytes_held", c_uint64),
        ("host_pins", c_uint64),
        ("host_pin_bytes", c_uint64),
        ("host_pin_ns", c_uint64),
        ("host_pin_max_ns", c_uint64),
        ("host_pin_failures", c_uint64),
    ]

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name, _ in self._fields_}


HOST_KERNEL = ctypes.CFUNCTYPE(c_int, c_void_p)
_SIGNATURES = {
    "lt_create": (c_int, c_uint64, c_uint32, c_uint32, c_uint64, POINTER(c_void_p)),
    "lt_destroy": (c_void_p,),
    "lt_prepare_staging": (c_void_p,),
    "lt_get_stats": (c_void_p, POINTER(Stats)),
    "lt_host_alloc": (c_void_p, c_uint64, POINTER(c_void_p)),
    "lt_host_free": (c_void_p, c_void_p),
    "lt_out_gather": (c_void_p, c_void_p, c_uint64, POINTER(c_uint32), c_uint32, c_void_p, POINTER(c_void_p)),
    "lt_in_scatter": (c_void_p, c_void_p, c_uint64, POINTER(c_uint32), c_uint32, c_void_p, POINTER(c_void_p)),
    "lt_out_per_block": (c_void_p, c_void_p, c_uint64, POINTER(c_uint32), c_uint32, c_void_p, POINTER(c_void_p)),
    "lt_in_per_block": (c_void_p, c_void_p, c_uint64, POINTER(c_uint32), c_uint32, c_void_p, POINTER(c_void_p)),
    "lt_ext_wait": (c_void_p, c_void_p, c_uint32, c_int),
    "lt_ext_record": (c_void_p, c_void_p, c_uint32, c_int),
    "lt_ext_copy_out": (c_void_p, c_void_p, c_void_p, c_uint64, c_uint32),
    "lt_ext_copy_in": (c_void_p, c_void_p, c_void_p, c_uint64, c_uint32),
    "lt_ext_finish": (c_void_p, c_void_p, POINTER(c_void_p)),
    "lt_poll": (c_void_p, c_void_p, POINTER(c_int)),
    "lt_wait": (c_void_p, c_void_p),
    "lt_release": (c_void_p, c_void_p),
    "lt_host_stream": (c_void_p, POINTER(c_void_p)),
    "lt_host_call": (c_void_p, c_void_p, HOST_KERNEL, c_void_p),
    "lt_host_set_paused": (c_void_p, c_int),
    "lt_host_set_copy_delay_us": (c_void_p, c_uint32),
    "lt_host_set_pin_limit": (c_void_p, c_uint64),
}

_loaded: dict[str, ctypes.CDLL] = {}


def library_path(flavor: str) -> Path:
    override = os.environ.get("LAYOVER_NATIVE_LIB")
    if override:
        return Path(override)
    return NATIVE_DIR / "build" / f"{flavor}-{platform.system()}-{platform.machine()}" / LIBRARY_NAME


def build_host() -> Path:
    """Build the host flavor if needed, for tests. make decides whether anything changed."""
    subprocess.run(["make", "-C", str(NATIVE_DIR), "host"], check=True, capture_output=True, text=True)
    return library_path("host")


def load_library(flavor: str) -> ctypes.CDLL:
    """Load a flavor, refusing a stale or mismatched build instead of misreading its ABI."""
    if flavor in _loaded:
        return _loaded[flavor]
    path = library_path(flavor)
    if not path.exists():
        raise NativeUnavailable(f"{path} does not exist: run 'make -C native {flavor}'")
    try:
        # ctypes defaults to RTLD_LOCAL and the GIL is released during calls, so a long wait
        # in lt_wait does not stall other Python threads.
        lib = ctypes.CDLL(str(path))
    except OSError as error:
        raise NativeUnavailable(f"{path} cannot be loaded: {error}") from error
    lib.lt_abi_version.restype = c_int
    lib.lt_flavor.restype = c_char_p
    lib.lt_last_error.restype = c_char_p
    if lib.lt_abi_version() != ABI_VERSION:
        raise NativeUnavailable(f"{path} has ABI {lib.lt_abi_version()}, expected {ABI_VERSION}: rebuild it")
    built = lib.lt_flavor().decode()
    if built != flavor:
        raise NativeUnavailable(f"{path} is the {built} flavor, expected {flavor}")
    for name, argtypes in _SIGNATURES.items():
        function = getattr(lib, name)
        function.argtypes = list(argtypes)
        function.restype = c_int
    _loaded[flavor] = lib
    return lib


def _pointer(value: int | None) -> c_void_p:
    return c_void_p(value)


class Engine:
    """One lt_engine. Only the thread driving transfers may call it."""

    def __init__(
        self,
        flavor: str = "host",
        device: int = 0,
        staging_bytes: int = 128 << 20,
        num_staging: int = 2,
        ids_capacity: int = 1 << 20,
        host_cache_bytes: int = 8 << 30,
    ):
        self.flavor = flavor
        self.staging_bytes = staging_bytes
        self.num_staging = num_staging
        self._lib = load_library(flavor)
        handle = c_void_p()
        self._check(self._lib.lt_create(device, staging_bytes, num_staging, ids_capacity, host_cache_bytes, byref(handle)))
        self._handle = handle
        self._kernel_failure: BaseException | None = None

    def _check(self, status: int) -> None:
        if status != 0:
            message = self._lib.lt_last_error().decode()
            if status == _ERR_NOMEM:
                raise HostMemoryError(message)
            failure = getattr(self, "_kernel_failure", None)
            if failure is not None:
                raise NativeError(f"{message}: {failure!r}") from failure
            raise NativeError(message)

    def _require_open(self) -> c_void_p:
        if self._handle is None:
            raise NativeError("the transfer engine is closed")
        return self._handle

    @property
    def closed(self) -> bool:
        return self._handle is None

    def destroy(self) -> None:
        """Waits for transfers in flight and frees everything. Safe to repeat."""
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        self._check(self._lib.lt_destroy(handle))

    def prepare_staging(self) -> None:
        self._check(self._lib.lt_prepare_staging(self._require_open()))

    def stats(self) -> dict[str, int]:
        out = Stats()
        self._check(self._lib.lt_get_stats(self._require_open(), byref(out)))
        return out.as_dict()

    def host_alloc(self, nbytes: int) -> int:
        pointer = c_void_p()
        self._check(self._lib.lt_host_alloc(self._require_open(), nbytes, byref(pointer)))
        return pointer.value

    def host_free(self, pointer: int) -> None:
        self._check(self._lib.lt_host_free(self._require_open(), _pointer(pointer)))

    def _blocks_call(self, function, pool: int, block_bytes: int, ids: list[int], host: int) -> int:
        array = (c_uint32 * len(ids))(*ids)
        transfer = c_void_p()
        self._check(function(self._require_open(), _pointer(pool), block_bytes, array, len(ids), _pointer(host), byref(transfer)))
        return transfer.value

    def out_gather(self, pool: int, block_bytes: int, ids: list[int], host: int) -> int:
        return self._blocks_call(self._lib.lt_out_gather, pool, block_bytes, ids, host)

    def in_scatter(self, pool: int, block_bytes: int, ids: list[int], host: int) -> int:
        return self._blocks_call(self._lib.lt_in_scatter, pool, block_bytes, ids, host)

    def out_per_block(self, pool: int, block_bytes: int, ids: list[int], host: int) -> int:
        return self._blocks_call(self._lib.lt_out_per_block, pool, block_bytes, ids, host)

    def in_per_block(self, pool: int, block_bytes: int, ids: list[int], host: int) -> int:
        return self._blocks_call(self._lib.lt_in_per_block, pool, block_bytes, ids, host)

    def ext_wait(self, stream: int, slot: int, which: int) -> None:
        self._check(self._lib.lt_ext_wait(self._require_open(), _pointer(stream), slot, which))

    def ext_record(self, stream: int, slot: int, which: int) -> None:
        self._check(self._lib.lt_ext_record(self._require_open(), _pointer(stream), slot, which))

    def ext_copy_out(self, staging: int, host: int, nbytes: int, slot: int) -> None:
        self._check(self._lib.lt_ext_copy_out(self._require_open(), _pointer(staging), _pointer(host), nbytes, slot))

    def ext_copy_in(self, staging: int, host: int, nbytes: int, slot: int) -> None:
        self._check(self._lib.lt_ext_copy_in(self._require_open(), _pointer(staging), _pointer(host), nbytes, slot))

    def ext_finish(self, stream: int | None) -> int:
        transfer = c_void_p()
        self._check(self._lib.lt_ext_finish(self._require_open(), _pointer(stream), byref(transfer)))
        return transfer.value

    def poll(self, transfer: int) -> bool:
        done = c_int()
        self._check(self._lib.lt_poll(self._require_open(), _pointer(transfer), byref(done)))
        return bool(done.value)

    def wait(self, transfer: int) -> None:
        self._check(self._lib.lt_wait(self._require_open(), _pointer(transfer)))

    def release(self, transfer: int) -> None:
        self._check(self._lib.lt_release(self._require_open(), _pointer(transfer)))

    def host_stream(self) -> int:
        stream = c_void_p()
        self._check(self._lib.lt_host_stream(self._require_open(), byref(stream)))
        return stream.value

    def host_call(self, stream: int, fn) -> HOST_KERNEL:
        """Run fn on a host stream. Keep the returned callback alive until it has run."""

        def trampoline(_context):
            try:
                fn()
                return 0
            except BaseException as failure:  # An exception must not cross into C++.
                self._kernel_failure = failure
                return 1

        callback = HOST_KERNEL(trampoline)
        self._check(self._lib.lt_host_call(self._require_open(), _pointer(stream), callback, None))
        return callback

    def set_paused(self, paused: bool) -> None:
        self._check(self._lib.lt_host_set_paused(self._require_open(), int(paused)))

    def set_copy_delay_us(self, delay_us: int) -> None:
        self._check(self._lib.lt_host_set_copy_delay_us(self._require_open(), delay_us))

    def set_pin_limit(self, nbytes: int) -> None:
        self._check(self._lib.lt_host_set_pin_limit(self._require_open(), nbytes))

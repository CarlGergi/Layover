// C ABI of the KV transfer library, loaded from Python with ctypes (layover/kv/native.py).
//
// Calls are not thread-safe: one engine belongs to the thread that runs the scheduler. A
// transfer's stream work is fully enqueued when its call returns, so a transfer makes
// progress without being polled and a slow Python step cannot stall a DMA.
#ifndef LAYOVER_TRANSFER_H
#define LAYOVER_TRANSFER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LT_ABI_VERSION 2

enum {
    LT_OK = 0,
    LT_ERR_INVALID = 1,      // Bad argument or misuse; the engine state is unchanged.
    LT_ERR_DEVICE = 2,       // The device reported a failure; the engine refuses further work.
    LT_ERR_UNSUPPORTED = 3,  // The entry point does not exist in this build flavor.
    LT_ERR_NOMEM = 4,        // lt_host_alloc only: no host memory; the engine stays usable.
};

// Events of an externally staged slot, see lt_ext_wait.
enum { LT_SLOT_FILLED = 0, LT_SLOT_RELEASED = 1 };

typedef struct lt_engine lt_engine;
typedef struct lt_transfer lt_transfer;

typedef struct {
    uint64_t kernels;          // Gather and scatter launches.
    uint64_t memcpys;          // KV copies on the copy stream.
    uint64_t ids_uploads;      // Block-id uploads, which precede kernels.
    uint64_t stream_waits;
    uint64_t bytes_to_host;
    uint64_t bytes_to_device;
    uint64_t transfers;
    uint64_t host_bytes_live;  // Pinned bytes handed out by lt_host_alloc and not returned.
    uint64_t host_bytes_held;  // Pinned bytes the engine has allocated, cached ones included.
    uint64_t host_pins;        // Allocations the cache could not serve, each stalling its caller.
    uint64_t host_pin_bytes;
    uint64_t host_pin_ns;      // Time those stalls took.
    uint64_t host_pin_max_ns;
    uint64_t host_pin_failures;  // lt_host_alloc calls that returned LT_ERR_NOMEM.
} lt_stats;

int lt_abi_version(void);
const char* lt_flavor(void);  // "host" or "cuda".
// Text of this thread's most recent failure.
const char* lt_last_error(void);

// staging_bytes is per staging buffer. ids_capacity bounds the block ids of transfers not
// yet finished. Returned host memory stays pinned for reuse up to host_cache_bytes; past that
// the buffers returned longest ago go back to the system.
int lt_create(int device, uint64_t staging_bytes, uint32_t num_staging, uint32_t ids_capacity,
              uint64_t host_cache_bytes, lt_engine** out);
// Safe with transfers in flight: waits for them, then frees everything the engine allocated.
// Every lt_transfer and host buffer of the engine is invalid afterwards.
int lt_destroy(lt_engine* engine);
// Allocates the staging buffers now instead of on first use.
int lt_prepare_staging(lt_engine* engine);
int lt_get_stats(lt_engine* engine, lt_stats* out);

// Pinned host memory for transfers, reused across calls because pinning is slow and stalls
// the caller. When the host is out of memory it releases the cache, tries once more, and
// otherwise returns LT_ERR_NOMEM, which leaves the engine usable.
int lt_host_alloc(lt_engine* engine, uint64_t bytes, void** out);
int lt_host_free(lt_engine* engine, void* pointer);

// Gather `count` blocks into staging and DMA them to `host`, in the order of `ids`. pool
// and host must be 8-byte aligned and block_bytes a multiple of 8.
int lt_out_gather(lt_engine* engine, const void* pool, uint64_t block_bytes, const uint32_t* ids,
                  uint32_t count, void* host, lt_transfer** out);
int lt_in_scatter(lt_engine* engine, void* pool, uint64_t block_bytes, const uint32_t* ids,
                  uint32_t count, const void* host, lt_transfer** out);

// The rejected design, kept for the benchmark: one copy per block, no staging, no kernel.
int lt_out_per_block(lt_engine* engine, const void* pool, uint64_t block_bytes, const uint32_t* ids,
                     uint32_t count, void* host, lt_transfer** out);
int lt_in_per_block(lt_engine* engine, void* pool, uint64_t block_bytes, const uint32_t* ids,
                    uint32_t count, const void* host, lt_transfer** out);

// Staging owned by the caller, for pipelines whose kernels run elsewhere (FP8 in Triton).
// The caller's kernels go on `stream`, never null, ordered against the copy stream by per-slot
// events:
//   out: ext_wait(stream, slot, RELEASED); kernel; ext_record(stream, slot, FILLED);
//        ext_copy_out(slot, ...); then ext_finish(NULL).
//   in:  ext_copy_in(slot, ...); ext_wait(stream, slot, FILLED); kernel;
//        ext_record(stream, slot, RELEASED); then ext_finish(stream).
int lt_ext_wait(lt_engine* engine, void* stream, uint32_t slot, int which);
int lt_ext_record(lt_engine* engine, void* stream, uint32_t slot, int which);
int lt_ext_copy_out(lt_engine* engine, const void* staging, void* host, uint64_t bytes, uint32_t slot);
int lt_ext_copy_in(lt_engine* engine, void* staging, const void* host, uint64_t bytes, uint32_t slot);
// Completion after everything enqueued so far on `stream`, or on the copy stream if NULL.
int lt_ext_finish(lt_engine* engine, void* stream, lt_transfer** out);

// Non-blocking. *done becomes 1 when the transfer's last operation has completed.
int lt_poll(lt_engine* engine, lt_transfer* transfer, int* done);
int lt_wait(lt_engine* engine, lt_transfer* transfer);
// Frees a finished transfer. A transfer in flight is refused.
int lt_release(lt_engine* engine, lt_transfer* transfer);

// Host flavor only: a stream for caller kernels, a way to run one on it, and test controls
// that hold streams still or slow the copy stream, so ordering bugs corrupt data visibly.
typedef int (*lt_host_kernel)(void* context);
int lt_host_stream(lt_engine* engine, void** out);
int lt_host_call(lt_engine* engine, void* stream, lt_host_kernel kernel, void* context);
int lt_host_set_paused(lt_engine* engine, int paused);
int lt_host_set_copy_delay_us(lt_engine* engine, uint32_t delay_us);
// Host allocations past this many bytes in total fail as if the machine had run out.
int lt_host_set_pin_limit(lt_engine* engine, uint64_t bytes);

#ifdef __cplusplus
}
#endif

#endif  // LAYOVER_TRANSFER_H

// The transfer engine on a CUDA device: gather and scatter kernels and the stream, event,
// and memory calls behind the device interface in transfer_core.h.
//
// This library links its own CUDA runtime and keeps it private (see Makefile). cudaSetDevice
// makes the device's primary context current, which is one object per process whichever
// runtime asks, so device pointers from torch are valid here and check_pool proves it on
// every transfer. Stream and event handles are the driver's, which is why a torch.cuda.Stream
// works in lt_ext_*. Torch's default stream stays out of the pipeline by design.
#include <cuda_runtime.h>
#include <stdint.h>

#include <stdexcept>
#include <string>

#include "layover_transfer.h"
#include "pack.h"

namespace lt {

__global__ void gather_kernel(const uint8_t* __restrict__ pool, const uint32_t* __restrict__ ids, uint8_t* __restrict__ staging,
                              uint64_t block_bytes) {
    gather_lane(pool, ids, staging, block_bytes, blockIdx.y, blockIdx.x, threadIdx.x, blockDim.x);
}

__global__ void scatter_kernel(uint8_t* __restrict__ pool, const uint32_t* __restrict__ ids, const uint8_t* __restrict__ staging,
                               uint64_t block_bytes) {
    scatter_lane(pool, ids, staging, block_bytes, blockIdx.y, blockIdx.x, threadIdx.x, blockDim.x);
}

class CudaDevice {
   public:
    using Stream = cudaStream_t;
    using Event = cudaEvent_t;

    static const char* flavor() { return "cuda"; }

    explicit CudaDevice(int device) : device_(device) {
        // A driver older than this toolkit's runtime fails here, and the versions say why.
        check(cudaDriverGetVersion(&driver_version_), "cudaDriverGetVersion");
        check(cudaRuntimeGetVersion(&runtime_version_), "cudaRuntimeGetVersion");
        activate();
    }

    // The scheduler may run on any thread, and a CUDA runtime's current device is per thread.
    void activate() {
        const cudaError_t status = cudaSetDevice(device_);
        if (status != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice(" + std::to_string(device_) + "): " + cudaGetErrorName(status) + ": " +
                                     cudaGetErrorString(status) + " (driver supports CUDA API " + std::to_string(driver_version_) +
                                     ", library linked with " + std::to_string(runtime_version_) + ")");
        }
    }

    void prepare_teardown() {}

    // A pointer this runtime cannot see as this device's memory means the shared-context
    // assumption above is false for this process, and a kernel would fault the whole context.
    void check_pool(const void* pool) {
        cudaPointerAttributes attributes{};
        check(cudaPointerGetAttributes(&attributes, pool), "cudaPointerGetAttributes");
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != device_) {
            throw std::invalid_argument("pool is not memory of CUDA device " + std::to_string(device_) +
                                        ": it is on another device, on the host, or in a context this runtime does not share");
        }
    }

    void* device_alloc(uint64_t bytes) {
        void* pointer = nullptr;
        check(cudaMalloc(&pointer, bytes), "cudaMalloc");
        return pointer;
    }

    void device_free(void* pointer) { check(cudaFree(pointer), "cudaFree"); }

    // Null when out of memory, which unlike any other failure the engine survives.
    void* host_alloc(uint64_t bytes) {
        void* pointer = nullptr;
        const cudaError_t status = cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault);
        if (status == cudaErrorMemoryAllocation) return nullptr;
        check(status, "cudaHostAlloc");
        return pointer;
    }

    void host_free(void* pointer) { check(cudaFreeHost(pointer), "cudaFreeHost"); }

    // Torch's default stream is the legacy default stream, which implicitly synchronizes with
    // every blocking stream. Without cudaStreamNonBlocking, each decode kernel would wait for
    // the transfer in progress and the overlap this library exists for would silently vanish.
    Stream stream_create() {
        Stream stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        return stream;
    }

    void stream_sync(Stream stream) { check(cudaStreamSynchronize(stream), "cudaStreamSynchronize"); }
    void stream_destroy(Stream stream) { check(cudaStreamDestroy(stream), "cudaStreamDestroy"); }
    static Stream stream_from_handle(void* handle) { return static_cast<Stream>(handle); }

    Event event_create() {
        Event event = nullptr;
        check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags");
        return event;
    }

    void event_destroy(Event event) { check(cudaEventDestroy(event), "cudaEventDestroy"); }
    void event_record(Event event, Stream stream) { check(cudaEventRecord(event, stream), "cudaEventRecord"); }
    void stream_wait(Stream stream, Event event) { check(cudaStreamWaitEvent(stream, event, 0), "cudaStreamWaitEvent"); }

    bool event_query(Event event) {
        const cudaError_t status = cudaEventQuery(event);
        if (status == cudaErrorNotReady) return false;
        check(status, "cudaEventQuery");
        return true;
    }

    void event_sync(Event event) { check(cudaEventSynchronize(event), "cudaEventSynchronize"); }

    void memcpy_to_host(Stream stream, void* dst, const void* src, uint64_t bytes) {
        check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(device to host)");
    }

    void memcpy_to_device(Stream stream, void* dst, const void* src, uint64_t bytes) {
        check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(host to device)");
    }

    void launch_gather(Stream stream, const uint8_t* pool, const uint32_t* ids, uint8_t* staging, uint64_t block_bytes, uint32_t blocks) {
        gather_kernel<<<dim3(chunks_for(block_bytes), blocks), dim3(kLanes), 0, stream>>>(pool, ids, staging, block_bytes);
        check(cudaGetLastError(), "launch gather_kernel");
    }

    void launch_scatter(Stream stream, uint8_t* pool, const uint32_t* ids, const uint8_t* staging, uint64_t block_bytes, uint32_t blocks) {
        scatter_kernel<<<dim3(chunks_for(block_bytes), blocks), dim3(kLanes), 0, stream>>>(pool, ids, staging, block_bytes);
        check(cudaGetLastError(), "launch scatter_kernel");
    }

   private:
    static void check(cudaError_t status, const char* call) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(call) + ": " + cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }

    int device_;
    int driver_version_ = 0;
    int runtime_version_ = 0;
};

}  // namespace lt

using Device = lt::CudaDevice;

#include "transfer_core.h"

extern "C" {

static int host_only(const char* name) {
    lt::last_error = std::string(name) + " exists only in the host build";
    return LT_ERR_UNSUPPORTED;
}

LT_EXPORT int lt_host_stream(lt_engine*, void**) { return host_only("lt_host_stream"); }
LT_EXPORT int lt_host_call(lt_engine*, void*, lt_host_kernel, void*) { return host_only("lt_host_call"); }
LT_EXPORT int lt_host_set_paused(lt_engine*, int) { return host_only("lt_host_set_paused"); }
LT_EXPORT int lt_host_set_copy_delay_us(lt_engine*, uint32_t) { return host_only("lt_host_set_copy_delay_us"); }
LT_EXPORT int lt_host_set_pin_limit(lt_engine*, uint64_t) { return host_only("lt_host_set_pin_limit"); }

}  // extern "C"

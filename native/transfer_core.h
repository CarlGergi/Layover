// The transfer engine, written against a device interface so the same segmenting, staging,
// and event logic runs on a GPU (transfer_cuda.cu) and on host threads (transfer_host.cc).
// Include after defining the Device type; this also defines the C ABI on top of it.
//
// The pipeline, for each segment on staging slot s, out direction:
//   compute stream: wait s.released; gather into s; record s.filled
//   copy stream:    wait s.filled; DMA s to host; record s.released
// and the reverse going in. Everything is enqueued up front and streams run in FIFO order,
// so the wait always captures the most recent record, every dependency points backwards,
// and a cycle is impossible. With two slots, a gather overlaps the previous slot's DMA.
#ifndef LAYOVER_TRANSFER_CORE_H
#define LAYOVER_TRANSFER_CORE_H

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layover_transfer.h"
#include "pack.h"

namespace lt {

// A device failure. Anything else thrown is a caller mistake (std::invalid_argument).
struct DeviceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Host memory ran out. No stream work was enqueued, so unlike DeviceError it is survivable.
struct HostOutOfMemory : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline thread_local std::string last_error;

// Size classes let a freed buffer serve the next offload of similar size. Four per doubling
// waste at most a quarter of what gets pinned; whole powers of two would pin 16 GiB for a
// 10 GiB context.
inline uint64_t host_size_class(uint64_t bytes) {
    const uint64_t smallest = uint64_t(1) << 20;
    if (bytes <= smallest) return smallest;
    uint64_t power = smallest;
    while (power * 2 < bytes) power *= 2;
    const uint64_t step = power / 4;
    return (bytes + step - 1) / step * step;
}

template <class Dev>
class Engine {
    using Stream = typename Dev::Stream;
    using Event = typename Dev::Event;

    struct CachedHost {
        void* pointer;
        uint64_t age;
    };

    struct Slot {
        void* staging = nullptr;
        Event filled{};
        Event released{};
    };

   public:
    struct Transfer {
        Event done{};
        uint64_t ring_end = 0;
        bool holds_ring = false;
        bool finished = false;
        bool released = false;
    };

    Engine(int device, uint64_t staging_bytes, uint32_t num_staging, uint32_t ids_capacity, uint64_t host_cache_bytes)
        : dev_(device), staging_bytes_(staging_bytes), ids_capacity_(ids_capacity), host_cache_bytes_(host_cache_bytes) {
        if (staging_bytes == 0 || num_staging == 0 || ids_capacity == 0) {
            throw std::invalid_argument("staging_bytes, num_staging, and ids_capacity must be positive");
        }
        // Each step leaves the object destructible, so a failure part-way frees what exists.
        try {
            compute_ = dev_.stream_create();
            have_compute_ = true;
            copy_ = dev_.stream_create();
            have_copy_ = true;
            slots_.resize(num_staging);
            ext_slots_.resize(num_staging);
            for (std::vector<Slot>* group : {&slots_, &ext_slots_}) {
                for (Slot& slot : *group) {
                    slot.filled = dev_.event_create();
                    slot.released = dev_.event_create();
                    events_.push_back(slot.filled);
                    events_.push_back(slot.released);
                }
            }
            ids_host_ = static_cast<uint32_t*>(dev_.host_alloc(uint64_t(ids_capacity) * sizeof(uint32_t)));
            if (ids_host_ == nullptr) throw std::runtime_error("cannot pin the block id ring");
            ids_device_ = static_cast<uint32_t*>(dev_.device_alloc(uint64_t(ids_capacity) * sizeof(uint32_t)));
        } catch (...) {
            teardown();
            throw;
        }
    }

    ~Engine() { teardown(); }

    void prepare_staging() {
        for (Slot& slot : slots_) {
            if (slot.staging == nullptr) slot.staging = dev_.device_alloc(staging_bytes_);
        }
    }

    // Reports the first failure seen while shutting down, after freeing everything anyway.
    std::string shutdown() {
        teardown();
        return teardown_error_;
    }

    lt_stats stats() const {
        lt_stats out = stats_;
        out.host_bytes_held = host_held_;
        return out;
    }

    // Pinning runs inline, so a miss here stalls the calling step. The stats make that visible.
    void* host_alloc(uint64_t bytes) {
        if (bytes == 0 || bytes > (uint64_t(1) << 62)) throw std::invalid_argument("cannot allocate that many bytes");
        const uint64_t rounded = host_size_class(bytes);
        void* pointer = nullptr;
        auto cached = host_free_.find(rounded);
        if (cached != host_free_.end()) {
            pointer = cached->second.back().pointer;
            cached->second.pop_back();
            if (cached->second.empty()) host_free_.erase(cached);
            host_cached_ -= rounded;
        } else {
            pointer = pin(rounded);
            if (pointer == nullptr && !host_free_.empty()) {
                // Cached buffers are the only memory this engine can give back.
                while (!host_free_.empty()) evict_oldest();
                pointer = pin(rounded);
            }
            if (pointer == nullptr) {
                stats_.host_pin_failures++;
                throw HostOutOfMemory("cannot pin " + std::to_string(rounded) + " bytes of host memory");
            }
            host_held_ += rounded;
        }
        host_live_[pointer] = rounded;
        stats_.host_bytes_live += rounded;
        return pointer;
    }

    void host_free(void* pointer) {
        auto live = host_live_.find(pointer);
        if (live == host_live_.end()) throw std::invalid_argument("not a live host buffer of this engine");
        const uint64_t rounded = live->second;
        host_live_.erase(live);
        stats_.host_bytes_live -= rounded;
        if (rounded > host_cache_bytes_) {
            dev_.host_free(pointer);
            host_held_ -= rounded;
            return;
        }
        // Oldest out: contexts only grow, so what sits longest in the cache is sizes no session
        // has any more. Keeping it instead would turn every larger offload into a new pin.
        while (host_cached_ + rounded > host_cache_bytes_) evict_oldest();
        host_free_[rounded].push_back({pointer, ++host_free_clock_});
        host_cached_ += rounded;
    }

    Transfer* out_gather(const uint8_t* pool, uint64_t block_bytes, const uint32_t* ids, uint32_t count, uint8_t* host) {
        check_gather_args(pool, block_bytes, ids, count, host);
        prepare_staging();
        uint64_t ring_end = 0;
        const uint32_t* device_ids = upload_ids(ids, count, &ring_end);
        Transfer* transfer = begin(ring_end);
        const uint32_t per_segment = blocks_per_segment(block_bytes);
        for (uint64_t start = 0; start < count; start += per_segment) {
            const uint32_t blocks = uint32_t(std::min<uint64_t>(per_segment, count - start));
            Slot& slot = next_slot();
            const uint64_t bytes = uint64_t(blocks) * block_bytes;
            wait(compute_, slot.released);
            dev_.launch_gather(compute_, pool, device_ids + start, static_cast<uint8_t*>(slot.staging), block_bytes, blocks);
            stats_.kernels++;
            dev_.event_record(slot.filled, compute_);
            wait(copy_, slot.filled);
            dev_.memcpy_to_host(copy_, host + start * block_bytes, slot.staging, bytes);
            count_copy_out(bytes);
            dev_.event_record(slot.released, copy_);
        }
        dev_.event_record(transfer->done, copy_);
        return transfer;
    }

    Transfer* in_scatter(uint8_t* pool, uint64_t block_bytes, const uint32_t* ids, uint32_t count, const uint8_t* host) {
        check_gather_args(pool, block_bytes, ids, count, host);
        prepare_staging();
        uint64_t ring_end = 0;
        const uint32_t* device_ids = upload_ids(ids, count, &ring_end);
        Transfer* transfer = begin(ring_end);
        const uint32_t per_segment = blocks_per_segment(block_bytes);
        for (uint64_t start = 0; start < count; start += per_segment) {
            const uint32_t blocks = uint32_t(std::min<uint64_t>(per_segment, count - start));
            Slot& slot = next_slot();
            const uint64_t bytes = uint64_t(blocks) * block_bytes;
            wait(copy_, slot.released);
            dev_.memcpy_to_device(copy_, slot.staging, host + start * block_bytes, bytes);
            count_copy_in(bytes);
            dev_.event_record(slot.filled, copy_);
            wait(compute_, slot.filled);
            dev_.launch_scatter(compute_, pool, device_ids + start, static_cast<const uint8_t*>(slot.staging), block_bytes, blocks);
            stats_.kernels++;
            dev_.event_record(slot.released, compute_);
        }
        dev_.event_record(transfer->done, compute_);
        return transfer;
    }

    Transfer* out_per_block(const uint8_t* pool, uint64_t block_bytes, const uint32_t* ids, uint32_t count, uint8_t* host) {
        check_block_args(pool, block_bytes, ids, count, host);
        Transfer* transfer = begin();
        for (uint32_t i = 0; i < count; ++i) {
            dev_.memcpy_to_host(copy_, host + uint64_t(i) * block_bytes, pool + uint64_t(ids[i]) * block_bytes, block_bytes);
            count_copy_out(block_bytes);
        }
        dev_.event_record(transfer->done, copy_);
        return transfer;
    }

    Transfer* in_per_block(uint8_t* pool, uint64_t block_bytes, const uint32_t* ids, uint32_t count, const uint8_t* host) {
        check_block_args(pool, block_bytes, ids, count, host);
        Transfer* transfer = begin();
        for (uint32_t i = 0; i < count; ++i) {
            dev_.memcpy_to_device(copy_, pool + uint64_t(ids[i]) * block_bytes, host + uint64_t(i) * block_bytes, block_bytes);
            count_copy_in(block_bytes);
        }
        dev_.event_record(transfer->done, copy_);
        return transfer;
    }

    void ext_wait(void* stream, uint32_t slot, int which) {
        check_ready();
        wait(caller_stream(stream), ext_event(slot, which));
    }

    void ext_record(void* stream, uint32_t slot, int which) {
        check_ready();
        dev_.event_record(ext_event(slot, which), caller_stream(stream));
    }

    void ext_copy_out(const void* staging, uint8_t* host, uint64_t bytes, uint32_t slot) {
        check_ready();
        check_ext_copy(staging, host, bytes);
        wait(copy_, ext_event(slot, LT_SLOT_FILLED));
        dev_.memcpy_to_host(copy_, host, staging, bytes);
        count_copy_out(bytes);
        dev_.event_record(ext_event(slot, LT_SLOT_RELEASED), copy_);
    }

    void ext_copy_in(void* staging, const uint8_t* host, uint64_t bytes, uint32_t slot) {
        check_ready();
        check_ext_copy(staging, host, bytes);
        wait(copy_, ext_event(slot, LT_SLOT_RELEASED));
        dev_.memcpy_to_device(copy_, staging, host, bytes);
        count_copy_in(bytes);
        dev_.event_record(ext_event(slot, LT_SLOT_FILLED), copy_);
    }

    Transfer* ext_finish(void* stream) {
        check_ready();
        Transfer* transfer = begin();
        dev_.event_record(transfer->done, stream == nullptr ? copy_ : caller_stream(stream));
        return transfer;
    }

    bool poll(Transfer* transfer) {
        check_live(transfer);
        if (!transfer->finished && dev_.event_query(transfer->done)) transfer->finished = true;
        if (transfer->finished) advance_ring();
        return transfer->finished;
    }

    void wait_for(Transfer* transfer) {
        check_live(transfer);
        if (!transfer->finished) {
            dev_.event_sync(transfer->done);
            transfer->finished = true;
        }
        advance_ring();
    }

    void release(Transfer* transfer) {
        check_live(transfer);
        if (!poll(transfer)) throw std::invalid_argument("transfer is still in flight");
        transfer->released = true;
        advance_ring();
    }

    void mark_failed() { failed_ = true; }
    Dev& device() { return dev_; }

   private:
    void check_ready() const {
        if (failed_) throw DeviceError("engine refused work after an earlier device failure");
    }

    void check_block_args(const void* pool, uint64_t block_bytes, const uint32_t* ids, uint32_t count, const void* host) {
        check_ready();
        if (pool == nullptr || host == nullptr || block_bytes == 0 || (ids == nullptr && count != 0)) {
            throw std::invalid_argument("null pool, host, or ids, or zero block_bytes");
        }
        dev_.check_pool(pool);
    }

    void check_gather_args(const void* pool, uint64_t block_bytes, const uint32_t* ids, uint32_t count, const void* host) {
        check_block_args(pool, block_bytes, ids, count, host);
        if (!words_aligned(pool, block_bytes) || !words_aligned(host, block_bytes)) {
            throw std::invalid_argument("pool and host must be 8-byte aligned and block_bytes a multiple of 8");
        }
        if (block_bytes > staging_bytes_) {
            throw std::invalid_argument("one block is larger than a staging buffer");
        }
        if (count > ids_capacity_) {
            throw std::invalid_argument("transfer has more blocks than ids_capacity");
        }
    }

    void check_ext_copy(const void* staging, const void* host, uint64_t bytes) const {
        if (staging == nullptr || host == nullptr || bytes == 0) {
            throw std::invalid_argument("null staging or host, or zero bytes");
        }
    }

    void check_live(Transfer* transfer) const {
        for (const auto& known : order_) {
            if (known.get() == transfer && !known->released) return;
        }
        throw std::invalid_argument("not a live transfer of this engine");
    }

    // Null is the legacy default stream on CUDA, which would serialize against this engine's
    // work in ways nothing here accounts for. The pipeline needs a stream of its own.
    Stream caller_stream(void* stream) const {
        if (stream == nullptr) throw std::invalid_argument("null stream");
        return Dev::stream_from_handle(stream);
    }

    uint32_t blocks_per_segment(uint64_t block_bytes) const {
        return uint32_t(std::min<uint64_t>(staging_bytes_ / block_bytes, kMaxBlocksPerLaunch));
    }

    Event ext_event(uint32_t slot, int which) {
        if (slot >= ext_slots_.size() || (which != LT_SLOT_FILLED && which != LT_SLOT_RELEASED)) {
            throw std::invalid_argument("bad slot or event");
        }
        return which == LT_SLOT_FILLED ? ext_slots_[slot].filled : ext_slots_[slot].released;
    }

    void wait(Stream stream, Event event) {
        dev_.stream_wait(stream, event);
        stats_.stream_waits++;
    }

    Slot& next_slot() {
        Slot& slot = slots_[next_slot_];
        next_slot_ = (next_slot_ + 1) % uint32_t(slots_.size());
        return slot;
    }

    void count_copy_out(uint64_t bytes) {
        stats_.memcpys++;
        stats_.bytes_to_host += bytes;
    }

    void count_copy_in(uint64_t bytes) {
        stats_.memcpys++;
        stats_.bytes_to_device += bytes;
    }

    // Only called with the completion about to be recorded and nothing in between that looks
    // at transfers: an event never recorded reads as complete. Waiting on the ring therefore
    // has to come before this.
    Transfer* begin(uint64_t ring_end = 0) {
        auto transfer = std::make_unique<Transfer>();
        transfer->done = dev_.event_create();
        transfer->ring_end = ring_end;
        transfer->holds_ring = ring_end != 0;
        Transfer* raw = transfer.get();
        order_.push_back(std::move(transfer));
        stats_.transfers++;
        return raw;
    }

    // Copies ids into a pinned ring, then uploads them on the compute stream. A pageable
    // source would make cudaMemcpyAsync synchronize with the stream first, which is the
    // stall this whole design avoids. The ring space stays reserved until the transfer ends,
    // so it does not matter how far behind the stream is.
    const uint32_t* upload_ids(const uint32_t* ids, uint32_t count, uint64_t* ring_end) {
        if (count == 0) return nullptr;
        const uint64_t capacity = ids_capacity_;
        uint64_t skip = 0;
        for (;;) {
            // An empty ring can start over, so a run as long as the ring always fits.
            if (ring_head_ == ring_tail_) ring_head_ = ring_tail_ = 0;
            const uint64_t position = ring_head_ % capacity;
            skip = position + count > capacity ? capacity - position : 0;
            if (ring_head_ + skip + count - ring_tail_ <= capacity) break;
            // Only a backlog this large gets here; the oldest transfer ends without us.
            wait_for_oldest();
        }
        ring_head_ += skip;
        const uint64_t offset = ring_head_ % capacity;
        memcpy(ids_host_ + offset, ids, uint64_t(count) * sizeof(uint32_t));
        ring_head_ += count;
        *ring_end = ring_head_;
        dev_.memcpy_to_device(compute_, ids_device_ + offset, ids_host_ + offset, uint64_t(count) * sizeof(uint32_t));
        stats_.ids_uploads++;
        return ids_device_ + offset;
    }

    void wait_for_oldest() {
        for (const auto& transfer : order_) {
            if (transfer->holds_ring) {
                wait_for(transfer.get());
                return;
            }
        }
        throw std::logic_error("id ring is full with nothing in flight");
    }

    // Ring space comes back in allocation order, whichever order transfers finish in.
    void advance_ring() {
        for (const auto& transfer : order_) {
            if (!transfer->finished) {
                if (!dev_.event_query(transfer->done)) break;
                transfer->finished = true;
            }
            if (transfer->holds_ring) {
                ring_tail_ = transfer->ring_end;
                transfer->holds_ring = false;
            }
        }
        while (!order_.empty() && order_.front()->finished && order_.front()->released) {
            dev_.event_destroy(order_.front()->done);
            order_.pop_front();
        }
    }

    void note_teardown_error(const std::exception& error) {
        if (teardown_error_.empty()) teardown_error_ = error.what();
    }

    template <class Fn>
    void quietly(Fn&& fn) {
        try {
            fn();
        } catch (const std::exception& error) {
            note_teardown_error(error);
        }
    }

    void teardown() {
        if (torn_down_) return;
        torn_down_ = true;
        dev_.prepare_teardown();
        // Transfers can end on a caller's stream, so waiting for each one covers streams this
        // engine cannot synchronize itself, and only then is any memory safe to free.
        for (const auto& transfer : order_) {
            if (!transfer->finished) quietly([&] { dev_.event_sync(transfer->done); });
        }
        if (have_compute_) quietly([&] { dev_.stream_sync(compute_); });
        if (have_copy_) quietly([&] { dev_.stream_sync(copy_); });
        for (const auto& transfer : order_) {
            quietly([&] { dev_.event_destroy(transfer->done); });
        }
        order_.clear();
        for (Event event : events_) {
            quietly([&] { dev_.event_destroy(event); });
        }
        events_.clear();
        for (Slot& slot : slots_) {
            if (slot.staging != nullptr) quietly([&] { dev_.device_free(slot.staging); });
        }
        slots_.clear();
        if (ids_device_ != nullptr) quietly([&] { dev_.device_free(ids_device_); });
        if (ids_host_ != nullptr) quietly([&] { dev_.host_free(ids_host_); });
        for (auto& entry : host_live_) {
            quietly([&] { dev_.host_free(entry.first); });
        }
        host_live_.clear();
        for (auto& entry : host_free_) {
            for (const CachedHost& cached : entry.second) {
                quietly([&] { dev_.host_free(cached.pointer); });
            }
        }
        host_free_.clear();
        if (have_copy_) quietly([&] { dev_.stream_destroy(copy_); });
        if (have_compute_) quietly([&] { dev_.stream_destroy(compute_); });
    }

    // Null when the host is out of memory. The time counts failures too: they stall the step.
    void* pin(uint64_t bytes) {
        const auto start = std::chrono::steady_clock::now();
        void* pointer = dev_.host_alloc(bytes);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const uint64_t ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        stats_.host_pins++;
        stats_.host_pin_ns += ns;
        stats_.host_pin_max_ns = std::max(stats_.host_pin_max_ns, ns);
        if (pointer != nullptr) stats_.host_pin_bytes += bytes;
        return pointer;
    }

    void evict_oldest() {
        auto oldest = host_free_.begin();
        for (auto entry = host_free_.begin(); entry != host_free_.end(); ++entry) {
            if (entry->second.front().age < oldest->second.front().age) oldest = entry;
        }
        const uint64_t rounded = oldest->first;
        void* pointer = oldest->second.front().pointer;
        oldest->second.pop_front();
        if (oldest->second.empty()) host_free_.erase(oldest);
        host_cached_ -= rounded;
        dev_.host_free(pointer);
        host_held_ -= rounded;
    }

    Dev dev_;
    uint64_t staging_bytes_;
    uint32_t ids_capacity_;
    uint64_t host_cache_bytes_;

    Stream compute_{};
    Stream copy_{};
    bool have_compute_ = false;
    bool have_copy_ = false;
    std::vector<Slot> slots_;
    std::vector<Slot> ext_slots_;
    std::vector<Event> events_;
    uint32_t next_slot_ = 0;

    uint32_t* ids_host_ = nullptr;
    uint32_t* ids_device_ = nullptr;
    uint64_t ring_head_ = 0;
    uint64_t ring_tail_ = 0;

    std::deque<std::unique_ptr<Transfer>> order_;
    std::map<void*, uint64_t> host_live_;
    // Per size class, oldest first. Never holds an empty class.
    std::map<uint64_t, std::deque<CachedHost>> host_free_;
    uint64_t host_free_clock_ = 0;
    uint64_t host_cached_ = 0;
    uint64_t host_held_ = 0;

    lt_stats stats_{};
    bool failed_ = false;
    bool torn_down_ = false;
    std::string teardown_error_;
};

using EngineImpl = Engine<Device>;

}  // namespace lt

struct lt_engine {
    lt::EngineImpl impl;
    template <class... Args>
    explicit lt_engine(Args&&... args) : impl(std::forward<Args>(args)...) {}
};

namespace lt {

template <class Fn>
inline int guarded(lt_engine* engine, Fn&& fn) {
    if (engine == nullptr) {
        last_error = "null engine";
        return LT_ERR_INVALID;
    }
    try {
        engine->impl.device().activate();
        fn();
        return LT_OK;
    } catch (const std::invalid_argument& error) {
        last_error = error.what();
        return LT_ERR_INVALID;
    } catch (const HostOutOfMemory& error) {
        last_error = error.what();
        return LT_ERR_NOMEM;
    } catch (const std::exception& error) {
        // A device failure can leave half a transfer enqueued, so nothing more is safe.
        last_error = error.what();
        engine->impl.mark_failed();
        return LT_ERR_DEVICE;
    }
}

}  // namespace lt

#define LT_EXPORT __attribute__((visibility("default")))
#define LT_AS_TRANSFER(pointer) reinterpret_cast<lt_transfer*>(pointer)
#define LT_AS_ENGINE_TRANSFER(pointer) reinterpret_cast<lt::EngineImpl::Transfer*>(pointer)

extern "C" {

LT_EXPORT int lt_abi_version(void) { return LT_ABI_VERSION; }
LT_EXPORT const char* lt_flavor(void) { return Device::flavor(); }
LT_EXPORT const char* lt_last_error(void) { return lt::last_error.c_str(); }

LT_EXPORT int lt_create(int device, uint64_t staging_bytes, uint32_t num_staging, uint32_t ids_capacity,
                        uint64_t host_cache_bytes, lt_engine** out) {
    if (out == nullptr) {
        lt::last_error = "null out pointer";
        return LT_ERR_INVALID;
    }
    try {
        *out = new lt_engine(device, staging_bytes, num_staging, ids_capacity, host_cache_bytes);
        return LT_OK;
    } catch (const std::invalid_argument& error) {
        lt::last_error = error.what();
        return LT_ERR_INVALID;
    } catch (const std::exception& error) {
        lt::last_error = error.what();
        return LT_ERR_DEVICE;
    }
}

LT_EXPORT int lt_destroy(lt_engine* engine) {
    if (engine == nullptr) return LT_OK;
    std::string problem;
    try {
        engine->impl.device().activate();
    } catch (const std::exception& error) {
        problem = error.what();
    }
    std::string shutdown = engine->impl.shutdown();
    delete engine;
    if (problem.empty()) problem = shutdown;
    if (problem.empty()) return LT_OK;
    lt::last_error = problem;
    return LT_ERR_DEVICE;
}

LT_EXPORT int lt_prepare_staging(lt_engine* engine) {
    return lt::guarded(engine, [&] { engine->impl.prepare_staging(); });
}

LT_EXPORT int lt_get_stats(lt_engine* engine, lt_stats* out) {
    if (engine == nullptr || out == nullptr) {
        lt::last_error = "null engine or out pointer";
        return LT_ERR_INVALID;
    }
    *out = engine->impl.stats();
    return LT_OK;
}

LT_EXPORT int lt_host_alloc(lt_engine* engine, uint64_t bytes, void** out) {
    return lt::guarded(engine, [&] { *out = engine->impl.host_alloc(bytes); });
}

LT_EXPORT int lt_host_free(lt_engine* engine, void* pointer) {
    return lt::guarded(engine, [&] { engine->impl.host_free(pointer); });
}

LT_EXPORT int lt_out_gather(lt_engine* engine, const void* pool, uint64_t block_bytes, const uint32_t* ids, uint32_t count,
                            void* host, lt_transfer** out) {
    return lt::guarded(engine, [&] {
        *out = LT_AS_TRANSFER(engine->impl.out_gather(static_cast<const uint8_t*>(pool), block_bytes, ids, count, static_cast<uint8_t*>(host)));
    });
}

LT_EXPORT int lt_in_scatter(lt_engine* engine, void* pool, uint64_t block_bytes, const uint32_t* ids, uint32_t count,
                            const void* host, lt_transfer** out) {
    return lt::guarded(engine, [&] {
        *out = LT_AS_TRANSFER(engine->impl.in_scatter(static_cast<uint8_t*>(pool), block_bytes, ids, count, static_cast<const uint8_t*>(host)));
    });
}

LT_EXPORT int lt_out_per_block(lt_engine* engine, const void* pool, uint64_t block_bytes, const uint32_t* ids, uint32_t count,
                               void* host, lt_transfer** out) {
    return lt::guarded(engine, [&] {
        *out = LT_AS_TRANSFER(engine->impl.out_per_block(static_cast<const uint8_t*>(pool), block_bytes, ids, count, static_cast<uint8_t*>(host)));
    });
}

LT_EXPORT int lt_in_per_block(lt_engine* engine, void* pool, uint64_t block_bytes, const uint32_t* ids, uint32_t count,
                              const void* host, lt_transfer** out) {
    return lt::guarded(engine, [&] {
        *out = LT_AS_TRANSFER(engine->impl.in_per_block(static_cast<uint8_t*>(pool), block_bytes, ids, count, static_cast<const uint8_t*>(host)));
    });
}

LT_EXPORT int lt_ext_wait(lt_engine* engine, void* stream, uint32_t slot, int which) {
    return lt::guarded(engine, [&] { engine->impl.ext_wait(stream, slot, which); });
}

LT_EXPORT int lt_ext_record(lt_engine* engine, void* stream, uint32_t slot, int which) {
    return lt::guarded(engine, [&] { engine->impl.ext_record(stream, slot, which); });
}

LT_EXPORT int lt_ext_copy_out(lt_engine* engine, const void* staging, void* host, uint64_t bytes, uint32_t slot) {
    return lt::guarded(engine, [&] { engine->impl.ext_copy_out(staging, static_cast<uint8_t*>(host), bytes, slot); });
}

LT_EXPORT int lt_ext_copy_in(lt_engine* engine, void* staging, const void* host, uint64_t bytes, uint32_t slot) {
    return lt::guarded(engine, [&] { engine->impl.ext_copy_in(staging, static_cast<const uint8_t*>(host), bytes, slot); });
}

LT_EXPORT int lt_ext_finish(lt_engine* engine, void* stream, lt_transfer** out) {
    return lt::guarded(engine, [&] { *out = LT_AS_TRANSFER(engine->impl.ext_finish(stream)); });
}

LT_EXPORT int lt_poll(lt_engine* engine, lt_transfer* transfer, int* done) {
    return lt::guarded(engine, [&] { *done = engine->impl.poll(LT_AS_ENGINE_TRANSFER(transfer)) ? 1 : 0; });
}

LT_EXPORT int lt_wait(lt_engine* engine, lt_transfer* transfer) {
    return lt::guarded(engine, [&] { engine->impl.wait_for(LT_AS_ENGINE_TRANSFER(transfer)); });
}

LT_EXPORT int lt_release(lt_engine* engine, lt_transfer* transfer) {
    return lt::guarded(engine, [&] { engine->impl.release(LT_AS_ENGINE_TRANSFER(transfer)); });
}

}  // extern "C"

#endif  // LAYOVER_TRANSFER_CORE_H

// A host-thread imitation of the CUDA stream model, for running the transfer engine and
// its tests without a GPU. It keeps the semantics the engine depends on: each stream is a
// FIFO on its own thread, cudaEventRecord replaces an event's latest record, and a wait or
// query captures the record current when it is called.
#ifndef LAYOVER_HOST_DEVICE_H
#define LAYOVER_HOST_DEVICE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "layover_transfer.h"
#include "pack.h"

namespace lt {

// One lock and condition variable for everything: this build exists for correctness, and a
// single lock makes its own bugs easy to rule out.
class HostDevice {
   public:
    struct EventImpl {
        uint64_t recorded = 0;
        uint64_t completed = 0;
    };

    struct StreamImpl {
        std::deque<std::function<void()>> queue;
        std::thread worker;
        bool stopping = false;
        bool busy = false;
    };

    using Stream = StreamImpl*;
    using Event = EventImpl*;

    static const char* flavor() { return "host"; }

    explicit HostDevice(int) {}

    void activate() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_.empty()) throw std::runtime_error(failure_);
    }

    void check_pool(const void*) {}

    void* device_alloc(uint64_t bytes) { return aligned(bytes); }
    void device_free(void* pointer) { free(pointer); }
    // Null when out of memory, which unlike any other failure the engine survives.
    void* host_alloc(uint64_t bytes) {
        if (pinned_bytes_ + bytes > pin_limit_) return nullptr;
        void* pointer = nullptr;
        if (posix_memalign(&pointer, 4096, bytes) != 0) return nullptr;
        pinned_bytes_ += bytes;
        pinned_[pointer] = bytes;
        return pointer;
    }

    void host_free(void* pointer) {
        auto entry = pinned_.find(pointer);
        if (entry != pinned_.end()) {
            pinned_bytes_ -= entry->second;
            pinned_.erase(entry);
        }
        free(pointer);
    }

    void set_pin_limit(uint64_t bytes) { pin_limit_ = bytes; }

    Stream stream_create() {
        auto* stream = new StreamImpl();
        stream->worker = std::thread([this, stream] { run(stream); });
        return stream;
    }

    void stream_sync(Stream stream) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return stream->queue.empty() && !stream->busy; });
    }

    void stream_destroy(Stream stream) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stream->stopping = true;
            changed_.notify_all();
        }
        stream->worker.join();
        delete stream;
    }

    static Stream stream_from_handle(void* handle) { return static_cast<Stream>(handle); }

    Event event_create() { return new EventImpl(); }
    void event_destroy(Event event) { delete event; }

    void event_record(Event event, Stream stream) {
        uint64_t generation;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            generation = ++event->recorded;
        }
        enqueue(stream, [this, event, generation] {
            std::lock_guard<std::mutex> lock(mutex_);
            if (generation > event->completed) event->completed = generation;
            changed_.notify_all();
        });
    }

    void stream_wait(Stream stream, Event event) {
        uint64_t target;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            target = event->recorded;
        }
        enqueue(stream, [this, event, target] {
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock, [&] { return event->completed >= target; });
        });
    }

    bool event_query(Event event) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_.empty()) throw std::runtime_error(failure_);
        return event->completed >= event->recorded;
    }

    void event_sync(Event event) {
        std::unique_lock<std::mutex> lock(mutex_);
        const uint64_t target = event->recorded;
        changed_.wait(lock, [&] { return event->completed >= target || !failure_.empty(); });
        if (!failure_.empty()) throw std::runtime_error(failure_);
    }

    void memcpy_to_host(Stream stream, void* dst, const void* src, uint64_t bytes) { copy(stream, dst, src, bytes); }
    void memcpy_to_device(Stream stream, void* dst, const void* src, uint64_t bytes) { copy(stream, dst, src, bytes); }

    void launch_gather(Stream stream, const uint8_t* pool, const uint32_t* ids, uint8_t* staging, uint64_t block_bytes, uint32_t blocks) {
        enqueue(stream, [=] {
            const uint32_t chunks = chunks_for(block_bytes);
            for (uint32_t kv = 0; kv < blocks; ++kv) {
                for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                    for (uint32_t lane = 0; lane < kLanes; ++lane) {
                        gather_lane(pool, ids, staging, block_bytes, kv, chunk, lane, kLanes);
                    }
                }
            }
        });
    }

    void launch_scatter(Stream stream, uint8_t* pool, const uint32_t* ids, const uint8_t* staging, uint64_t block_bytes, uint32_t blocks) {
        enqueue(stream, [=] {
            const uint32_t chunks = chunks_for(block_bytes);
            for (uint32_t kv = 0; kv < blocks; ++kv) {
                for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                    for (uint32_t lane = 0; lane < kLanes; ++lane) {
                        scatter_lane(pool, ids, staging, block_bytes, kv, chunk, lane, kLanes);
                    }
                }
            }
        });
    }

    void call(Stream stream, lt_host_kernel kernel, void* context) {
        enqueue(stream, [this, kernel, context] {
            if (kernel(context) != 0) {
                std::lock_guard<std::mutex> lock(mutex_);
                failure_ = "a host kernel failed";
                changed_.notify_all();
            }
        });
    }

    Stream extra_stream() {
        if (!extra_) extra_.reset(new Owned{stream_create(), this});
        return extra_->stream;
    }

    // Tests may tear down with streams held still, which must not hang.
    void prepare_teardown() { set_paused(false); }

    void set_paused(bool paused) {
        std::lock_guard<std::mutex> lock(mutex_);
        paused_ = paused;
        changed_.notify_all();
    }

    void set_copy_delay_us(uint32_t delay_us) {
        std::lock_guard<std::mutex> lock(mutex_);
        copy_delay_us_ = delay_us;
    }

    ~HostDevice() { extra_.reset(); }

   private:
    struct Owned {
        Stream stream;
        HostDevice* device;
        ~Owned() {
            device->stream_sync(stream);
            device->stream_destroy(stream);
        }
    };

    std::map<void*, uint64_t> pinned_;
    uint64_t pinned_bytes_ = 0;
    uint64_t pin_limit_ = UINT64_MAX;

    static void* aligned(uint64_t bytes) {
        void* pointer = nullptr;
        if (posix_memalign(&pointer, 4096, bytes) != 0) throw std::runtime_error("posix_memalign failed");
        return pointer;
    }

    void copy(Stream stream, void* dst, const void* src, uint64_t bytes) {
        enqueue(stream, [this, dst, src, bytes] {
            uint32_t delay;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                delay = copy_delay_us_;
            }
            if (delay != 0) std::this_thread::sleep_for(std::chrono::microseconds(delay));
            memcpy(dst, src, bytes);
        });
    }

    void enqueue(Stream stream, std::function<void()> command) {
        std::lock_guard<std::mutex> lock(mutex_);
        stream->queue.push_back(std::move(command));
        changed_.notify_all();
    }

    void run(Stream stream) {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            changed_.wait(lock, [&] { return stream->stopping || (!stream->queue.empty() && !paused_); });
            if (stream->stopping && (stream->queue.empty() || paused_)) return;
            std::function<void()> command = std::move(stream->queue.front());
            stream->queue.pop_front();
            stream->busy = true;
            lock.unlock();
            command();
            lock.lock();
            stream->busy = false;
            changed_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable changed_;
    bool paused_ = false;
    uint32_t copy_delay_us_ = 0;
    std::string failure_;
    std::unique_ptr<Owned> extra_;
};

}  // namespace lt

#endif  // LAYOVER_HOST_DEVICE_H

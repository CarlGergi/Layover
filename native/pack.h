// Block packing shared by the CUDA kernels and the host build, so the host tests run the
// exact arithmetic the GPU will.
#ifndef LAYOVER_PACK_H
#define LAYOVER_PACK_H

#include <stdint.h>

#ifdef __CUDACC__
#define LT_HD __host__ __device__
#else
#define LT_HD
#endif

namespace lt {

// One thread block copies one 64 KiB chunk of a KV block. Why these numbers (reasoned from
// the architecture; bench/transfer_gather.py is what checks them on the GPU):
// - 256 threads is a whole number of 32-lane warps, with no partial warps to waste slots.
//   Neighbouring threads copy neighbouring words, so one warp touches 256 contiguous
//   bytes, which coalesces into whole L2 sectors instead of 32 scattered requests.
// - 64 KiB per thread block makes a Qwen3-4B block (2.36 MiB) 36 thread blocks. Thread
//   blocks are spawned per SM, so a transfer of a few KV blocks still spreads over every
//   SM, and the spawn cost stays small next to 32 loop iterations per thread.
constexpr uint32_t kLanes = 256;
constexpr uint32_t kWordsPerLane = 32;
constexpr uint64_t kChunkBytes = uint64_t(kLanes) * kWordsPerLane * 8;
// A launch's grid has KV blocks on y, and CUDA caps grid y at 65535 where x may reach 2**31-1.
constexpr uint32_t kMaxBlocksPerLaunch = 65535;

LT_HD inline uint32_t chunks_for(uint64_t block_bytes) {
    return uint32_t((block_bytes + kChunkBytes - 1) / kChunkBytes);
}

// Whether 64-bit copies are legal. A misaligned 64-bit access faults the whole GPU context,
// so callers check this before anything is launched.
LT_HD inline bool words_aligned(const void* pointer, uint64_t block_bytes) {
    return (uintptr_t(pointer) & 7) == 0 && (block_bytes & 7) == 0;
}

// This thread's share of a chunk: every `lanes`-th word starting at word `lane`.
LT_HD inline void copy_lane(const uint8_t* src, uint8_t* dst, uint64_t bytes, uint32_t lane, uint32_t lanes) {
    const uint64_t* from = reinterpret_cast<const uint64_t*>(src);
    uint64_t* to = reinterpret_cast<uint64_t*>(dst);
    const uint64_t words = bytes / 8;
    for (uint64_t word = lane; word < words; word += lanes) {
        to[word] = from[word];
    }
}

LT_HD inline uint64_t chunk_length(uint64_t block_bytes, uint32_t chunk) {
    const uint64_t remaining = block_bytes - uint64_t(chunk) * kChunkBytes;
    return remaining < kChunkBytes ? remaining : kChunkBytes;
}

// kv_index counts blocks within this kernel's segment; ids[kv_index] places it in the pool.
LT_HD inline void gather_lane(const uint8_t* pool, const uint32_t* ids, uint8_t* staging, uint64_t block_bytes,
                              uint32_t kv_index, uint32_t chunk, uint32_t lane, uint32_t lanes) {
    const uint64_t offset = uint64_t(chunk) * kChunkBytes;
    copy_lane(pool + uint64_t(ids[kv_index]) * block_bytes + offset, staging + uint64_t(kv_index) * block_bytes + offset,
              chunk_length(block_bytes, chunk), lane, lanes);
}

LT_HD inline void scatter_lane(uint8_t* pool, const uint32_t* ids, const uint8_t* staging, uint64_t block_bytes,
                               uint32_t kv_index, uint32_t chunk, uint32_t lane, uint32_t lanes) {
    const uint64_t offset = uint64_t(chunk) * kChunkBytes;
    copy_lane(staging + uint64_t(kv_index) * block_bytes + offset, pool + uint64_t(ids[kv_index]) * block_bytes + offset,
              chunk_length(block_bytes, chunk), lane, lanes);
}

}  // namespace lt

#endif  // LAYOVER_PACK_H

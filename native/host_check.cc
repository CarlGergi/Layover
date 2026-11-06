// Runs the host build under the address and undefined sanitizers: scattered round trips,
// segments, ring wraparound, teardown mid-flight. Exits non-zero on any failure.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <random>
#include <vector>

#include "layover_transfer.h"

static void require(int status, const char* what) {
    if (status != LT_OK) {
        fprintf(stderr, "%s: %s\n", what, lt_last_error());
        exit(1);
    }
}

static void run(uint64_t block_bytes, uint32_t pool_blocks, uint32_t count, uint64_t staging, uint32_t ring, unsigned seed) {
    std::mt19937_64 random(seed);
    std::vector<uint64_t> pool(block_bytes * pool_blocks / 8);
    for (uint64_t& word : pool) word = random();
    const std::vector<uint64_t> original = pool;

    std::vector<uint32_t> ids(pool_blocks);
    for (uint32_t i = 0; i < pool_blocks; ++i) ids[i] = i;
    std::shuffle(ids.begin(), ids.end(), random);
    ids.resize(count);

    lt_engine* engine = nullptr;
    require(lt_create(0, staging, 2, ring, 1 << 24, &engine), "create");
    for (int round = 0; round < 4; ++round) {
        void* host = nullptr;
        require(lt_host_alloc(engine, block_bytes * count, &host), "alloc");
        lt_transfer* transfer = nullptr;
        require(lt_out_gather(engine, pool.data(), block_bytes, ids.data(), count, host, &transfer), "out");
        require(lt_wait(engine, transfer), "wait");
        require(lt_release(engine, transfer), "release");
        for (uint32_t id : ids) memset(reinterpret_cast<uint8_t*>(pool.data()) + id * block_bytes, 0, block_bytes);
        require(lt_in_scatter(engine, pool.data(), block_bytes, ids.data(), count, host, &transfer), "in");
        require(lt_wait(engine, transfer), "wait");
        require(lt_release(engine, transfer), "release");
        if (pool != original) {
            fprintf(stderr, "round trip changed the pool\n");
            exit(1);
        }
        require(lt_host_free(engine, host), "free");
    }
    // Teardown with work still queued must wait for it and free everything.
    void* host = nullptr;
    require(lt_host_alloc(engine, block_bytes * count, &host), "alloc");
    lt_transfer* transfer = nullptr;
    require(lt_out_gather(engine, pool.data(), block_bytes, ids.data(), count, host, &transfer), "out");
    require(lt_destroy(engine), "destroy");
}

int main() {
    run(4096, 16, 9, 8192, 12, 1);         // Many segments and a ring that wraps every round.
    run(6144, 8, 8, 6144, 8, 2);           // One block per segment, exactly the staging size.
    run(1 << 20, 6, 5, 3 << 20, 16, 3);    // Blocks larger than a chunk, with a partial last chunk.
    run((1 << 18) + 8, 6, 6, 1 << 21, 8, 4);
    printf("host_check: ok\n");
    return 0;
}

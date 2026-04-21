#pragma once
// Lightweight CUPTI Activity-based kernel launch tracker.
// Captures grid/block dimensions and kernel names for all GPU kernel launches
// without requiring admin privileges (no hardware counters needed).
//
// Usage:
//   KernelLaunchTracker tracker;
//   tracker.start();
//   // ... run CUDA kernels ...
//   cudaDeviceSynchronize();
//   auto launches = tracker.stop();  // returns vector of KernelLaunch records

#include <cupti_activity.h>
#include <cupti.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUPTI(call)                                                    \
    do {                                                                      \
        CUptiResult _status = (call);                                         \
        if (_status != CUPTI_SUCCESS) {                                       \
            const char *errstr;                                               \
            cuptiGetResultString(_status, &errstr);                           \
            fprintf(stderr, "CUPTI error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, errstr);                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

struct KernelLaunch {
    std::string name;
    int32_t gridX, gridY, gridZ;
    int32_t blockX, blockY, blockZ;
    int32_t staticSharedMem;
    int32_t dynamicSharedMem;
    uint16_t registersPerThread;

    int64_t totalBlocks() const {
        return (int64_t)gridX * gridY * gridZ;
    }

    int64_t threadsPerBlock() const {
        return (int64_t)blockX * blockY * blockZ;
    }
};

class KernelLaunchTracker {
public:
    KernelLaunchTracker() = default;
    ~KernelLaunchTracker() = default;

    // Non-copyable (global state)
    KernelLaunchTracker(const KernelLaunchTracker&) = delete;
    KernelLaunchTracker& operator=(const KernelLaunchTracker&) = delete;

    void start() {
        std::lock_guard<std::mutex> lock(s_mutex);
        s_launches.clear();
        CHECK_CUPTI(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
        CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }

    // Call cudaDeviceSynchronize() BEFORE calling stop() to ensure all kernels
    // have completed and their activity records are available.
    std::vector<KernelLaunch> stop() {
        CHECK_CUPTI(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
        CHECK_CUPTI(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

        std::lock_guard<std::mutex> lock(s_mutex);
        return s_launches;
    }

    // Flush and return records collected so far, without stopping.
    // Call cudaDeviceSynchronize() before this.
    std::vector<KernelLaunch> flush() {
        CHECK_CUPTI(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));

        std::lock_guard<std::mutex> lock(s_mutex);
        auto result = s_launches;
        s_launches.clear();
        return result;
    }

private:
    static constexpr size_t BUF_SIZE = 1 * 1024 * 1024; // 1 MB
    static constexpr size_t ALIGN = 8;

    static std::vector<KernelLaunch> s_launches;
    static std::mutex s_mutex;

    static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                                          size_t *maxNumRecords) {
        auto *buf = (uint8_t *)aligned_alloc(ALIGN, BUF_SIZE);
        if (!buf) {
            fprintf(stderr, "KernelLaunchTracker: aligned_alloc failed\n");
            exit(EXIT_FAILURE);
        }
        *buffer = buf;
        *size = BUF_SIZE;
        *maxNumRecords = 0; // unlimited
    }

    static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
                                          uint8_t *buffer, size_t size,
                                          size_t validSize) {
        CUpti_Activity *record = nullptr;
        while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
            if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
                record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                auto *kernel = reinterpret_cast<CUpti_ActivityKernel9 *>(record);
                KernelLaunch kl;
                kl.name = kernel->name ? kernel->name : "<unknown>";
                kl.gridX = kernel->gridX;
                kl.gridY = kernel->gridY;
                kl.gridZ = kernel->gridZ;
                kl.blockX = kernel->blockX;
                kl.blockY = kernel->blockY;
                kl.blockZ = kernel->blockZ;
                kl.staticSharedMem = kernel->staticSharedMemory;
                kl.dynamicSharedMem = kernel->dynamicSharedMemory;
                kl.registersPerThread = kernel->registersPerThread;

                std::lock_guard<std::mutex> lock(s_mutex);
                s_launches.push_back(std::move(kl));
            }
        }
        free(buffer);
    }
};

// Static member definitions — include this header in exactly one .cu file,
// or treat it as header-only with inline statics (C++17).
inline std::vector<KernelLaunch> KernelLaunchTracker::s_launches;
inline std::mutex KernelLaunchTracker::s_mutex;

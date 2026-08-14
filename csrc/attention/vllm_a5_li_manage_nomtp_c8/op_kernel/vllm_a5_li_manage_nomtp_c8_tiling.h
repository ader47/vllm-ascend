#ifndef VLLM_A5_LI_MANAGE_NOMTP_C8_TILING_H
#define VLLM_A5_LI_MANAGE_NOMTP_C8_TILING_H

#include <cstdint>

// Static geometry only.  All request-dependent values remain device tensors so
// graph replay may refresh them without re-tiling.
struct VllmA5LiManageNomtpC8TilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t totalQueryRows;
    uint32_t poolSize;
    uint32_t tokenCapacity;
    uint32_t outputCapacity;
    uint32_t indexHeads;
    uint32_t maxBlockNumPerBatch;
    uint32_t maxCandidateLen;
    uint32_t weightStride;
    uint32_t keyStride;
    uint32_t scaleStride;
    uint32_t scoreWorkspaceStride;
};

#endif

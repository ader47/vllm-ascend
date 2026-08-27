#ifndef VLLM_A5_LI_MANAGE_C8_TILING_H
#define VLLM_A5_LI_MANAGE_C8_TILING_H

#include <cstdint>

struct VllmA5LiManageC8TilingData {
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
    uint32_t fastScoreWorkspaceStride;
    uint32_t fastQueryTileSize;
    uint32_t fastPathEnabled;
};

#endif

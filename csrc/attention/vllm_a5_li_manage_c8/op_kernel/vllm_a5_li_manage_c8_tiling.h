#ifndef VLLM_A5_LI_MANAGE_C8_TILING_H
#define VLLM_A5_LI_MANAGE_C8_TILING_H

#include <cstdint>

struct VllmA5LiManageC8TilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t poolSize;
    uint32_t tokenCapacity;
    uint32_t outputCapacity;
};

#endif

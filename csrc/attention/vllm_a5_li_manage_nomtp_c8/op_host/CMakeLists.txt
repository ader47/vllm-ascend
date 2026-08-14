# Copyright (c) 2026 Huawei Technologies Co., Ltd.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        vllm_a5_li_manage_nomtp_c8_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME VllmA5LiManageNomtpC8
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE vllm_a5_li_manage_nomtp_c8
        ACLNNTYPE aclnn
    )
endif()

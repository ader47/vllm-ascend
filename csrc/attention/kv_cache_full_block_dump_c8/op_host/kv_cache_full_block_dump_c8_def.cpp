/** Host definition for the Ascend 950 packed-C8 full-block dump. */

#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class KvCacheFullBlockDumpC8 : public OpDef {
public:
    explicit KvCacheFullBlockDumpC8(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> bytes = {ge::DT_INT8};
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        this->Input("src_cache").ParamType(REQUIRED)
            .DataType(bytes).Format(formats);
        this->Input("dst_cache").ParamType(REQUIRED)
            .DataType(bytes).Format(formats);
        this->Input("src_block_ids").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("dst_block_ids").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        // The output shares the mutable input name so aclnn models an
        // in-place update. The kernel writes the input GM pointer directly.
        this->Output("dst_cache").ParamType(REQUIRED)
            .DataType(bytes).Format(formats);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(KvCacheFullBlockDumpC8);
}  // namespace ops

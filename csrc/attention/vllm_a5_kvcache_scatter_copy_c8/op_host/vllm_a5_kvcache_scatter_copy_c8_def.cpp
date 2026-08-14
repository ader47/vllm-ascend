/** Host definition for the Ascend 950 GLM-5.x packed-C8 scatter copy. */

#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class VllmA5KvcacheScatterCopyC8 : public OpDef {
public:
    explicit VllmA5KvcacheScatterCopyC8(const char *name)
        : OpDef(name)
    {
        const std::vector<ge::DataType> bytes = {ge::DT_INT8};
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};

        this->Input("hbm_kv")
            .ParamType(REQUIRED).DataType(bytes).Format(formats);
        this->Input("dram_kv")
            .ParamType(REQUIRED).DataType(bytes).Format(formats);
        this->Input("hbm_block_table")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("dram_block_table")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("copy_src_ids")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("copy_dst_slots")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("copy_counts")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        // The matching input/output name declares an in-place reference. The
        // generated ACLNN host API consequently receives hbm_kv only once;
        // the device kernel still gets the framework-generated output alias.
        this->Output("hbm_kv")
            .ParamType(REQUIRED).DataType(bytes).Format(formats);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(VllmA5KvcacheScatterCopyC8);
} // namespace ops

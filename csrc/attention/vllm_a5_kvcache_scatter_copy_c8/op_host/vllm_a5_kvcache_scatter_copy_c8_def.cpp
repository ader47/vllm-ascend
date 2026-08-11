/** Host definition for the Ascend 950 DSA packed-C8 scatter copy. */

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
        this->Input("source_token_ids")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("destination_slots")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("copy_counts")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_tokens")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("candidate_lens")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("actual_seq_lengths_kv")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        // Same-name output declares an in-place update of the packed HBM
        // cache, matching the established A3 KSC ACLNN contract.
        this->Output("hbm_kv")
            .ParamType(REQUIRED).DataType(bytes).Format(formats);
        this->Output("attention_slots")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("resident_seq_lengths")
            .ParamType(REQUIRED).DataType(ints).Format(formats);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(VllmA5KvcacheScatterCopyC8);
}  // namespace ops

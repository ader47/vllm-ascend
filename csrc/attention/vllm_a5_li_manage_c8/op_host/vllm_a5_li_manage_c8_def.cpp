/**
 * Host definition for request-pool management after the native A5 C8
 * QuantLightningIndexer.
 */

#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class VllmA5LiManageC8 : public OpDef {
public:
    explicit VllmA5LiManageC8(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        this->Input("topk_indices")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("req_pool_entries")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_slots_pool")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("row_modes")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("actual_seq_lengths_key")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("source_ids")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("destination_slots")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("miss_counts")
            .ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("tail_info")
            .ParamType(REQUIRED).DataType(ints).Format(formats);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(VllmA5LiManageC8);
}  // namespace ops

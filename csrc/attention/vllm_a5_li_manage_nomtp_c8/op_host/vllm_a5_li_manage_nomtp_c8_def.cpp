/** Host definition for the one-kernel A5 non-MTP C8 LI + manage op. */

#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class VllmA5LiManageNomtpC8 : public OpDef {
public:
    explicit VllmA5LiManageNomtpC8(const char *name) : OpDef(name)
    {
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        this->Input("index_weights").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format(formats)
            .IgnoreContiguous();
        this->Input("query").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN}).Format(formats);
        this->Input("query_dequant_scale").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format(formats);
        this->Input("actual_seq_lengths_query").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("index_key_cache").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN}).Format(formats)
            .IgnoreContiguous();
        this->Input("index_key_dequant_scale").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format(formats)
            .IgnoreContiguous();
        this->Input("index_block_table").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("candidate_lens").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("final_seq_lengths_kv").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("row_modes").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("req_pool_entries").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("cache_slots_pool").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Attr("key_stride").Int();
        this->Attr("scale_stride").Int();
        this->Attr("weight_stride").Int();
        this->Output("sparse_and_tail_slots").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Output("resident_seq_lengths").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Output("copy_src_ids").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Output("copy_dst_slots").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Output("copy_counts").ParamType(REQUIRED)
            .DataType(ints).Format(formats);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(VllmA5LiManageNomtpC8);
} // namespace ops

// =============================================================================
// reduction_kernel.cpp  (v2 — clean, no static locals)
// =============================================================================

#include <adf.h>
#include "../kernels.h"

void reduction_kernel(
    input_stream<int32>*  in_a,
    input_stream<int32>*  in_b,
    output_stream<int32>* out
)
{
    int32_t cost_a = readincr(in_a);
    int32_t id_a   = readincr(in_a);
    int32_t cost_b = readincr(in_b);
    int32_t id_b   = readincr(in_b);

    if (cost_a <= cost_b) {
        writeincr(out, cost_a);
        writeincr(out, id_a);
    } else {
        writeincr(out, cost_b);
        writeincr(out, id_b);
    }
}
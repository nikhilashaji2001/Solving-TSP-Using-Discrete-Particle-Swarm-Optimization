// =============================================================================
// broadcast_relay.cpp  (v2 — clean stack-only locals, no static)
// =============================================================================

#include <adf.h>
#include "../kernels.h"
#include "../dpso_config.h"

void broadcast_relay(
    input_stream<int32>*  in,
    output_stream<int32>* out_a,
    output_stream<int32>* out_b
)
{
    // Stack buffer — allocated fresh each call, thread safe
    int32_t position[N_CITIES];

    for (int i = 0; i < N_CITIES; i++)
        position[i] = readincr(in);

    for (int i = 0; i < N_CITIES; i++)
        writeincr(out_a, position[i]);

    for (int i = 0; i < N_CITIES; i++)
        writeincr(out_b, position[i]);
}
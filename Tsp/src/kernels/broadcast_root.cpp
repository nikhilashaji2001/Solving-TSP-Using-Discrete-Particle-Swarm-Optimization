// =============================================================================
// broadcast_root.cpp  (v4 — deadlock fix for iteration 0)
//
// On iteration 0:
//   - Particles did NOT read from in_gbest (they bootstrapped)
//   - Reduction tree DID run (particles sent {cost,pid} after writing buffer)
//   - bc_root reads winner_id from r_final normally
//   - bc_root reads winner's position from buffer normally
//   - bc_root writes 50 words to out_left and out_right
//   - These words are consumed by particles on iteration 1
//
// The fix in particle_kernel (skip gBest read on iter 0) breaks the circle:
//   iter 0: particle writes buffer → reduction runs → bc_root reads buffer
//           → bc_root writes to broadcast streams
//   iter 1: particle reads from broadcast streams (now data is available)
//
// bc_root itself does not need to change for the iter-0 case — that fix is
// entirely in particle_kernel skipping the stream read on iter 0.
//
// SEPARATE BUG FIXED HERE (unrelated to iter 0): winner_id and winner_cost
// were being read in the wrong order relative to how reduction_kernel.cpp
// writes them (cost, then id). This made safe_id fail its 0..7 range check
// on effectively every call (a TSP tour cost is never that small) and fall
// back to hardcoded particle 0, every iteration, regardless of who actually
// won. Read order below now matches the write order.
// =============================================================================

#include <adf.h>
#include "../kernels.h"
#include "../dpso_config.h"

using namespace adf;

void broadcast_root(
    input_stream<int32>*    in_winner,
    input_buffer<int32>&    in_pos0,
    input_buffer<int32>&    in_pos1,
    input_buffer<int32>&    in_pos2,
    input_buffer<int32>&    in_pos3,
    input_buffer<int32>&    in_pos4,
    input_buffer<int32>&    in_pos5,
    input_buffer<int32>&    in_pos6,
    input_buffer<int32>&    in_pos7,
    output_stream<int32>*   out_left,
    output_stream<int32>*   out_right
)
{
    // Read winner from reduction tree.
    // reduction_kernel.cpp writes COST first, then ID (writeincr(out, cost); writeincr(out, id);)
    // — read order here must match, or winner_id silently receives a cost value instead.
    int32_t winner_cost = readincr(in_winner);
    int32_t winner_id   = readincr(in_winner);
    (void)winner_cost;

    // Select winning particle's position buffer
    int32_t* pos_ptrs[8] = {
        (int32_t*)in_pos0.data(), (int32_t*)in_pos1.data(),
        (int32_t*)in_pos2.data(), (int32_t*)in_pos3.data(),
        (int32_t*)in_pos4.data(), (int32_t*)in_pos5.data(),
        (int32_t*)in_pos6.data(), (int32_t*)in_pos7.data()
    };
    int32_t safe_id = (winner_id >= 0 && winner_id < 8) ? winner_id : 0;
    int32_t* src = pos_ptrs[safe_id];

    // Copy to local buffer
    int32_t gbest_position[N_CITIES];
    for (int i = 0; i < N_CITIES; i++) gbest_position[i] = src[i];

    // Fan out to both outputs (2 output limit respected)
    for (int i = 0; i < N_CITIES; i++) writeincr(out_left,  gbest_position[i]);
    for (int i = 0; i < N_CITIES; i++) writeincr(out_right, gbest_position[i]);
}
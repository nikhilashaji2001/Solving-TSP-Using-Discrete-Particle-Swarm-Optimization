#pragma once
#include <adf.h>
#include "dpso_config.h"

using namespace adf;

// Distance matrix is delivered as an RTP (Runtime Parameter). Confirmed
// against official Xilinx AI Engine RTP array documentation: array RTP
// requires (1) a CONST array reference on the kernel side — a non-const
// reference or input_buffer<T>& are both the wrong port kind and fail to
// compile — and (2) an async() wrapper on the graph-side connect<parameter>
// call. Written once via graph.update() before the first run() call and
// persists in the kernel's local memory across all MAX_ITER invocations.
// Stored as uint8 (verified max value in dataset = 120, fits in 0-255).
void particle_kernel(
    input_stream<int32>*    in_trigger,
    input_stream<int32>*    in_gbest,
    const uint8 (&in_dist_matrix)[N_CITIES * N_CITIES],
    output_stream<int32>*   out_pbest,
    output_buffer<int32>&   out_position
);

void reduction_kernel(
    input_stream<int32>*    in_a,
    input_stream<int32>*    in_b,
    output_stream<int32>*   out
);

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
);

void broadcast_relay(
    input_stream<int32>*    in,
    output_stream<int32>*   out_a,
    output_stream<int32>*   out_b
);
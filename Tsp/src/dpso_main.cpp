
// dpso_main.cpp  (v3 — RTP distance matrix load added)
//
// For x86sim the guard must be __X86SIM__ (not __AIESIM__).
// The graph instance must be at global scope.
// main() must call init() → update RTPs → run(N) → end() in that order.
//
// RTP data must be written after init() (tiles/memory are allocated then)
// and before the first run() call, since particle_kernel reads it on its
// very first invocation. [likely — exact sync point can vary slightly by
// Vitis version; if the build/sim complains about RTP not being ready,
// check whether your toolchain requires graph.wait() after update() before
// run() for this AIE/Vitis release.]
// =============================================================================
 
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include "dpso_graph.h"
 
// Global graph instance — must be at file scope, not inside main()
DPSOGraph dpso_graph;
 
// Load the single (non-repeated) distance matrix file and write it into
// every particle's RTP port. Same content for all particles — they're
// solving the same TSP instance — but each RTP port lives in a different
// tile's local memory, so each still needs its own update() call.
//
// Tries a few relative-path candidates because plain std::ifstream resolves
// against the process's real CWD, which does NOT necessarily match wherever
// PLIO ports resolve their paths (those honor the simulator's own input-dir
// argument, e.g. --i=../../, which this loader has no visibility into).
// On failure this prints exactly what it tried instead of failing silently.
static bool load_and_broadcast_dist_matrix() {
    const char* candidates[] = {
        "data/dist_matrix_base.txt",
        "../../data/dist_matrix_base.txt",
        "./data/dist_matrix_base.txt",
        "../data/dist_matrix_base.txt"
    };
 
    for (const char* path : candidates) {
        std::ifstream f(path);
        if (!f) {
            fprintf(stderr, "[dist_matrix_load] tried '%s': could not open\n", path);
            continue;
        }
 
        std::vector<uint8_t> matrix;
        matrix.reserve(N_CITIES * N_CITIES);
        int value;
        while (f >> value) {
            matrix.push_back((uint8_t)value);
        }
 
        if ((int)matrix.size() != N_CITIES * N_CITIES) {
            fprintf(stderr,
                "[dist_matrix_load] opened '%s' but got %d values, expected %d — wrong file or truncated read\n",
                path, (int)matrix.size(), N_CITIES * N_CITIES);
            continue;
        }
 
        fprintf(stderr, "[dist_matrix_load] loaded %d values from '%s'\n", (int)matrix.size(), path);
        for (int i = 0; i < N_PARTICLES; i++) {
            dpso_graph.update(dpso_graph.dist_matrix_rtp[i], matrix.data(), matrix.size());
        }
        return true;
    }
 
    fprintf(stderr, "[dist_matrix_load] FAILED — none of the candidate paths worked. "
                     "Check the actual CWD the simulator launches from and adjust the path list.\n");
    return false;
}
 
#if defined(__X86SIM__) || defined(__AIESIM__)
 
int main(int argc, char* argv[]) {
 
    dpso_graph.init();
 
    if (!load_and_broadcast_dist_matrix()) {
        // Fail loudly rather than silently running PSO on garbage/zeroed data.
        return 1;
    }
 
    // Run MAX_ITER graph iterations
    // Each run(1) call fires all kernels once = one PSO iteration
    dpso_graph.run(MAX_ITER);
 
    dpso_graph.end();
 
    return 0;
}
 
#endif
 

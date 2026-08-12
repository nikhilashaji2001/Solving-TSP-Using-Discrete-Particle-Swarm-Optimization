#include <fstream>
#include <vector>
#include <cstdint>
#include "dpso_config.h"
#include "dpso_graph.h"

using namespace adf;

DPSOGraph g;

// Same RTP loader as dpso_main.cpp — kept in sync manually since these are
// two separate entry points (x86sim vs aiesim/hardware) that both need the
// distance matrix written before run().
static bool load_and_broadcast_dist_matrix(const char* path) {
    std::ifstream f(path);
    if (!f) return false;

    std::vector<uint8_t> matrix;
    matrix.reserve(N_CITIES * N_CITIES);
    int value;
    while (f >> value) {
        matrix.push_back((uint8_t)value);
    }
    if ((int)matrix.size() != N_CITIES * N_CITIES) return false;

    for (int i = 0; i < N_PARTICLES; i++) {
        g.update(g.dist_matrix_rtp[i], matrix.data(), matrix.size());
    }
    return true;
}

int main()
{
    g.init();
    if (!load_and_broadcast_dist_matrix("data/dist_matrix_base.txt")) {
        return 1;
    }
    g.run(MAX_ITER);
    g.end();
    return 0;
}
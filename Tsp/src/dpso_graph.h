#pragma once
#include <adf.h>
#include "dpso_config.h"
#include "kernels.h"

using namespace adf;

#define POSITION_ELEMS  (N_CITIES)

// Calculate tree depth at compile time
constexpr int calc_tree_depth(int n) {
    int depth = 0;
    while (n > 1) {
        n = (n + 1) / 2;
        depth++;
    }
    return depth;
}

constexpr int TREE_DEPTH = calc_tree_depth(N_PARTICLES);

// Calculate max nodes at any level
constexpr int calc_max_nodes() {
    int max_nodes = N_PARTICLES;
    int temp = N_PARTICLES;
    while (temp > 1) {
        temp = (temp + 1) / 2;
        if (temp > max_nodes) max_nodes = temp;
    }
    return max_nodes;
}

constexpr int MAX_NODES = calc_max_nodes();

class DPSOGraph : public graph {
public:
    // Fixed-size arrays (AIE compiler needs compile-time sizes)
    kernel particles[N_PARTICLES];
    kernel reduction_tree[TREE_DEPTH][MAX_NODES];
    kernel broadcast_tree[TREE_DEPTH][MAX_NODES];
    
    // Track actual nodes per level
    int reduction_nodes[TREE_DEPTH];
    int broadcast_nodes[TREE_DEPTH];
    
    input_plio trigger_in[N_PARTICLES];
    output_plio gbest_out;

    // RTP: one port per particle (each maps to a different tile's local
    // memory, so each needs its own write even though the content — one
    // fixed TSP instance's distance matrix — is identical for all of them).
    // port<direction::in> matches official Xilinx array-RTP documentation.
    port<direction::in> dist_matrix_rtp[N_PARTICLES];

    DPSOGraph() {
        // =====================================================================
        // 1. CREATE PARTICLE KERNELS
        // =====================================================================
        for (int i = 0; i < N_PARTICLES; i++) {
            particles[i] = kernel::create(particle_kernel);
            
            trigger_in[i] = input_plio::create(
                "TriggerIn" + std::to_string(i), 
                plio_32_bits,
                "data/trigger_" + std::to_string(i) + ".txt");
        }
        
        // =====================================================================
        // 2. CREATE REDUCTION TREE
        // =====================================================================
        // Level 0: pairs of particles
        reduction_nodes[0] = (N_PARTICLES + 1) / 2;
        for (int i = 0; i < reduction_nodes[0]; i++) {
            reduction_tree[0][i] = kernel::create(reduction_kernel);
        }
        
        // Higher levels
        for (int level = 1; level < TREE_DEPTH; level++) {
            reduction_nodes[level] = (reduction_nodes[level - 1] + 1) / 2;
            for (int i = 0; i < reduction_nodes[level]; i++) {
                reduction_tree[level][i] = kernel::create(reduction_kernel);
            }
        }
        
        // =====================================================================
        // 3. CREATE BROADCAST TREE
        // =====================================================================
        // Level 0: root
        broadcast_nodes[0] = 1;
        broadcast_tree[0][0] = kernel::create(broadcast_root);
        
        // Level 1+: relay nodes
        for (int level = 1; level < TREE_DEPTH; level++) {
            broadcast_nodes[level] = 1 << level;  // 2^level
            for (int i = 0; i < broadcast_nodes[level]; i++) {
                broadcast_tree[level][i] = kernel::create(broadcast_relay);
            }
        }
        
        gbest_out = output_plio::create("GBestOut", plio_32_bits, "gbest_out.txt");
        
        // =====================================================================
        // 4. CONNECT PARTICLE INPUTS
        // =====================================================================
        for (int i = 0; i < N_PARTICLES; i++) {
            connect<stream>(trigger_in[i].out[0], particles[i].in[0]);
            connect<parameter>(dist_matrix_rtp[i], async(particles[i].in[2]));
            adf::dimensions(particles[i].out[1]) = {POSITION_ELEMS};
        }
        
        // =====================================================================
        // 5. CONNECT REDUCTION TREE
        // =====================================================================
        // Level 0: particles to reduction leaves
        for (int i = 0; i < reduction_nodes[0]; i++) {
            int left = 2 * i;
            int right = 2 * i + 1;
            
            if (left < N_PARTICLES) {
                connect<stream>(particles[left].out[0], reduction_tree[0][i].in[0]);
            }
            if (right < N_PARTICLES) {
                connect<stream>(particles[right].out[0], reduction_tree[0][i].in[1]);
            } else {
                connect<stream>(particles[left].out[0], reduction_tree[0][i].in[1]);
            }
        }
        
        // Higher levels
        for (int level = 1; level < TREE_DEPTH; level++) {
            for (int i = 0; i < reduction_nodes[level]; i++) {
                int left = 2 * i;
                int right = 2 * i + 1;
                
                if (left < reduction_nodes[level - 1]) {
                    connect<stream>(reduction_tree[level - 1][left].out[0],
                                  reduction_tree[level][i].in[0]);
                }
                if (right < reduction_nodes[level - 1]) {
                    connect<stream>(reduction_tree[level - 1][right].out[0],
                                  reduction_tree[level][i].in[1]);
                } else {
                    connect<stream>(reduction_tree[level - 1][left].out[0],
                                  reduction_tree[level][i].in[1]);
                }
            }
        }
        
        // =====================================================================
        // 6. CONNECT BROADCAST TREE
        // =====================================================================
        // Connect reduction root to broadcast root
        connect<stream>(reduction_tree[TREE_DEPTH - 1][0].out[0], 
                       broadcast_tree[0][0].in[0]);
        
        // Connect particle positions to broadcast root
        for (int i = 0; i < N_PARTICLES; i++) {
            connect(particles[i].out[1], broadcast_tree[0][0].in[i + 1]);
            adf::dimensions(broadcast_tree[0][0].in[i + 1]) = {POSITION_ELEMS};
        }
        
        // Connect broadcast levels
        if (TREE_DEPTH > 1) {
            connect<stream>(broadcast_tree[0][0].out[0], broadcast_tree[1][0].in[0]);
            if (broadcast_nodes[1] > 1) {
                connect<stream>(broadcast_tree[0][0].out[1], broadcast_tree[1][1].in[0]);
            }
            // Tap gbest output
            connect<stream>(broadcast_tree[1][0].out[1], gbest_out.in[0]);
        }
        
        // Connect remaining levels
        for (int level = 1; level < TREE_DEPTH - 1; level++) {
            for (int i = 0; i < broadcast_nodes[level]; i++) {
                int left = 2 * i;
                int right = 2 * i + 1;
                
                if (left < broadcast_nodes[level + 1]) {
                    if (level == 1 && i == 0) {
                        connect<stream>(broadcast_tree[level][i].out[0],
                                      broadcast_tree[level + 1][left].in[0]);
                    } else {
                        connect<stream>(broadcast_tree[level][i].out[0],
                                      broadcast_tree[level + 1][left].in[0]);
                    }
                }
                if (right < broadcast_nodes[level + 1]) {
                    connect<stream>(broadcast_tree[level][i].out[1],
                                  broadcast_tree[level + 1][right].in[0]);
                }
            }
        }
        
        // Connect leaves to particles
        int leaf_level = TREE_DEPTH - 1;
        for (int i = 0; i < broadcast_nodes[leaf_level] && i * 2 < N_PARTICLES; i++) {
            int left = 2 * i;
            int right = 2 * i + 1;
            
            if (left < N_PARTICLES) {
                connect<stream>(broadcast_tree[leaf_level][i].out[0], particles[left].in[1]);
            }
            if (right < N_PARTICLES) {
                connect<stream>(broadcast_tree[leaf_level][i].out[1], particles[right].in[1]);
            }
        }
        
        // =====================================================================
        // 7. SET KERNEL SOURCES
        // =====================================================================
        for (int i = 0; i < N_PARTICLES; i++) {
            source(particles[i]) = "kernels/particle_kernel.cpp";
            runtime<ratio>(particles[i]) = 1.0;
        }
        
        for (int level = 0; level < TREE_DEPTH; level++) {
            for (int i = 0; i < reduction_nodes[level]; i++) {
                source(reduction_tree[level][i]) = "kernels/reduction_kernel.cpp";
                runtime<ratio>(reduction_tree[level][i]) = 1.0;
            }
        }
        
        source(broadcast_tree[0][0]) = "kernels/broadcast_root.cpp";
        runtime<ratio>(broadcast_tree[0][0]) = 1.0;
        
        for (int level = 1; level < TREE_DEPTH; level++) {
            for (int i = 0; i < broadcast_nodes[level]; i++) {
                source(broadcast_tree[level][i]) = "kernels/broadcast_relay.cpp";
                runtime<ratio>(broadcast_tree[level][i]) = 1.0;
            }
        }
        
        // =====================================================================
        // 8. SET TILE LOCATIONS
        // =====================================================================
        // Particles in row 0
        for (int i = 0; i < N_PARTICLES; i++) {
            location<kernel>(particles[i]) = tile(i, 0);
        }
        
        // Reduction tree
        int row = 1;
        for (int level = 0; level < TREE_DEPTH; level++) {
            for (int i = 0; i < reduction_nodes[level]; i++) {
                int spacing = (reduction_nodes[level] > 0) ? (N_PARTICLES / reduction_nodes[level]) : 1;
                if (spacing == 0) spacing = 1;
                int col = i * spacing + spacing / 2;
                if (col >= N_PARTICLES) col = N_PARTICLES - 1;
                location<kernel>(reduction_tree[level][i]) = tile(col, row);
            }
            row++;
        }
        
        // Broadcast tree
        for (int level = 0; level < TREE_DEPTH; level++) {
            for (int i = 0; i < broadcast_nodes[level]; i++) {
                int spacing = (broadcast_nodes[level] > 0) ? (N_PARTICLES / broadcast_nodes[level]) : 1;
                if (spacing == 0) spacing = 1;
                int col = i * spacing + spacing / 2;
                if (col >= N_PARTICLES) col = N_PARTICLES - 1;
                location<kernel>(broadcast_tree[level][i]) = tile(col, row);
            }
            row++;
        }
    }
};
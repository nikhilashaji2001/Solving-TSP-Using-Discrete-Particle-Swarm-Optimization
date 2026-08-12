// =============================================================================
// dpso_config.h
// Compile-time constants for the DPSO AIE graph.
// 
// **CHANGE N_PARTICLES TO AUTOMATICALLY GENERATE GRAPH TOPOLOGY**
// The reduction and broadcast trees are generated automatically based on this value.
// =============================================================================

#pragma once

// =============================================================================
// *** MAIN CONFIGURATION PARAMETER ***
// =============================================================================
// Change this value to use different numbers of particles
// Valid values: any positive integer (2, 3, 4, 5, 8, 16, 32, ...)
// The reduction and broadcast trees will be generated automatically
//
// Examples:
//   N_PARTICLES = 4  → 3 reduction kernels, 3 broadcast kernels  (7 total aux kernels)
//   N_PARTICLES = 8  → 7 reduction kernels, 7 broadcast kernels  (14 total aux kernels)
//   N_PARTICLES = 16 → 15 reduction kernels, 15 broadcast kernels (30 total aux kernels)

#define N_PARTICLES      8       // ← **CHANGE THIS VALUE**

// =============================================================================
// Problem parameters
// =============================================================================
#define N_CITIES        50          // number of TSP cities

// =============================================================================
// PSO hyperparameters (fixed-point scaled by 256 for integer arithmetic)
// w=0.7 → W_SCALED=179,  c1=1.5 → C1_SCALED=384,  c2=1.5 → C2_SCALED=384
// =============================================================================
#define W_SCALED        179         // inertia:    0.7  × 256
#define C1_SCALED       192         // cognitive:  0.75 × 256  (c1/2 normalised)
#define C2_SCALED       192         // social:     0.75 × 256

#define MAX_VELOCITY    25          // maximum swaps in one velocity (N/2)
#define MAX_ITER        500         // iterations the host will run

// =============================================================================
// Memory sizes (bytes) — must match window<> declarations in graph
// =============================================================================
// Tour position: N_CITIES × sizeof(int32) — padded to multiple of 32 bytes
#define POSITION_BYTES  (((N_CITIES * 4) + 31) & ~31)

// Velocity: MAX_VELOCITY × 2 indices × sizeof(int32)
#define VELOCITY_BYTES  (MAX_VELOCITY * 2 * 4)

// Distance matrix: N_CITIES × N_CITIES × sizeof(int32)
#define DIST_MATRIX_BYTES (N_CITIES * N_CITIES * 4)

// pBest / gBest payload: position + cost = (N_CITIES + 1) × sizeof(int32)
#define PBEST_PAYLOAD_WORDS  (N_CITIES + 1)
#define PBEST_PAYLOAD_BYTES  (PBEST_PAYLOAD_WORDS * 4)

// =============================================================================
// Graph topology — derived automatically, do not edit
// =============================================================================
//
// The graph structure is now automatically generated based on N_PARTICLES:
//
// Reduction Tree (binary tree, bottom-up):
//   - Combines particle costs to find global best
//   - Total reduction kernels = N_PARTICLES - 1
//   - Depth = ceil(log2(N_PARTICLES))
//
// Broadcast Tree (binary tree, top-down):
//   - Distributes global best position to all particles
//   - Total broadcast kernels = N_PARTICLES - 1
//   - Depth = ceil(log2(N_PARTICLES))
//
// Example for N_PARTICLES = 8:
//   - Reduction: 4 kernels (level 0) + 2 (level 1) + 1 (level 2) = 7 kernels
//   - Broadcast: 1 kernel (level 0) + 2 (level 1) + 4 (level 2) = 7 kernels
//   - Total: 8 particles + 7 reduction + 7 broadcast = 22 kernels
//
// Example for N_PARTICLES = 4:
//   - Reduction: 2 kernels (level 0) + 1 (level 1) = 3 kernels
//   - Broadcast: 1 kernel (level 0) + 2 (level 1) = 3 kernels
//   - Total: 4 particles + 3 reduction + 3 broadcast = 10 kernels
//
// Example for N_PARTICLES = 16:
//   - Reduction: 8 (level 0) + 4 (level 1) + 2 (level 2) + 1 (level 3) = 15 kernels
//   - Broadcast: 1 (level 0) + 2 (level 1) + 4 (level 2) + 8 (level 3) = 15 kernels
//   - Total: 16 particles + 15 reduction + 15 broadcast = 46 kernels

#define N_REDUCTION_KERNELS  (N_PARTICLES - 1)
#define N_BROADCAST_KERNELS  (N_PARTICLES - 1)
#define N_TOTAL_KERNELS      (N_PARTICLES + N_REDUCTION_KERNELS + N_BROADCAST_KERNELS)
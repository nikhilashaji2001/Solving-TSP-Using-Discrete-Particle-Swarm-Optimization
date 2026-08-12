# Solving TSP Using Discrete Particle Swarm Optimization on AMD Versal VCK190

Solving the Traveling Salesman Problem (TSP) with Discrete Particle Swarm Optimization (DPSO), implemented as a dataflow graph running on the AI Engine (AIE) array of an AMD Versal ACAP (target: VCK190).

Instead of running the swarm as a loop on a CPU, each particle is a separate kernel mapped to its own physical AIE tile. All particles evaluate and update their tours **in parallel, every iteration**, and exchange information over on-chip streaming interconnect rather than shared memory.

---

## 1. The problem

- **Instance size:** 50 cities (`N_CITIES = 50`)
- **Input:** a 50×50 integer distance matrix (`data/dist_matrix_base.txt`), one value per line, row-major, symmetric-ish, values 0–120 (fits in a `uint8`)
- **Goal:** find a visiting order (a permutation of all 50 cities, returning to the start) that minimizes total tour length

---

## 2. What DPSO means here

Standard PSO moves particles through continuous space using position + velocity vectors. TSP tours aren't continuous, so this implementation uses the standard **discrete PSO** trick for permutation problems:

| Concept | Continuous PSO | This implementation |
|---|---|---|
| Position | real-valued vector | a tour = permutation of 50 city indices |
| Velocity | real-valued vector | a **swap sequence**: an ordered list of `(i, j)` index-swap pairs |
| Applying velocity | position += velocity | apply each swap in the sequence to the tour, in order |
| "Difference" between two positions | subtraction | the swap sequence that transforms one tour into the other |

Each iteration, every particle:
1. Computes the swap sequence from its current tour → its own personal best (**cognitive** component)
2. Computes the swap sequence from its current tour → the swarm's global best (**social** component)
3. Builds a new velocity by probabilistically keeping swaps from: its old velocity (**inertia**, `w`), the cognitive sequence (`c1`), and the social sequence (`c2`) — each swap is kept with a probability equal to that term's weight
4. Applies the new velocity (the kept swaps) to its tour
5. Re-evaluates tour cost; updates its personal best if improved

This is a **swap-sequence / swap-probability** DPSO variant, not the alternative "velocity as a probability matrix" formulation — worth knowing if you're comparing against other DPSO-for-TSP papers.

### Hyperparameters (fixed-point, scaled ×256 for integer arithmetic on the AIE)

| Symbol | Meaning | Value | Scaled value |
|---|---|---|---|
| `w`  | inertia weight  | 0.70 | 179 |
| `c1` | cognitive weight | 0.75 | 192 |
| `c2` | social weight    | 0.75 | 192 |
| `MAX_VELOCITY` | max swaps kept per velocity | 25 (N/2) | — |
| `MAX_ITER` | iterations per run | 500 | — |
| `N_PARTICLES` | swarm size | 8 | — |

All defined in `src/dpso_config.h`.

---

## 3. Architecture on the AIE array

Three kernel types, laid out as a dataflow graph and physically placed across AIE tiles:

```
Row 0:   [P0] [P1] [P2] [P3] [P4] [P5] [P6] [P7]      <- 8 particle kernels
            \  /       \  /       \  /       \  /
Row 1:     [R00]      [R01]      [R02]      [R03]     <- reduction, level 0 (4 kernels)
              \          /          \          /
Row 2:          [R10]                 [R11]           <- reduction, level 1 (2 kernels)
                    \                    /
Row 3:                    [R20]                        <- reduction, level 2 (1 kernel = swarm winner)
                             |
                        [BC root]                       <- broadcast, level 0 (1 kernel)
                          /      \
                    [BC10]        [BC11]                <- broadcast, level 1
                    /    \          /    \
                 [BC20][BC21]   [BC22][BC23]             <- broadcast, level 2
                   |     |         |     |
                  P0    P1  ...  P6    P7                <- fed back into particles
```

- **Particle kernels (`particle_kernel.cpp`)** — one per swarm member. Runs the DPSO update above, outputs `(cost, particle_id)` toward the reduction tree and its candidate tour toward the broadcast tree.
- **Reduction tree (`reduction_kernel.cpp`)** — binary tree of pairwise compare kernels. Each node reads two `(cost, id)` pairs and forwards whichever is lower. `log2(N_PARTICLES)` levels; the root holds the swarm-wide best each iteration.
- **Broadcast tree (`broadcast_root.cpp`, `broadcast_relay.cpp`)** — mirror of the reduction tree, top-down. `broadcast_root` reads the winning `(cost, id)`, selects that particle's full 50-city tour from its buffer inputs, and fans it out; relay kernels copy it down each level until every particle receives the new global best for the next iteration.
- **`gbest_out` (PLIO)** — taps the broadcast tree at level 1 to log the running global best to `gbest_out.txt`.

Particles talk to the tree over **AIE streams** (`input_stream`/`output_stream`); the distance matrix is delivered once per particle as an **RTP** (runtime parameter, `dist_matrix_rtp[]`) before the first iteration and cached in each tile's local memory — it is *not* re-sent every iteration.

### Why iteration 0 is a special case

On iteration 0 no global best exists yet, so particles skip the `in_gbest` stream read and instead reuse their own personal best. The reduction tree still runs on iteration 0 (particles always emit their cost), and `broadcast_root` writes the first real global best, which particles consume starting on iteration 1. This ordering is what avoids a deadlock on the very first graph run — see the comment block at the top of `broadcast_root.cpp`.

---

## 4. Repository layout

```
rtj/
├── src/
│   ├── dpso_config.h        # all tunable constants (N_PARTICLES, N_CITIES, PSO weights, iteration count)
│   ├── dpso_graph.h         # ADF graph: builds particle/reduction/broadcast trees, wiring, tile placement
│   ├── dpso_main.cpp        # entry point guarded for __X86SIM__ / __AIESIM__
│   ├── graph.cpp            # second, ungated entry point (see note below)
│   ├── kernels.h            # kernel function signatures shared across .cpp files
│   └── kernels/
│       ├── particle_kernel.cpp     # DPSO update per particle
│       ├── reduction_kernel.cpp    # pairwise cost compare
│       ├── broadcast_root.cpp      # selects and fans out the winning tour
│       └── broadcast_relay.cpp     # passes the tour down one more tree level
└── data/
    ├── dist_matrix_base.txt        # 50x50 distance matrix, 2500 integers, one per line
    └── trigger_0.txt … trigger_7.txt  # one file per particle, drives its PLIO trigger port
```

### Data file format

- **`dist_matrix_base.txt`** — flattened row-major 50×50 matrix (2,500 lines, one integer each). Loaded once at startup and written to every particle's RTP port (same instance for all particles — they're solving one shared TSP problem, not different ones).
- **`trigger_N.txt`** — drives particle `N`'s trigger input. Each file is `MAX_ITER` pairs of `(iteration_number, particle_id)` — 1,000 integers for 500 iterations — so the particle knows which iteration it's on and what its own ID is.

---

## 5. Building and running

It contains kernel/graph source and input data only. There is no `Makefile`, `v++`/AI Engine compiler config, or Vitis platform project checked in here, so it will not build standalone yet. To run it you need a Vitis Unified IDE (or `v++`/`aiecompiler`) AI Engine project targeting the VCK190, with these files added as sources.

General steps once you have a project set up:

1. Add everything under `src/` as kernel/graph sources; keep the `data/` folder alongside the working directory the simulator launches from — `dist_matrix_base.txt` is loaded with relative paths (`data/...`, `../../data/...`, etc.) and the loader prints exactly which path it tried if none resolve.
2. Pick **one** entry point — see the note below — and exclude the other from the build.
3. For a quick functional check, compile and run under the **x86 simulator** (`__X86SIM__`) — fastest turnaround, functional correctness only, no AIE timing/resource information.
4. For cycle-accurate behavior, run under the **AIE simulator** (`__AIESIM__`) / on hardware via the Versal AI Engine flow.

### Duplicate entry points — pick one

`dpso_main.cpp` and `graph.cpp` are **two independent `main()` functions** that both build the graph, load the distance matrix, and run it — `dpso_main.cpp` is guarded behind `__X86SIM__`/`__AIESIM__`, `graph.cpp` is not guarded at all and declares its own global graph instance (`g`). Compiling both into the same target will fail with a duplicate-`main()` link error. Keep one, or make `graph.cpp`'s guard match `dpso_main.cpp`'s if the intent was for it to be a second simulation target.

---

## 6. Known limitations

- **`N_PARTICLES` is not actually free to change**, despite the comment in `dpso_config.h` claiming the topology auto-generates. `dpso_graph.h`'s reduction/broadcast tree *construction* does generalize to other swarm sizes, but `broadcast_root.cpp`'s kernel signature hard-codes exactly 8 position-buffer inputs (`in_pos0`…`in_pos7`) and an 8-entry pointer table inside the kernel body. Changing `N_PARTICLES` away from 8 will break the build at `broadcast_root`, not just at the graph-topology level.
- **x86 functional simulation only** — nothing in this bundle confirms AIE-cycle-accurate timing or hardware resource usage (tile/memory/routing budget) has been validated for this graph.
- **No results committed.** `broadcast_root.cpp`'s own header notes a fix to the order it reads `(winner_cost, winner_id)` from the reduction tree — before that fix, `safe_id`'s range check failed on effectively every call and the kernel silently defaulted to particle 0 as "the winner" regardless of actual cost. Any makespan or improvement percentage measured before this fix does not reflect this code and should be re-run before being reported.
- **No license file** — add one before treating this as a public, reusable repo.

---

## 7. Background

Implemented as part of embedded/AIE systems coursework exploring parallel metaheuristics on spatial dataflow architectures — a precursor to a DAG-scheduling DPSO variant targeting the same VCK190 platform.

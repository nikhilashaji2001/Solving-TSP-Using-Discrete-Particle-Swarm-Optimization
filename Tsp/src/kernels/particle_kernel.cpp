#include <adf.h>
#include <stdint.h>
#include "../kernels.h"
#include "../dpso_config.h"

using namespace adf;

// __thread (TLS) is required on x86sim: all N_PARTICLES kernel instances run
// inside one shared OS process there, so without per-thread storage they'd
// stomp on each other's state. On real hardware and aiesim (both compiled by
// chess-clang, the actual tile-targeting backend) TLS is not implemented at
// all — "thread-local storage is not supported for the current target" — and
// it isn't needed there anyway: each particle kernel is compiled separately
// and deployed to its own physically separate tile, so plain static storage
// is already private per-instance by construction.
#ifdef __X86SIM__
#define TL_STORAGE __thread
#else
#define TL_STORAGE static
#endif

TL_STORAGE int32_t  tl_position[N_CITIES];
TL_STORAGE int32_t  tl_velocity[MAX_VELOCITY * 2];
TL_STORAGE int32_t  tl_vel_len;
TL_STORAGE int32_t  tl_pbest[N_CITIES];
TL_STORAGE int32_t  tl_pbest_cost;
TL_STORAGE int32_t  tl_particle_id;
TL_STORAGE uint32_t tl_rng_state;
TL_STORAGE bool     tl_initialised;
TL_STORAGE uint8_t  tl_dist_matrix[N_CITIES * N_CITIES];  // Store matrix locally (uint8: verified max=120)

// Stack-size fix: these were previously stack-local inside compute_velocity()
// and particle_kernel(), contributing to a 1472/1024-byte stack overflow on
// hardware (5 of 8 particle cores over budget). compute_velocity() is called
// sequentially (never nested/reentrant), so shared scratch storage is safe.
TL_STORAGE int32_t  tl_cv_tmp[N_CITIES];
TL_STORAGE int32_t  tl_cv_pos[N_CITIES];
TL_STORAGE int32_t  tl_gbest_scratch[N_CITIES];

static inline uint32_t lcg_next() {
    tl_rng_state = tl_rng_state * 1664525u + 1013904223u;
    return tl_rng_state;
}

static inline int32_t lcg_range(int32_t n) {
    return (int32_t)((lcg_next() >> 8) % (uint32_t)n);
}

static inline int32_t lcg_byte() {
    return (int32_t)((lcg_next() >> 16) & 0xFF);
}

static int32_t eval_cost(const int32_t* tour, const uint8_t* dist_matrix) {
    int32_t cost = 0;
    for (int i = 0; i < N_CITIES; i++) {
        int32_t from = tour[i];
        int32_t to   = tour[(i + 1) % N_CITIES];
        cost += (int32_t)dist_matrix[from * N_CITIES + to];  // widen before accumulating — do not accumulate in uint8/uint16
    }
    return cost;
}

static int32_t compute_velocity(const int32_t* from, const int32_t* to,
                                 int32_t* out_vel) {
    int32_t* tmp = tl_cv_tmp;
    int32_t* pos = tl_cv_pos;
    for (int i = 0; i < N_CITIES; i++) { tmp[i] = from[i]; pos[from[i]] = i; }
    int32_t n_swaps = 0;
    for (int i = 0; i < N_CITIES - 1 && n_swaps < MAX_VELOCITY; i++) {
        if (tmp[i] != to[i]) {
            int32_t j = pos[to[i]];
            out_vel[n_swaps*2] = i; out_vel[n_swaps*2+1] = j;
            n_swaps++;
            pos[tmp[i]] = j; pos[tmp[j]] = i;
            int32_t t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t;
        }
    }
    return n_swaps;
}

static void apply_velocity(int32_t* tour, const int32_t* vel, int32_t n_swaps) {
    for (int k = 0; k < n_swaps; k++) {
        int32_t i = vel[k*2], j = vel[k*2+1];
        int32_t t = tour[i]; tour[i] = tour[j]; tour[j] = t;
    }
}

static void shuffle_tour(int32_t* tour) {
    for (int i = 0; i < N_CITIES; i++) tour[i] = i;
    for (int i = N_CITIES - 1; i > 0; i--) {
        int32_t j = lcg_range(i + 1);
        int32_t t = tour[i]; tour[i] = tour[j]; tour[j] = t;
    }
}

void particle_kernel(
    input_stream<int32>*    in_trigger,
    input_stream<int32>*    in_gbest,
    const uint8 (&in_dist_matrix)[N_CITIES * N_CITIES],
    output_stream<int32>*   out_pbest,
    output_buffer<int32>&   out_position
)
{
    int32_t iter_num    = readincr(in_trigger);
    int32_t particle_id = readincr(in_trigger);

    if (!tl_initialised) {
        tl_particle_id = particle_id;
        tl_rng_state = 12345u + (uint32_t)(particle_id * 2654435761u);
        for (int i = 0; i < 20 + particle_id * 7; i++) lcg_next();

        // RTP: data was written once via graph.update() before run() started.
        // No DMA refill per invocation — copy into thread-local storage once,
        // same caching pattern as before, just a different source type.
        for (int i = 0; i < N_CITIES * N_CITIES; i++) {
            tl_dist_matrix[i] = in_dist_matrix[i];
        }

        shuffle_tour(tl_position);

        tl_vel_len = 5 + (int32_t)(lcg_range(6));
        for (int k = 0; k < tl_vel_len; k++) {
            tl_velocity[k*2]   = lcg_range(N_CITIES);
            tl_velocity[k*2+1] = lcg_range(N_CITIES);
        }

        tl_pbest_cost = eval_cost(tl_position, tl_dist_matrix);
        for (int i = 0; i < N_CITIES; i++) tl_pbest[i] = tl_position[i];
        tl_initialised = true;
    }

    int32_t* gbest = tl_gbest_scratch;
    if (iter_num == 0) {
        for (int i = 0; i < N_CITIES; i++) gbest[i] = tl_pbest[i];
    } else {
        for (int i = 0; i < N_CITIES; i++) gbest[i] = readincr(in_gbest);
    }

    int32_t v_cognitive[MAX_VELOCITY*2], v_social[MAX_VELOCITY*2];
    int32_t n_cog = compute_velocity(tl_position, tl_pbest, v_cognitive);
    int32_t n_soc = compute_velocity(tl_position, gbest,    v_social);

    int32_t new_vel[MAX_VELOCITY*2], new_vel_len = 0;

    for (int k = 0; k < tl_vel_len && new_vel_len < MAX_VELOCITY; k++)
        if (lcg_byte() < W_SCALED) {
            new_vel[new_vel_len*2]   = tl_velocity[k*2];
            new_vel[new_vel_len*2+1] = tl_velocity[k*2+1];
            new_vel_len++;
        }
    for (int k = 0; k < n_cog && new_vel_len < MAX_VELOCITY; k++)
        if (lcg_byte() < C1_SCALED) {
            new_vel[new_vel_len*2]   = v_cognitive[k*2];
            new_vel[new_vel_len*2+1] = v_cognitive[k*2+1];
            new_vel_len++;
        }
    for (int k = 0; k < n_soc && new_vel_len < MAX_VELOCITY; k++)
        if (lcg_byte() < C2_SCALED) {
            new_vel[new_vel_len*2]   = v_social[k*2];
            new_vel[new_vel_len*2+1] = v_social[k*2+1];
            new_vel_len++;
        }

    if (new_vel_len == 0) {
        new_vel[0] = lcg_range(N_CITIES);
        new_vel[1] = lcg_range(N_CITIES);
        new_vel_len = 1;
    }

    tl_vel_len = new_vel_len;
    for (int k = 0; k < tl_vel_len*2; k++) tl_velocity[k] = new_vel[k];
    apply_velocity(tl_position, tl_velocity, tl_vel_len);

    // Use locally stored distance matrix
    int32_t new_cost = eval_cost(tl_position, tl_dist_matrix);
    if (new_cost < tl_pbest_cost) {
        tl_pbest_cost = new_cost;
        for (int i = 0; i < N_CITIES; i++) tl_pbest[i] = tl_position[i];
    }

    int32_t* out_ptr = (int32_t*)out_position.data();
    for (int i = 0; i < N_CITIES; i++) out_ptr[i] = tl_pbest[i];
    writeincr(out_pbest, tl_pbest_cost);
    writeincr(out_pbest, tl_particle_id);
}
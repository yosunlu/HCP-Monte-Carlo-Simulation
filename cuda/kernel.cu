#include "kernel.h"

__global__ void mc_kernel_shared(
    float *d_s,      // device storage for results
    float T,
    float K,         // strike price
    float B,         // barrier price
    float S0,        // initial price
    float sigma,     // volatility
    float mu,        // drift
    float r,         // risk-free rate
    float dt,        // time step size
    float *d_normals, // random numbers
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    extern __shared__ float sdata[]; // shared memory for intermediate path storage

    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned bsz = blockDim.x;

    // Shared memory index for this thread
    float *s_path = &sdata[tid * N_STEPS];

    // Global indices
    int s_idx = tid + bid * bsz; // global path index
    int n_idx = s_idx * N_STEPS; // global random number index

    if (s_idx < N_PATHS) {
        float s_curr = S0;

        // Store initial value in shared memory
        s_path[0] = s_curr;

        // Euler discretization for this path
        for (int n = 0; n < N_STEPS; n++) {
            s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * d_normals[n_idx + n];
            s_path[n] = s_curr; // Save intermediate step in shared memory
            if (s_curr <= B) {
                break; // Stop if barrier condition is met
            }
        }

        // Compute payoff
        float payoff = (s_curr > K ? s_curr - K : 0.0);

        // Synchronize threads to ensure all shared memory updates are complete
        __syncthreads();

        // Store discounted payoff in global memory
        d_s[s_idx] = exp(-r * T) * payoff;
    }
}


    void mc_dao_call(
        float * d_s,
        float T,
        float K,
        float B,
        float S0,
        float sigma,
        float mu,
        float r,
        float dt,
        float * d_normals,
        unsigned N_STEPS,
        unsigned N_PATHS) {
            const unsigned BLOCK_SIZE = 1024;
            const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
            size_t shared_mem_size = BLOCK_SIZE * N_STEPS * sizeof(float); // Shared memory size
            mc_kernel_shared<<<GRID_SIZE, BLOCK_SIZE, shared_mem_size>>>(
                d_s, T, K, B, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
        }
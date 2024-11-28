#include "kernel.h"

__global__ void mc_kernel(
    float *d_s,  // device storage, data of which will be copied back to host
    float T,
    float K,     // strike price
    float B,     // barrier price
    float S0,    // market price
    float sigma, // expected volatility per year
    float mu,    // expected return per year
    float r,     // risk-free rate
    float dt,    // amount of time elapsing at each step
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    const unsigned tid = threadIdx.x;    // Thread index within block
    const unsigned bid = blockIdx.x;    // Block index
    const unsigned bsz = blockDim.x;    // Threads per block
    const unsigned s_idx = tid + bid * bsz; // Global thread ID

    if (s_idx < N_PATHS)
    {
        // Starting index for this thread's random numbers
        int base_idx = s_idx * N_STEPS;

        // Initialize current stock price
        float s_curr = S0;

        int n = 0;
        do
        {
            // Use random numbers for Euler discretisation
            s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * d_normals[base_idx + n];
            n++;
        } while (n < N_STEPS && s_curr > B);

        // Compute payoff
        float payoff = (s_curr > K ? s_curr - K : 0.0);
        d_s[s_idx] = exp(-r * T) * payoff;
    }
}


void mc_dao_call(
    float *d_s,
    float T,
    float K,
    float B,
    float S0,
    float sigma,
    float mu,
    float r,
    float dt,
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    const unsigned BLOCK_SIZE = 32;
    const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
    mc_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        d_s, T, K, B, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
}


__global__ void mc_kernel_shared(
    float *d_s,  // device storage, data of which will be copied back to host
    float T,
    float K,     // strike price
    float B,     // barrier price
    float S0,    // market price
    float sigma, // expected volatility per year
    float mu,    // expected return per year
    float r,     // risk-free rate
    float dt,    // amount of time elapsing at each step
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    const unsigned tid = threadIdx.x;    // Thread index within block
    const unsigned bid = blockIdx.x;    // Block index
    const unsigned bsz = blockDim.x;    // Threads per block
    const unsigned s_idx = tid + bid * bsz; // Global thread ID

    // Shared memory allocation for this block
    extern __shared__ float shared_mem[];

    // Starting index in global memory for this thread's random numbers
    int base_idx = s_idx * N_STEPS;

    // Load random numbers into shared memory
    for (size_t i = 0; i < N_STEPS; ++i) {
        shared_mem[tid * N_STEPS + i] = d_normals[base_idx + i];
    }

    if (s_idx < N_PATHS)
    {
        // Initialize current stock price
        float s_curr = S0;

        // Perform simulation
        int s = 0;
        do
        {
            // Euler discretisation using shared memory
            s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * shared_mem[tid * N_STEPS + s];
            s++;
        } while (s < N_STEPS && s_curr > B);

        // Compute payoff
        float payoff = (s_curr > K ? s_curr - K : 0.0);
        d_s[s_idx] = exp(-r * T) * payoff;
    }
}

void mc_dao_call_shared(
    float *d_s,
    float T,
    float K,
    float B,
    float S0,
    float sigma,
    float mu,
    float r,
    float dt,
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    const unsigned BLOCK_SIZE = 32;
    const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
    mc_kernel_shared<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * N_STEPS * sizeof(float)>>>(
        d_s, T, K, B, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
}
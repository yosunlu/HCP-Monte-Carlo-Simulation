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
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned bsz = blockDim.x;
    int s_idx = tid + bid * bsz;
    int n_idx = tid + bid * bsz;
    float s_curr = S0;
    if (s_idx < N_PATHS)
    {
        int n = 0;
        do
        {
            //  Euler discretisation
            s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * d_normals[n_idx];
            n_idx++;
            n++;
        } while (n < N_STEPS && s_curr > B);
        double payoff = (s_curr > K ? s_curr - K : 0.0);
        __syncthreads();
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
    const unsigned BLOCK_SIZE = 1024;
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
    const unsigned tid = threadIdx.x;    // Thread index within the block
    const unsigned bid = blockIdx.x;    // Block index
    const unsigned bsz = blockDim.x;    // Number of threads per block
    const unsigned s_idx = tid + bid * bsz;  // Global thread index

    // Allocate shared memory for a tile of random numbers
    extern __shared__ float shared_normals[];

    // Initialize variables
    float s_curr = S0;

    if (s_idx < N_PATHS)
    {
        // Process the path for this thread
        int n = 0;
        while (n < N_STEPS && s_curr > B)
        {
            // Load random numbers for this tile into shared memory
            if (tid + n < N_STEPS)
            {
                shared_normals[tid] = d_normals[s_idx * N_STEPS + n];
            }
            __syncthreads();

            // Use shared memory to update `s_curr`
            if (tid + n < N_STEPS)
            {
                s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * shared_normals[tid];
            }
            __syncthreads();

            n++;
        }

        // Compute payoff
        float payoff = (s_curr > K) ? s_curr - K : 0.0;
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
    const unsigned BLOCK_SIZE = 1024;
    const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));

    // Calculate shared memory size (one float per thread per block)
    const unsigned SHARED_MEM_SIZE = BLOCK_SIZE * sizeof(float);

    // Launch the kernel with shared memory size specified
    mc_kernel_shared<<<GRID_SIZE, BLOCK_SIZE, SHARED_MEM_SIZE>>>(
        d_s, T, K, B, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
}
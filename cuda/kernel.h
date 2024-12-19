#ifndef _KERNEL_CUH_ 
#define _KERNEL_CUH_

void mc_dao_call(float * d_s, float T, float K, float B, float S0, float sigma,
                 float mu, float r, float dt, float* d_normals, unsigned N_STEPS,
                 unsigned N_PATHS);

void mc_dao_call_shared(float * d_s, float T, float K, float B, float S0,
                        float sigma, float mu, float r, float dt, float* d_normals,
                        unsigned N_STEPS, unsigned N_PATHS);

// Add __global__ here
__global__ void rearrange_random_numbers(
    float *d_normals_src,
    float *d_normals_dst,
    unsigned N_STEPS,
    unsigned N_PATHS);

#endif
#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include "kernel.h"
#include "dev_array.h"
#include <curand.h>
#include <cstdlib>
#include <omp.h> // Include OpenMP header

using namespace std;

int main()
{
    try
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "Shared Memory Per SM: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;

        // Declare variables and constants
        // Dimensional constants
        const size_t N_PATHS = 5000000;
        const size_t N_STEPS = 365;
        const size_t N_NORMALS = N_PATHS * N_STEPS;

        // Market parameters
        const float T = 1.0f;
        const float K = 100.0f;   // Strike price
        const float B = 95.0f;    // Barrier price
        const float S0 = 100.0f;  // Market price
        const float sigma = 0.2f; // Expected volatility per year
        const float mu = 0.1f;    // Expected return per year
        const float r = 0.05f;    // Risk-free rate

        // Derived variables
        float dt = T / N_STEPS;
        float sqrdt = sqrt(dt); // Used for generating random numbers

        // Generate arrays
        vector<float> s(N_PATHS); // Host array for results
        dev_array<float> d_s(N_PATHS);        // Device array for results
        dev_array<float> d_s_shared(N_PATHS); // Device array for shared memory results

        dev_array<float> d_normals(N_NORMALS); // Array to store normally distributed random numbers

        // Create the CURAND generator
        curandGenerator_t curandGenerator;
        curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32); // Mersenne Twister algorithm
        curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);      // Seed for the generator

        // Allocate temporary device array
        dev_array<float> d_normals_temp(N_NORMALS);

        // Generate random numbers into the temporary array
        curandGenerateNormal(curandGenerator, d_normals_temp.getData(), N_NORMALS, 0.0f, sqrdt);

        // Rearrange random numbers for coalesced access
        const unsigned TOTAL_ELEMENTS = N_NORMALS;
        const unsigned BLOCK_SIZE_RAND = 256;
        const unsigned GRID_SIZE_RAND = (TOTAL_ELEMENTS + BLOCK_SIZE_RAND - 1) / BLOCK_SIZE_RAND;

        rearrange_random_numbers<<<GRID_SIZE_RAND, BLOCK_SIZE_RAND>>>(
            d_normals_temp.getData(),
            d_normals.getData(),
            N_STEPS,
            N_PATHS);

        // Clean up temporary array (optional)
        // d_normals_temp.clear();

        // Ensure CUDA calls have finished
        cudaDeviceSynchronize();

        // Set up CUDA events for timing
        cudaEvent_t start1, stop1, start2, stop2;
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);
        cudaEventCreate(&start2);
        cudaEventCreate(&stop2);

        // ================================
        // Run mc_dao_call (Original Kernel)
        // ================================
        // Start timing for mc_dao_call
        cudaEventRecord(start1);

        // Launch the original kernel
        mc_dao_call(
            d_s.getData(), T, K, B, S0, sigma, mu, r, dt,
            d_normals.getData(), N_STEPS, N_PATHS);

        // Stop timing for mc_dao_call
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);

        // Get elapsed time for mc_dao_call
        float ms1;
        cudaEventElapsedTime(&ms1, start1, stop1);

        // ================================
        // Run mc_dao_call_shared (Optimized Kernel)
        // ================================
        // Start timing for mc_dao_call_shared
        cudaEventRecord(start2);

        // Launch the optimized kernel
        mc_dao_call_shared(
            d_s_shared.getData(), T, K, B, S0, sigma, mu, r, dt,
            d_normals.getData(), N_STEPS, N_PATHS);

        // Stop timing for mc_dao_call_shared
        cudaEventRecord(stop2);
        cudaEventSynchronize(stop2);

        // Get elapsed time for mc_dao_call_shared
        float ms2;
        cudaEventElapsedTime(&ms2, start2, stop2);

        // Copy results from device to host
        vector<float> s_shared(N_PATHS); // Host array for shared memory results

        // Copy results from device to host
        d_s.get(&s[0], N_PATHS);
        d_s_shared.get(&s_shared[0], N_PATHS);

        // Compute the payoff average for mc_dao_call
        double temp_sum1 = 0.0;
        #pragma omp parallel for reduction(+:temp_sum1)
        for (int i = 0; i < N_PATHS; i++)
        {
            temp_sum1 += s[i];
        }
        temp_sum1 /= N_PATHS;

        // Compute the payoff average for mc_dao_call_shared
        double temp_sum2 = 0.0;
        #pragma omp parallel for reduction(+:temp_sum2)
        for (int i = 0; i < N_PATHS; i++)
        {
            temp_sum2 += s_shared[i];
        }
        temp_sum2 /= N_PATHS;

        // Output results
        cout << "****************** INFO ******************\n";
        cout << "Number of Paths: " << N_PATHS << "\n";
        cout << "Number of Steps: " << N_STEPS << "\n";
        cout << "Underlying Initial Price: " << S0 << "\n";
        cout << "Strike: " << K << "\n";
        cout << "Barrier: " << B << "\n";
        cout << "Time to Maturity: " << T << " years\n";
        cout << "Risk-free Interest Rate: " << r * 100 << "%\n";
        cout << "Annual drift: " << mu * 100 << "%\n";
        cout << "Volatility: " << sigma * 100 << "%\n";
        cout << "************** PRICE COMPARISON **************\n";
        cout << "Option Price (GPU - Original Kernel): " << temp_sum1 << "\n";
        cout << "Option Price (GPU - Optimized Kernel): " << temp_sum2 << "\n";
        cout << "************** TIME COMPARISON ***************\n";
        cout << "GPU Computation Time (Original Kernel): " << ms1 << " ms\n";
        cout << "GPU Computation Time (Optimized Kernel): " << ms2 << " ms\n";
        cout << "******************* END *****************\n";

        // Destroy CUDA events
        cudaEventDestroy(start1);
        cudaEventDestroy(stop1);
        cudaEventDestroy(start2);
        cudaEventDestroy(stop2);

        // Destroy generator
        curandDestroyGenerator(curandGenerator);
    }
    catch (exception &e)
    {
        cout << "Exception: " << e.what() << "\n";
    }
}
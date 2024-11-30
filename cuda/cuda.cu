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
        vector<float> s(N_PATHS); // Host array
        dev_array<float> d_s(N_PATHS);
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

        // Clean up temporary array
        // No need to explicitly call the destructor; it will be called automatically when the variable goes out of scope
        // Alternatively, you can reset the device memory if needed
        // d_normals_temp.clear();

        // Set up CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Start the clock
        cudaEventRecord(start);

        // Launch the kernel
        const unsigned BLOCK_SIZE = 128; // Adjust as needed
        const unsigned GRID_SIZE = (N_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        mc_dao_call_shared(
            d_s.getData(), T, K, B, S0, sigma, mu, r, dt,
            d_normals.getData(), N_STEPS, N_PATHS);

        // Stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Get elapsed time
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        // Copy results from device to host
        d_s.get(&s[0], N_PATHS);

        // Compute the payoff average
        double temp_sum = 0.0;
        #pragma omp parallel for reduction(+:temp_sum)
        for (int i = 0; i < N_PATHS; i++)
        {
            temp_sum += s[i];
        }
        temp_sum /= N_PATHS;

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
        cout << "****************** PRICE *****************\n";
        cout << "Option Price (GPU): " << temp_sum << "\n";
        cout << "******************* TIME *****************\n";
        cout << "GPU Monte Carlo Computation: " << ms << " ms\n";
        cout << "******************* END *****************\n";

        // Destroy generator
        curandDestroyGenerator(curandGenerator);
    }
    catch (exception &e)
    {
        cout << "Exception: " << e.what() << "\n";
    }
}

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

        // declare variables and constants
        // dimensional constants
        const size_t N_PATHS = 5000000;
        const size_t N_STEPS = 365;
        const size_t N_NORMALS = N_PATHS * N_STEPS;

        // market parameters
        const float T = 1.0f;
        const float K = 100.0f;   // strike price
        const float B = 95.0f;    // barrier price
        const float S0 = 100.0f;  // market price
        const float sigma = 0.2f; // expected volatility per year
        const float mu = 0.1f;    // expected return per year
        const float r = 0.05f;    // risk-free rate

        // derived variables
        float dt = float(T) / float(N_STEPS);
        float sqrdt = sqrt(dt); // used for generating random numbers

        // generate arrays
        vector<float> s(N_PATHS); // host array
        dev_array<float> d_s(N_PATHS);
        dev_array<float> d_normals(N_NORMALS); // array to store normally distributed random numbers

        // generate random numbers
        curandGenerator_t curandGenerator;
        curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32); // Mersenne Twister algorithm
        curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL); // seed for the generator
        curandGenerateNormal(curandGenerator, d_normals.getData(), N_NORMALS, 0.0f, sqrdt); //  generate normally distributed random numbers, using Brownian motion

        // Set up CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // start the clock 
        cudaEventRecord(start);

        // call the kernel
        mc_dao_call(d_s.getData(), T, K, B, S0, sigma, mu, r, dt, d_normals.getData(), N_STEPS, N_PATHS);

        // End the clock
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Get the elapsed time in milliseconds
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        // Copy results from device to host
        d_s.get(&s[0], N_PATHS);

        // compute the payoff average
        double temp_sum = 0.0;
        for (size_t i = 0; i < N_PATHS; i++)
        {
            temp_sum += s[i];
        }
        temp_sum /= N_PATHS;

        
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

        // destroy generator
        curandDestroyGenerator(curandGenerator);
    }
    catch (exception &e)
    {
        cout << "exception: " << e.what() << "\n";
    }
}

#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include "dev_array.h"

using namespace std;

int main()
{
    try
    {
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
        float dt = float(T) / float(N_STEPS); // amount of time elapsing at each step
        float sqrdt = sqrt(dt);

        // use GPU to generate random numbers
        // generate array to store normally distributed random numbers
        dev_array<float> d_normals(N_NORMALS); // array

        // generate random numbers
        curandGenerator_t curandGenerator;
        curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32); // Mersenne Twister algorithm
        curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);
        curandGenerateNormal(curandGenerator, d_normals.getData(), N_NORMALS, 0.0f, sqrdt);

        // init variables for CPU Monte Carlo
        vector<float> normals(N_NORMALS);

        // copy the random number generated on GPU back to CPU
        d_normals.get(&normals[0], N_NORMALS);

        // CPU Monte Carlo Simulation
        double sum = 0.0;

        double start = double(clock()) / CLOCKS_PER_SEC;

        for (size_t i = 0; i < N_PATHS; i++)
        {
            int n_idx = i * N_STEPS;

            float s_curr = S0;
            int n = 0;

            do
            {
                s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * normals[n_idx];
                n_idx++;
                n++;
            } while (n < N_STEPS && s_curr > B);

            double payoff = (s_curr > K ? s_curr - K : 0.0);
            sum += exp(-r * T) * payoff;
        }

        sum /= N_PATHS;
        double end = double(clock()) / CLOCKS_PER_SEC;

        cout << "****************** INFO ******************\n";
        cout << "Number of Paths: " << N_PATHS << "\n";
        cout << "Number of Steps: " << N_STEPS << "\n";
        cout << "Underlying Initial Price: " << S0 << "\n";
        cout << "Strike: " << K << "\n";
        cout << "Barrier: " << B << "\n";
        cout << "Time to Maturity: " << T << " years\n";
        cout << "Risk-free Interest Rate: " << r << "%\n";
        cout << "Annual drift: " << mu * 100 << "%\n";
        cout << "Volatility: " << sigma * 100 << "%\n";
        cout << "****************** PRICE ******************\n";
        cout << "Option Price (CPU): " << sum << "\n";
        cout << "******************* TIME *****************\n";
        cout << "CPU Monte Carlo Computation: " << (end - start) * 1e3 << " ms\n";
        cout << "******************* END *****************\n";

        curandDestroyGenerator(curandGenerator);
    }
    catch (exception &e)
    {
        cout << "exception: " << e.what() << "\n";
    }
}

#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include "kernel.h"
#include "dev_array.h"
#include <curand.h>
#include <omp.h> // Include OpenMP header

using namespace std;

int main() {
    try {
        // declare variables and constants
        // dimensional constants 
        const size_t N_PATHS = 5000000;
        const size_t N_STEPS = 730;
        const size_t N_NORMALS = N_PATHS*N_STEPS;

        // market parameters 
        const float T = 1.0f;
        const float K = 100.0f; // strike price
        const float B = 95.0f; // barrier price
        const float S0 = 100.0f; // market price 
        const float sigma = 0.2f; // expected volatility per year
        const float mu = 0.1f; // expected return per year
        const float r = 0.05f; // 

        // derived variables 
        float dt = float(T)/float(N_STEPS);
        float sqrdt = sqrt(dt);

        // generate arrays
        cout << "debug\n" << endl;
        vector<float> s(N_PATHS);
        dev_array<float> d_s(N_PATHS);
        dev_array<float> d_normals(N_NORMALS); // array to store normally distributed random numbers
        
        // generate random numbers
        curandGenerator_t curandGenerator;
        curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32); // Mersenne Twister algorithm 
        curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL) ;
        curandGenerateNormal(curandGenerator, d_normals.getData(), N_NORMALS, 0.0f, sqrdt);
        double t2=double(clock())/CLOCKS_PER_SEC;
        
        // call the kernel
        mc_dao_call(d_s.getData(), T, K, B, S0, sigma, mu, r, dt, d_normals.getData(), N_STEPS, N_PATHS);
        cudaDeviceSynchronize();

        // copy results from device to host
        d_s.get(&s[0], N_PATHS);

        // compute the payoff average
        double temp_sum=0.0;
        for(size_t i=0; i<N_PATHS; i++) {
            temp_sum += s[i];
        }
        temp_sum/=N_PATHS;
        double t4=double(clock())/CLOCKS_PER_SEC;

        // init variables for CPU Monte Carlo
        // vector<float> normals(N_NORMALS);
        // d_normals.get(&normals[0],N_NORMALS);

        // // CPU Monte Carlo Simulation
        // double sum=0.0;

        // for (size_t i=0; i<N_PATHS; i++) {
        //     int n_idx = i*N_STEPS;

        //     float s_curr=S0;
        //     int n=0;

        //     do {
        //         s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*normals[n_idx];
        //         n_idx ++;
        //         n++;
        //     }
        //     while (n<N_STEPS && s_curr>B);
            
        //     double payoff = (s_curr>K ? s_curr-K : 0.0);
        //     sum += exp(-r*T) * payoff;
        // }

        // sum/=N_PATHS;
        // double t5=double(clock())/CLOCKS_PER_SEC;

        // // OpenMP CPU Monte Carlo Simulation
        // double sum_openmp = 0.0;
        // double t6 = double(clock())/CLOCKS_PER_SEC;

        // #pragma omp parallel for reduction(+:sum_openmp)
        // for (int i=0; i<N_PATHS; i++) {
        //     int n_idx = i*N_STEPS;

        //     float s_curr=S0;
        //     int n=0;

        //     do {
        //         s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*normals[n_idx];
        //         n_idx ++;
        //         n++;
        //     }
        //     while (n<N_STEPS && s_curr>B);
            
        //     double payoff = (s_curr>K ? s_curr-K : 0.0);
        //     sum_openmp += exp(-r*T) * payoff;
        // }

        // sum_openmp/=N_PATHS;
        // double t7=double(clock())/CLOCKS_PER_SEC;

        cout<<"****************** INFO ******************\n";
        cout<<"Number of Paths: " << N_PATHS << "\n";
        cout<<"Number of Steps: " << N_STEPS << "\n";
        cout<<"Underlying Initial Price: " << S0 << "\n";
        cout<<"Strike: " << K << "\n";
        cout<<"Barrier: " << B << "\n";
        cout<<"Time to Maturity: " << T << " years\n";
        cout<<"Risk-free Interest Rate: " << r << "%\n";
        cout<<"Annual drift: " << mu << "%\n";
        cout<<"Volatility: " << sigma << "%\n";
        cout<<"****************** PRICE ******************\n";
        cout<<"Option Price (GPU): " << temp_sum << "\n";
        // cout<<"Option Price (CPU): " << sum << "\n";
        // cout<<"Option Price (CPU with OpenMP): " << sum_openmp << "\n";
        cout<<"******************* TIME *****************\n";
        cout<<"GPU Monte Carlo Computation: " << (t4-t2)*1e3 << " ms\n";
        // cout<<"CPU Monte Carlo Computation: " << (t5-t4)*1e3 << " ms\n";
        // cout<<"CPU with OpenMP Monte Carlo Computation: " << (t7-t6)*1e3 << " ms\n";
        cout<<"******************* END *****************\n";

        // destroy generator
        curandDestroyGenerator( curandGenerator ) ;
    }
    catch(exception& e) {
        cout<< "exception: " << e.what() << "\n";
    }
}

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <ctime>
#include <fstream> // For CSV file output
#include <omp.h>   // Include OpenMP header

using namespace std;

int main()
{
    try
    {
        // Open a CSV file to save the results
        ofstream csv_file("execution_times.csv");
        if (!csv_file)
        {
            cerr << "Error: Could not open CSV file for writing.\n";
            return 1;
        }
        csv_file << "Threads,ExecutionTime(ms)\n";

        // Declare variables and constants
        const size_t N_PATHS = 5000000;
        const size_t N_STEPS = 365;
        const size_t N_NORMALS = N_PATHS * N_STEPS;

        // Market parameters
        const float T = 1.0f;
        const float K = 100.0f;   // strike price
        const float B = 95.0f;    // barrier price
        const float S0 = 100.0f;  // market price
        const float sigma = 0.2f; // expected volatility per year
        const float mu = 0.1f;    // expected return per year
        const float r = 0.05f;    // risk-free rate

        // Derived variables
        float dt = float(T) / float(N_STEPS); // amount of time elapsing at each step
        float sqrdt = sqrt(dt);

        // Generate normally distributed random numbers using <random>
        vector<float> normals(N_NORMALS);
        {
            random_device rd;  // Random number seed
            mt19937 gen(rd()); // Mersenne Twister generator
            normal_distribution<float> dist(0.0f, sqrdt);

            for (size_t i = 0; i < N_NORMALS; ++i)
            {
                normals[i] = dist(gen);
            }
        }

        // Test execution times for thread counts from 1 to 20
        for (int thread_count = 1; thread_count <= 15; ++thread_count)
        {
            // Set the number of threads for OpenMP
            omp_set_num_threads(thread_count);

            // OpenMP CPU Monte Carlo Simulation
            double sum_openmp = 0.0;
            double t6 = double(clock()) / CLOCKS_PER_SEC;

            #pragma omp parallel for reduction(+ : sum_openmp)
            for (int i = 0; i < N_PATHS; i++)
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
                sum_openmp += exp(-r * T) * payoff;
            }

            sum_openmp /= N_PATHS;
            double t7 = double(clock()) / CLOCKS_PER_SEC;

            // Calculate execution time in milliseconds
            double exec_time = (t7 - t6) * 1e3;

            // Write results to the CSV file
            csv_file << thread_count << "," << exec_time << "\n";
            cout << "Threads: " << thread_count << ", Execution Time: " << exec_time << " ms\n";
        }

        // Close the CSV file
        csv_file.close();
        cout << "Execution times saved to 'execution_times.csv'.\n";
    }
    catch (exception &e)
    {
        cout << "Exception: " << e.what() << "\n";
    }

    return 0;
}

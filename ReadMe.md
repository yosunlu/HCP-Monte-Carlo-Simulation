# Monte Carlo Simulation on Barrier Option with CUDA  

This repository demonstrates the Parallelization of Monte Carlo Simulation for Barrier Option Pricing, a project showcasing the application of GPU-accelerated simulations to efficiently price exotic financial options. The project compares sequential, parallelized CPU (OpenMP), and GPU-accelerated (CUDA) implementations, analyzing their performance, scalability, and suitability for high-performance computing in finance.

## 🔍Introduction
Monte Carlo simulations are a key tool for valuing path-dependent exotic options like barrier options, which lack closed-form solutions such as the Black-Scholes formula. This project simulates 5 million paths with 365 time steps per path, leading to over 1.8 billion random number computations. Using GPU acceleration with CUDA, we achieve significant speedups compared to traditional CPU-based approaches.

## 💡Key Features
- Barrier Options Pricing: Handles knock-in and knock-out options.
- Parallelization Strategies: Implements three approaches:
  - Sequential CPU Implementation (Baseline)
  - Parallelized CPU Implementation using OpenMP
  - GPU-Accelerated Implementation using CUDA
- Optimization Techniques:
  - Global memory coalescing for a 1.6x speedup.
  - Efficient handling of large datasets without relying on shared memory.

## 📜 Background
### What are Options?
Options are financial derivatives giving the holder the right to buy or sell an asset at a predetermined price within a specified time.
### What are Barrier Options?
Barrier options depend on whether the asset's price hits a specified level (barrier) during its life:
- Knock-In: Activated if the price breaches the barrier.
- Knock-Out: Deactivated if the price breaches the barrier.

## ⚙️ Implementation
1. Sequential CPU Implementation
- Uses C++ standard library <random> for random number generation.
- Simulates price paths using Geometric Brownian Motion.
- Computationally intensive and limited scalability.
2. Parallelized CPU Implementation with OpenMP
- Parallelizes simulation across CPU threads.
- Pre-generates random numbers using GPU.
- Efficient for systems without GPU support.
3. GPU-Accelerated Implementation with CUDA
- Fully parallelized random number generation using curand.
- Assigns each path to a GPU thread for massive parallelism.
- Leverages memory coalescing for optimized performance.

## 🚀 Results
- Performance Metrics:
  - Sequential CPU (Intel Xeon E5-2650 v3): 113,392 ms.
  - Parallel CPU (OpenMP): 69,873 ms (~1.6x speedup).
  - GPU (NVIDIA RTX 4000 Ada):
    - Block size 16: 93 ms (~751x speedup).
    - Block size 1024 with memory coalescing: 60 ms (~1,126x speedup).
Cumulative GPU Speedup: ~1,889x.

## 📈Optimization Highlights
- Shared Memory Challenges: Effective use of global memory made shared memory unnecessary.
- Global Memory Coalescing: Reordering data for optimal memory access patterns achieved a 1.6x speedup.


## 📚 Conclusion
This project highlights the power of parallel computing and CUDA optimizations in financial simulations, achieving extraordinary performance improvements. GPU technology proves invaluable for large-scale, computationally intensive simulations.

## Technologies Used
- Programming Languages: C++, CUDA
- Frameworks and Libraries: OpenMP, curand (CUDA random number library)
- Hardware:
  - CPU: Intel Xeon E5-2650 v3
  - GPU: NVIDIA RTX 4000 Ada

## 🧪How to Run on a Euler Machine
1. Clone the repository  
``
git clone https://github.com/yosunlu/HPC-Monte-Carlo-Simulation.git
cd HPC-Monte-Carlo-Simulation
``

3. Run the scripts  
``
make sequential  
make openmp  
make cuda  
``

#### Reference:
https://www.quantstart.com/articles/Monte-Carlo-Simulations-In-CUDA-Barrier-Option-Pricing/

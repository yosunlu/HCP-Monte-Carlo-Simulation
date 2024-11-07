# Monte Carlo Simulation on Barrier Option with CUDA  

This project leverages CUDA and OpenMP to perform Monte Carlo simulation for Barrier Option pricing, utilizing GPU parallelism to accelerate computation.

## Quick Start

#### Compile with Windows PowerShell

Ensure the NVIDIA CUDA toolkit and `nvcc` compiler are installed. Then compile the project by running:

```powershell
nvcc -o barrier_option_pricing main.cpp kernel.cu -lcurand -Xcompiler /openmp
```
#### Run Precompiled Executable  
Alternatively, you can download and run the latest build of the executable without compiling.  

#### Tested environment details:
Operating System: Windows 11  
CPU: Intel i7-12700k  
GPU: NVIDIA RTX 3080  
Build Date: 2024-11-07

#### Test Output Result
![image](https://github.com/user-attachments/assets/d2f06fb0-9983-41d1-b46d-ab03033eb041)

#### Reference:
https://www.quantstart.com/articles/Monte-Carlo-Simulations-In-CUDA-Barrier-Option-Pricing/

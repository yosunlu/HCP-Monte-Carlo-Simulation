#!/usr/bin/env bash
#SBATCH --job-name=openmp
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:20:00
#SBATCH --output="./output/openmp.out"
#SBATCH --error="./output/openmp.err"
#SBATCH --gres=gpu:1

# Add the output folder if not already exists
mkdir -p output

# Load the modules 
module load nvidia/cuda/11.8.0
module load gcc/9.4.0

nvcc -o barrier_option_pricing_openmp ./openmp/openmp.cpp -lcurand -Xcompiler -fopenmp
./barrier_option_pricing_openmp

rm barrier_option_pricing_openmp
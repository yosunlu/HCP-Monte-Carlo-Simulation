#!/usr/bin/env bash
#SBATCH --job-name=cuda
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:20:00
#SBATCH --output="./output/openmp.out"
#SBATCH --error="./output/openmp.err"
#SBATCH --gres=gpu:1

# Add the output folder if not already exists
mkdir -p output

nvcc -o barrier_option_pricing_openmp ./cuda/openmp.cpp -lcurand -Xcompiler -fopenmp
./barrier_option_pricing_openmp
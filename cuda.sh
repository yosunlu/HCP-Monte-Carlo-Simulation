#!/usr/bin/env bash
#SBATCH --job-name=task1
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="./output/cuda.out"
#SBATCH --error="./output/cuda.err"
#SBATCH --gres=gpu:1

# Add the output folder if not already exists
mkdir -p output

# Load the modules 
module load nvidia/cuda/11.8.0
module load gcc/9.4.0

nvcc -o barrier_option_pricing main.cpp kernel.cu -lcurand -Xcompiler -fopenmp
./barrier_option_pricing >> ./output/cuda.out

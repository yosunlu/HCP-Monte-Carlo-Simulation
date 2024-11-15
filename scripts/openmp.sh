#!/usr/bin/env bash
#SBATCH --job-name=cuda
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:20:00
#SBATCH --output="./output/cuda.out"
#SBATCH --error="./output/cuda.err"
#SBATCH --gres=gpu:1

# Add the output folder if not already exists
mkdir -p output

g++ ./openmp/main.cpp -Wall -O3 -std=c++17 -o barrier_option_pricing_openmp -fopenmp
./barrier_option_pricing_openmp
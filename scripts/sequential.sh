#!/usr/bin/env bash
#SBATCH --job-name=sequential
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:20:00
#SBATCH --output="./output/seq.out"
#SBATCH --error="./output/seq.err"
#SBATCH --gres=gpu:1

# Add the output folder if not already exists
mkdir -p output

# Load the modules 
module load nvidia/cuda/11.8.0
module load gcc/9.4.0

# Print CPU information
echo "****************** CPU INFO ******************"
lscpu

# Print GPU information
echo "****************** GPU INFO ******************"
nvidia-smi

# Compile the program
nvcc -o barrier_option_pricing_seq ./sequential/sequential.cpp -lcurand -Xcompiler -fopenmp

# Run the program
echo "****************** PROGRAM OUTPUT ******************"
./barrier_option_pricing_seq

rm barrier_option_pricing_seq
#!/usr/bin/env zsh
#SBATCH --job-name=tasktest
#SBATCH --partition=instruction    # Ensure this partition has GPU access
#SBATCH --time=00:02:00            # Corrected the syntax
#SBATCH --ntasks=1
#SBATCH --gpus=1                   
#SBATCH --cpus-per-task=1          
#SBATCH --output=task.out          
#SBATCH --error=task.err  
#SBATCH --mem=6000          

cd $SLURM_SUBMIT_DIR

# git clone https://github.com/Ashar25/repo759.git

# cd repo759
# cd HW06

# Load necessary modules
module load nvidia/cuda/11.8.0
module load gcc/11.3.0                  # Ensure CUDA and GCC are compatible

# Compile the code with nvcc
nvcc tasktest.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task



# Run the compiled CUDA executable
./task 5 1 5

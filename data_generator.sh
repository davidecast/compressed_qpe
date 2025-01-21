#!/bin/bash

# ==========================
# Bash Script for Parameter Sweep
# ==========================
# This script scans through multiple parameter combinations,
# runs a Python script for each combination, and saves the results.

# Activate virtual environment
source /data/soft/centos7/interpreter/compressed_env/bin/activate

# Set threading environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# ---- Parameters ----
# Define the ranges or lists of parameters
nstep_values=(250 350 450 550 650)           
molecule_names=("lih")        

# Output file for results
output_file="out_sampler.out"

# Path to your Python script
python_script="./molecular_autocorrelation.py"

# ---- Setup ----
# Create or clear the output file
echo "nstep,molecule,result" > "$output_file"  # Header row

# ---- Main Loop ----
# Iterate through all combinations of parameters
for nstep in "${nstep_values[@]}"; do

        echo "Running with nstep=$nstep"

        # Run the Python script and capture the output
        if ! python "$python_script" --nstep "$nstep" --molecule "lih" >> "$output_file"; then
            echo "Error running Python script for nstep=$nstep, molecule=$molecule" >> "$output_file"
        else
            echo "Completed: nstep=$nstep, molecule=$molecule" >> "$output_file"
        fi

        # Add a separator for readability
        echo "---------------------------" >> "$output_file"

done

echo "Parameter sweep completed. Results saved to $output_file."

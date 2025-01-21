#!/bin/bash

# ==========================
# Bash Script for Parameter Sweep with Dynamic Filenames
# ==========================

# Activate virtual environment
source /data/soft/centos7/interpreter/compressed_env/bin/activate

# Set threading environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ---- Parameters ----
# Hardcoded or user-defined constants
dt=0.005                               # Replace with your hardcoded value
molecule="lih"                       # Replace with the molecule of interest
nstep_values=(250 350 450 550 650)              # Replace with the range of nstep values
shots_values=(5 20 35 50 65)           # Replace with the range of shot values
samples_values=(10 20 40 80)          # Replace with the range of sample values
init_guess=(0 1)                      # Replace with your desired boolean value

# Path to your Python script
python_script="./compressed_qpe.py"

# ---- Main Loop ----
# Iterate through all combinations of parameters
for nstep in "${nstep_values[@]}"; do
    for shots in "${shots_values[@]}"; do
        for samples in "${samples_values[@]}"; do
            for init in "${init_guess[@]}"; do

                # Calculate T
                T=$(echo "$nstep * $dt" | bc -l)
                formatted_T=$(printf "%.17g" "$T")

                # Dynamically generate the input file name
                input_file="${molecule}_hadamard_measurements_${nstep}_steps_${shots}_shots.npy"

                echo "Running with input_file=$input_file, nstep=$nstep, shots=$shots, samples=$samples, T=$formatted_T, and init_guess=$init"

                # Run the Python script
                python "$python_script" --file "$input_file" --shots "$shots" --samples "$samples" --init_guess "$init_guess"

                # Add a separator for readability
                echo "----------------------------------------"
            done
        done
    done
done

echo "Parameter sweep completed."

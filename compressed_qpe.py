import jax.numpy as jnp
import numpy as np
import jax
from jax import jit
jax.config.update("jax_enable_x64", True)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--shots", type=int, required = True)
parser.add_argument("--samples", type=int, required = True)
parser.add_argument("--file", type=str, required = True)

from ivdst import run_ivdst_from_file
from music import process_signal, estimate_frequencies_and_compute_error
from utils_ivdst import compute_cost, save_or_append_to_npy


args = parser.parse_args()

filename = args.file
shots = args.shots
samples = args.samples

true_freqs = [-7.880873948, -7.75139657, -7.32777980, -7.236480038]
guess_freq = [-7.9, -7.76, -7.32, -7.24]
reconstructed_signal, indices = run_ivdst_from_file(filename, guess_freq, samples = 20)
x_axis, y_axis = process_signal(reconstructed_signal, len(reconstructed_signal) // 2, 8, 4)

max_t, total_runtime, number_of_sampled_points = compute_cost(indices, shots, len(reconstructed_signal))

estimated_frequencies, error, average_error = estimate_frequencies_and_compute_error(y_axis, x_axis, true_freqs)

true_ee = true_freqs - np.min(true_freqs)
estimated_ee = estimated_frequencies - np.min(estimated_frequencies)

average_error_ee = np.mean(abs(np.array(true_ee) - np.array(estimated_ee)))

cf = number_of_sampled_points / (len(reconstructed_signal) // 2)

### format : max_t, total_runtime, error_gs, average_error, average_error_ee, nspinorb, cf, n_eigvals

result = [max_t, total_runtime, error[0], average_error, average_error_ee, 12, cf, 4]

# Create descriptive labels
labels = [
    "max_t", 
    "total_runtime", 
    "error_gs", 
    "average_error", 
    "average_error_ee", 
    "nspinorb", 
    "cf", 
    "n_eigvals"
]

# Print formatted output
print(f"{'Parameter':<20} {'Value':<20}", flush = True)
print("-" * 40)
for label, value in zip(labels, result):
    print(f"{label:<20} {value:<20}", flush= True)

save_or_append_to_npy("./lih_results.npy", result)
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
#from low_pass_filter import hsvd

#Example usage
#Define a vector of complex exponentials
# t = jnp.linspace(0, 2 * jnp.pi, 100)
# exponential_vector = 0.5 * jnp.exp(1j * t) + 0.5 * jnp.exp(1j * 2 * t)

#  # Create the Toeplitz matrix
# toeplitz_matrix = jax.scipy.linalg.toeplitz(exponential_vector)


# # start = time.time()

# V, D, T_reconstructed = toeplitz_vandermonde_decomposition(toeplitz_matrix)

# # end = time.time()

# # print(end - start)

# def is_vandermonde_correct(T, T_reconstructed, tol = 1e-6):

#     return jnp.allclose(T, T_reconstructed, atol = tol)


# print("Is Vandermonde decomposition correct : " + str(is_vandermonde_correct(toeplitz_matrix, T_reconstructed,  tol = 1e-8)))


###### now let's test on real data! ######

from ivdst import run_ivdst_from_file


molecule = 'lih'
samples = '120'
steps = '600'

filename = 'lih_hadamard_measurements_npcs_500_steps_5_shots_better_integration.npy'
#filename_noisy = 'lih_hadamard_measurements_npcs_320_steps_25_shots.npy'
# filename_noisy2 = 'lih_hadamard_measurements_cs_320_steps_125_shots.npy'

# z, T2, ni0, non_null_indices = initialization(molecule, samples, steps)

# print(jnp.linalg.cond(T2))

# print(jnp.linalg.cond(T2 + 1e-2*jnp.eye(len(z))))

# variables = (z, T2, ni0)

# mask = create_modified_identity_matrix(len(z), non_null_indices)
guess_freq = [-7.9, -7.7, -7.3, -7.2]
guess_freq = [element/8 for element in guess_freq]

start = time.time()
#print(test_cost(filename, guess_freq))
# grad = test_grad(filename, guess_freq, number_of_samples=20)
# print(grad)
# print(len(grad[0]))
# print(len(grad[1]))


# z, non_null_indices = load_data(filename)
# params_init = initialization_theoretical_guess(len(z), guess_freq)
# A = create_modified_identity_matrix(len(z), non_null_indices)

# b = A @ z
#reconstructed_signal = recover_signal(filename, guess_freq)

reconstructed_signal, indices = run_ivdst_from_file(filename, samples=10)
reconstructed_signal_gg, indices = run_ivdst_from_file(filename, guess_freq, samples=10)


from music import process_signal, estimate_frequencies_and_compute_error

# x_axis, y_axis = process_signal(reconstructed_signal, len(reconstructed_signal) // 2, 8, 4)
# x_axis_noisy, y_axis_noisy = process_signal(reconstructed_signal_noisy, len(reconstructed_signal_noisy) // 2, 8, 4)
# x_axis_noisy2, y_axis_noisy2 = process_signal(reconstructed_signal_noisy2, len(reconstructed_signal_noisy2) // 2, 8, 4)

# signal = np.load(filename, allow_pickle = True)

end = time.time()

x_axis, y_axis = process_signal(reconstructed_signal, len(reconstructed_signal) // 2, 8, 4)
x_axis_gg, y_axis_gg = process_signal(reconstructed_signal_gg, len(reconstructed_signal_gg) // 2, 8, 4)

true_freqs = [-7.880873948, -7.75139657, -7.32777980, -7.236480038]
estimated_frequencies, error, average_error = estimate_frequencies_and_compute_error(y_axis, x_axis, true_freqs)
estimated_frequencies_gg, error_gg, average_error_gg = estimate_frequencies_and_compute_error(y_axis_gg, x_axis_gg, true_freqs)

true_ee = true_freqs - np.min(true_freqs)
estimated_ee = estimated_frequencies - np.min(estimated_frequencies)
estimated_ee_gg = estimated_frequencies_gg - np.min(estimated_frequencies_gg)

average_error_ee = np.mean(abs(np.array(true_ee) - np.array(estimated_ee)))
average_error_ee_gg = np.mean(abs(np.array(true_ee) - np.array(estimated_ee_gg)))


print(estimated_frequencies)
print(error)
print(average_error)
print(average_error_ee)
print('-------------- gg -----------------')
print(estimated_frequencies_gg)
print(error_gg)
print(average_error_gg)
print(average_error_ee_gg)

print(end - start)

#np.save("ivdst_test", reconstructed_signal)

plt.plot(x_axis, y_axis)
plt.plot(x_axis_gg, y_axis_gg)

# plt.plot(reconstructed_signal, '*')
# plt.plot(np.load('lih_hadamard_measurements_npcs_600_steps_5_shots_better_integration.npy', allow_pickle = True), '--')
# plt.plot(np.load('lih_hadamard_measurements_npcs_600_steps_infinite_shots_better_integration.npy', allow_pickle = True))
# plt.plot(params_init[0])
#plt.scatter(np.arange(len(b)), b)

plt.show()
# test_evolution_cisd_init.py

import numpy as np
import jax.numpy as jnp
from utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nstep", type=int, required=True)
parser.add_argument("--molecule", type=str, required=True)

args = parser.parse_args()

import matplotlib.pyplot as plt

# Parameters
####### molecule dependent parameters
symbols  = ['Li', 'H']
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.66]])
nelecs = 4
nso = 12
####### calculation dependent parameters
nstep = args.nstep       # Number of time steps for evolution
dt = 0.005          # Time step size
total_shots = np.arange(5, 80, 15).tolist()

# Step 1: Generate Hamiltonian using PySCF 
hamiltonian = compute_hamiltonian_pyscf(geometry, symbols)
hamiltonian /= 8

# # Step 2: Generate initial state using CISD method (if needed casci approximate states are implemented, there you need to specify active space)
nroots = 10        # Example number of roots for CISD state calculation
nstates = [0, 1, 4, 5]     # Example list of states for CISD
initial_state = set_arbitrary_cisd_state(geometry, symbols, nroots, nstates, verbose=True)

input_state = jnp.zeros(2 ** nso)

for state in initial_state:
    input_state += state

input_state = input_state/jnp.linalg.norm(input_state)

# Step 3: Generate noisy autocorrelation signal using quantum evolution

times= jnp.arange(0, nstep, dt)
autocorr = compute_autocorrelation(input_state, hamiltonian, times, freq_shift = 0)



signal = equally_spaced_points(autocorr, nstep)[0]
times_spaced = equally_spaced_points(times, nstep)[0]
mirrored_signal = transform_signal(signal)
times_rev = reversed_axis(times_spaced)

np.save(args.molecule+"_hadamard_measurements_"+str(nstep)+"_steps_infinite_shots.npy", mirrored_signal)
for shots in total_shots:
    print("Sampling with # " + str(shots) + " shots:", flush = True)
    hadamard_signal_real = [sample_with_hadamard(jnp.real(mirrored_signal), shots)]
    hadamard_signal_imag = [sample_with_hadamard(jnp.imag(mirrored_signal), shots)]
    
    hadamard_signal =  np.asarray([hadamard_signal_real[element] + 1j * hadamard_signal_imag[element] for element in range(len(hadamard_signal_imag))])
    hadamard_signal = hadamard_signal.flatten()

    np.save(args.molecule+"_hadamard_measurements_"+str(nstep)+"_steps_"+str(shots)+"_shots.npy", hadamard_signal)


np.save("test_signal", mirrored_signal)

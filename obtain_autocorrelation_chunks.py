# test_evolution_cisd_init.py

import numpy as np
import jax.numpy as jnp
from utils import *
import numpy as onp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nstep", type=int, required=True)
parser.add_argument("--molecule", type=str, required=True)
parser.add_argument("--file", type=str, required = True)
parser.add_argument("--freqs", type=int, default=4)
parser.add_argument("--ndets", type=int, default=None)
args = parser.parse_args()


# Define distance r
r = 2.0  # Modify this to set the desired spacing between hydrogen atoms

# Parameters
####### Molecule-dependent parameters
nstep = args.nstep
symbols = ['H'] * 8  # H10 molecule
geometry = np.array([[0.0, 0.0, i * r] for i in range(8)])  # Linear chain along z-axis
nelecs = 8  # Number of electrons (assuming neutral H10)
nso = 16  # Number of spin orbitals (2 per H atom)

####### calculation dependent parameters
dt = 0.001          # Time step size
#total_shots = np.arange(5, 80, 15).tolist()
s = args.freqs
ndets=args.ndets
total_shots = [int(np.sqrt(nstep * s * np.log(s) * np.log(nstep)))]

initial_signal = np.load(args.file, allow_pickle = True)

nstep = len(initial_signal) // 2 - args.nstep

signal = initial_signal[nstep:-nstep]

onp.save(args.molecule+str(ndets)+"_hadamard_measurements_"+str(args.nstep)+"steps_infinite_shots.npy", signal)
for shots in total_shots:
    print("Sampling with # " + str(shots) + " shots:", flush = True)
    hadamard_signal_real = [sample_with_hadamard(onp.real(signal), shots)]
    hadamard_signal_imag = [sample_with_hadamard(onp.imag(signal), shots)]
    
    hadamard_signal =  onp.asarray([hadamard_signal_real[element] + 1j * hadamard_signal_imag[element] for element in range(len(hadamard_signal_imag))])
    hadamard_signal = hadamard_signal.flatten()
    onp.save(args.molecule+str(ndets)+"_hadamard_measurements_"+str(args.nstep)+"steps_"+str(shots)+"_shots.npy", hadamard_signal)




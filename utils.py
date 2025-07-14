#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:39:51 2024

@author: davide
"""


import jax.numpy as jnp
from jax import lax
import jax
import jax.scipy as jsp
import jax.random as jrandom
import pennylane as qml
import pennylane.numpy as np
import numpy as rnp
import time
from itertools import combinations
from pyscf import gto, scf, ao2mo
from overlapper.state import do_hf, do_cisd, cisd_state, do_casci, casci_state
from overlapper.utils import wf_budget
from numpy.linalg import eigh as onp_eigh
from jax.scipy.linalg import eigh  # For diagonalizing the Hamiltonian
from scipy.sparse.linalg import expm_multiply, eigsh
import gc
from numba import njit
import numpy as onp

###### set of functions to generate molecular hamiltonian #####

def format_atom_positions(atom_labels, atom_positions):
    # Check if both lists have the same length
    if len(atom_labels) != len(atom_positions):
        raise ValueError("The length of the atom labels list and the atom positions list must be the same.")
    
    formatted_string = ""
    for label, position in zip(atom_labels, atom_positions):
        formatted_string += f"{label} {position[0]} {position[1]} {position[2]};\n"
    
    return formatted_string


def compute_hamiltonian_pyscf_sparse(geometry, symbols, active_electrons = None, active_orbitals = None):
    geometry = geometry 
    
    formatted_string = format_atom_positions(symbols, geometry)
    
    mol = gto.M(atom = formatted_string)


    rhf = scf.RHF(mol)
    energy = rhf.kernel()

    one_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    two_ao = mol.intor('int2e_sph')
    one_mo = np.einsum('pi,pq,qj->ij', rhf.mo_coeff, one_ao, rhf.mo_coeff)
    two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff)
    two_mo = np.swapaxes(two_mo, 1, 3)

    core_constant = np.array([rhf.energy_nuc()])    #### here we could just call the qml function with also the active space part 
    if active_electrons != None:
        core, active = qml.qchem.active_space(
        mol.nelectron, mol.nao, mult, active_electrons, active_orbitals)

        for i in core:
            core_constant = core_constant + 2 * one_mo[i][i]
            for j in core:
                core_constant = core_constant + 2 * two_mo[i][j][j][i] - two_mo[i][j][i][j]

        for p in active:
            for q in active:
                for i in core:
                    one_mo[p, q] = one_mo[p, q] + (2 * two_mo[i][p][q][i] - two_mo[i][p][i][q])

    H_fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo)
        
    H = qml.qchem.qubit_observable(H_fermionic)

    hamiltonian_matrix = H.sparse_matrix().real
    return hamiltonian_matrix


def compute_init_state_and_hamiltonian(geometry, symbols, nroots, nstates, nelecas, ncas, ndet_budget=None, padding=True, verbose = True):
    geometry = geometry

    formatted_string = format_atom_positions(symbols, geometry)

    mol = gto.M(atom = formatted_string)


    rhf = scf.RHF(mol)
    energy = rhf.kernel()

    one_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    two_ao = mol.intor('int2e_sph')
    one_mo = np.einsum('pi,pq,qj->ij', rhf.mo_coeff, one_ao, rhf.mo_coeff)
    two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff)
    two_mo = np.swapaxes(two_mo, 1, 3)

    core_constant = np.array([rhf.energy_nuc()])    #### here we could just call the qml function with also the active space part 
    H_fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo)

    H = qml.qchem.qubit_observable(H_fermionic)

    hamiltonian_matrix = H.sparse_matrix().real.toarray()
    ############# this is done because we want to compute the initial state with the exact same orbitals with which we compute the hamiltonian matrix ####################
    mycisd, mycisd_e, mycisd_ss, mycisd_sz = do_casci(rhf, ncas, nelecas, nroots = nroots)

    if verbose:
        for k in range(len(mycisd_e)):
            print("State number : " + str(k) + " - Energy = " + str(mycisd_e[k]) + " - S^2 = " + str(mycisd_ss[k]), flush = True)

    if padding==True:
        nao = mycisd.mol.nao
        #from overlapper.state import casci_state_active as casci_state
    else:
        nao = int(ncas)
    mapped_cisd = []
    for state in nstates:
        if ndet_budget==None:
            wf_cisd = casci_state(mycisd, state = state, tol=1e-6)
            #print(wf_cisd)
        else:
            wf_cisd = casci_state(mycisd, state=state, tol=1e-6)
            wf_cisd = wf_budget(wf_cisd, ndet_budget)
        wf_cisd_reordered = _sign_chem_to_phys(wf_cisd, mycisd.mol.nao)

        wf = qml.qchem.convert._wfdict_to_statevector(wf_cisd_reordered, mycisd.mol.nao)

        mapped_cisd.append(wf)
    return mapped_cisd, hamiltonian_matrix


def compute_hamiltonian_pyscf(geometry, symbols, active_el = None, active_orbs = None):
    geometry = geometry 
    
    formatted_string = format_atom_positions(symbols, geometry)
    
    mol = gto.M(atom = formatted_string)

    if active_el == None and active_orbs == None:

        rhf = scf.RHF(mol)
        energy = rhf.kernel()

        one_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
        two_ao = mol.intor('int2e_sph')
        one_mo = np.einsum('pi,pq,qj->ij', rhf.mo_coeff, one_ao, rhf.mo_coeff)
        two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff)

        two_mo = np.swapaxes(two_mo, 1, 3)

        core_constant = np.array([rhf.energy_nuc()])    #### here we could just call the qml function with also the active space part 
    else:

        core_constant, one_mo, two_mo = qml.qchem.openfermion_pyscf._pyscf_integrals(symbols, geometry, active_electrons = active_el, active_orbitals = active_orbs)
        H_fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo)

    H_fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo)
        
    H = qml.qchem.qubit_observable(H_fermionic)

    hamiltonian_matrix = H.sparse_matrix().real.toarray()

    return jnp.array(hamiltonian_matrix)



def _sign_chem_to_phys(fcimatr_dict, norb):
    r"""Convert the dictionary-form wavefunction from chemist sign convention
    for ordering the creation operators by spin (i.e. all spin-up operators
    on the left) to the physicist convention native to PennyLane, which
    storing spin operators as interleaved for the same spatial orbital index.

    Note that convention change in the opposite direction -- starting from physicist
    and going to chemist -- can be accomplished with the same function
    (the sign transformation is reversible).

    Args:
        fcimatr_dict (dict[tuple(int, int), float]): dictionary of the form `{(int_a, int_b) :coeff}`, with integers `int_a, int_b`
        having binary represention corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations
        norb (int): total number of spatial orbitals of the underlying system

    Returns:
        signed_dict (dict): the same dictionary-type wavefunction with appropriate signs converted

    **Example**

    >>> fcimatr_dict = {(3, 1): 0.96, (6, 1): 0.1, \
                        (3, 4): 0.1, (6, 4): 0.14, (5, 2): 0.19}
    >>> _sign_chem_to_phys(fcimatr_dict, 3)
    {(3, 1): -0.96, (6, 1): 0.1, (3, 4): 0.1, (6, 4): 0.14, (5, 2): -0.19}
    """

    signed_dict = {}
    for key, elem in fcimatr_dict.items():
        lsta, lstb = bin(key[0])[2:][::-1], bin(key[1])[2:][::-1]
        # highest energy state is on the right -- pad to the right
        lsta = np.array([int(elem) for elem in lsta] + [0] * (norb - len(lsta)))
        lstb = np.array([int(elem) for elem in lstb] + [0] * (norb - len(lstb)))
        which_occ = np.where(lsta == 1)[0]
        parity = (-1) ** np.sum([np.sum(lstb[: int(ind)]) for ind in which_occ])
        signed_dict[key] = parity * elem
    return signed_dict

def compute_hamiltonian(geometry, symbols, method = "dhf", active_electrons = None, active_space = None):   ##### this requires geometry input in Bohr units #####
    geometry *= 1.88973
    mol = qml.qchem.Molecule(symbols, geometry)
    hamiltonian = qml.qchem.molecular_hamiltonian(symbols, geometry, method = method, active_electrons = active_electrons, active_orbitals = active_space)[0]
    hamiltonian_matrix = hamiltonian.sparse_matrix().real.toarray()
    
    return jnp.array(hamiltonian_matrix)


###### set of functions to generate input state #######

def set_arbitrary_cisd_state(geometry, symbols, nroots, nstates, verbose = True):
    combined_list = format_atom_positions(symbols, geometry)
    mol = gto.M(atom = combined_list)

    hf, hf_e, hf_ss, hf_sz = do_hf(mol, "rhf")

    mycisd, mycisd_e, mycisd_ss, mycisd_sz = do_cisd(hf, nroots=nroots)

    if verbose:
        for k in range(len(mycisd_e)):
            print("State number : " + str(k) + " - Energy = " + str(mycisd_e[k]) + " - S^2 = " + str(mycisd_ss[k]), flush = True)
    
    mapped_cisd = []
    for state in nstates:
        wf_cisd = cisd_state(mycisd, state = state, tol=1e-5)

        wf_cisd_reordered = _sign_chem_to_phys(wf_cisd, mycisd.mol.nao)

        wf = qml.qchem.convert._wfdict_to_statevector(wf_cisd_reordered, mycisd.mol.nao)

        mapped_cisd.append(wf)
    return mapped_cisd


def set_arbitrary_casci_state(geometry, symbols, nroots, nstates, nelecas, ncas, ndet=None, verbose = True):
    combined_list = format_atom_positions(symbols, geometry)
    mol = gto.M(atom = combined_list)

    hf, hf_e, hf_ss, hf_sz = do_hf(mol, "rhf")

    mycisd, mycisd_e, mycisd_ss, mycisd_sz = do_casci(hf, ncas, nelecas, nroots = nroots)

    if verbose:
        for k in range(len(mycisd_e)):
            print("State number : " + str(k) + " - Energy = " + str(mycisd_e[k]) + " - S^2 = " + str(mycisd_ss[k]), flush = True)
    
    mapped_cisd = []
    for state in nstates:
        wf_cisd = casci_state(mycisd, state = state, tol=0.1)
        if ndet != None:
            wf_cisd = wf_budget(wf_cisd, ndet)

        wf_cisd_reordered = _sign_chem_to_phys(wf_cisd, mycisd.mol.nao)

        wf = qml.qchem.convert._wfdict_to_statevector(wf_cisd_reordered, mycisd.mol.nao)

        mapped_cisd.append(wf)
    return mapped_cisd

def compare_input_quality(geometry, symbols, nroots, nstates_casci, nstates_cisd, nelecas, ncas, verbose = True):
    combined_list = format_atom_positions(symbols, geometry)
    mol = gto.M(atom = combined_list)

    #mol.spin = 0

    hf, hf_e, hf_ss, hf_sz = do_hf(mol, "rhf")

    hf.mol.spin = 0

    mycisd, mycisd_e, mycisd_ss, mycisd_sz = do_cisd(hf, nroots=nroots)
    mycasci, mycasci_e, mycasci_ss, mycasci_sz = do_casci(hf, ncas, nelecas, nroots=nroots)

    if verbose:
        for k in range(len(mycisd_e)):
            print("State number : " + str(k) + " - Energy = " + str(mycisd_e[k]) + " - S^2 = " + str(mycisd_ss[k]), flush = True)
            print("State number : " + str(k) + " - Energy = " + str(mycasci_e[k]) + " - S^2 = " + str(mycasci_ss[k]), flush = True)
            print("--------------------------------------------------------------------")
    j = 0
    for state_casci, state_cisd in zip(nstates_casci, nstates_cisd):
        wf_cisd = casci_state(mycasci, state = state_casci, tol=0.0001)

        wf_cisd_reordered = _sign_chem_to_phys(wf_cisd, mycisd.mol.nao)

        wf_exact = qml.qchem.convert._wfdict_to_statevector(wf_cisd_reordered, mycisd.mol.nao)

        wf_cisd = cisd_state(mycisd, state = state_cisd, tol=0.1)

        wf_cisd_reordered = _sign_chem_to_phys(wf_cisd, mycisd.mol.nao)

        wf = qml.qchem.convert._wfdict_to_statevector(wf_cisd_reordered, mycisd.mol.nao)

        print("Candidate input state for level : " + str(j) + " has overlap of " + str(np.dot(np.transpose(wf), wf_exact)), flush = True)
        j += 1
    return None


def set_input_hf_state(nelecs, nso):
    hf_state = qml.qchem.hf_state(nelecs, nso)
    qml.BasisState(np.array(hf_state), wires=range(nso))
    
    dev = qml.device("default.qubit", wires=nso)
    
    @qml.qnode(dev)
    def circuit():
        qml.BasisState(np.array(hf_state), wires=range(nso))
        return qml.state()

    return circuit()

def set_doubly_excited_state(nelecs, nso, excitations, phi = np.pi):
    hf_state = qml.qchem.hf_state(nelecs, nso)
    qml.BasisState(np.array(hf_state), wires=range(nso))

    dev = qml.device("default.qubit", wires=nso)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(np.array(hf_state), wires = range(nso))
        qml.DoubleExcitation(phi, wires = excitations)
        return qml.state()

    return circuit()

def set_singly_excited_state(nelecs, nso, excitations, phi = np.pi):
    hf_state = qml.qchem.hf_state(nelecs, nso)
    qml.BasisState(np.array(hf_state), wires=range(nso))

    dev = qml.device("default.qubit", wires=nso)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(np.array(hf_state), wires = range(nso))
        qml.SingleExcitation(phi, wires = excitations)
        return qml.state()

    return circuit()




###### set of functions to generate unitary evolution ######

@njit
def check_spin_multiplicity(ones_positions, spin_multiplicities):
    """
    Checks if a given set of '1' positions satisfies any of the allowed spin multiplicities.
    
    Args:
        ones_positions: Tuple of positions where '1's are located.
        spin_multiplicities: NumPy array of allowed total Sz values.

    Returns:
        True if the constraint is satisfied, otherwise False.
    """
    sz = 0.0
    for pos in ones_positions:
        sz += 0.5 if pos % 2 == 0 else -0.5  # Compute Sz based on position parity
    
    # Explicit loop to replace `sz in spin_multiplicities`
    for s in spin_multiplicities:
        if int(sz) == s:
            return True
    return False

@njit
def bitwise_index(ones_positions):
    """
    Converts a tuple of '1' positions to an integer representation.
    """
    index = 0
    for pos in ones_positions:
        index |= 1 << pos  # Set bit at position 'pos'
    return index

def get_indices_by_hamming_weight_and_spin(n, w, spin_multiplicities):
    """
    Returns a sorted list of indices corresponding to binary strings of length n 
    that have exactly w ones and satisfy any allowed spin multiplicity.

    Args:
        n: Length of binary string.
        w: Hamming weight (number of 1s).
        spin_multiplicities: List or array of allowed total Sz values.

    Returns:
        A NumPy array of valid indices.
    """
    spin_multiplicities = np.array(spin_multiplicities, dtype=np.int32)  # Ensure it's a NumPy array

    valid_indices = [
        bitwise_index(ones_positions)
        for ones_positions in combinations(range(n), w)
        if check_spin_multiplicity(ones_positions, spin_multiplicities)
    ]
    
    return np.array(valid_indices, dtype=np.int64)

def get_indices_by_hamming_weight(n, w):
    """
    Returns the indices (as integers) of all binary strings of length n with Hamming weight w.
    """
    return np.array([sum(1 << i for i in ones) for ones in combinations(range(n), w)])

def extend_eigenvectors(eigenvectors, selected_indices, full_size):
    """
    Expands eigenvectors from the subspace back to the full space by zero-padding.
    """
    k = eigenvectors.shape[1]  # Number of eigenvectors
    extended_vectors = np.zeros((full_size, k), dtype=np.complex128)

    # Place each component in the correct location
    for new_index, old_index in enumerate(selected_indices):
        extended_vectors[old_index, :] = eigenvectors[new_index, :]

    return extended_vectors


def compute_autocorrelation_sparse4(psi_0, H, times, n, w, spin_multiplicity=None, hbar=1.0, freq_shift=0, k_eigen=1000):
    """
    Computes the autocorrelation function C(t) = <psi_0 | psi(t)> using sparse matrix diagonalization,
    restricting the Hamiltonian to a subspace with a given Hamming weight and spin multiplicity.
    
    Args:
        psi_0: Initial wavefunction as a 1D complex NumPy array.
        H: Time-independent sparse Hamiltonian (scipy.sparse matrix).
        times: 1D array of times at which to compute the autocorrelation function.
        n: Length of binary strings (size of Hilbert space is 2^n).
        w: Hamming weight for subspace selection.
        spin_multiplicity: Desired total Sz value.
        hbar: Reduced Planck constant (default is 1.0, natural units).
        freq_shift: Frequency shift to adjust the eigenvalues.
        k_eigen: Number of eigenvalues to compute (None means full diagonalization, otherwise uses eigsh).

    Returns:
        A 1D NumPy array representing the autocorrelation function at each time.
    """
    # Step 1: Extract submatrix corresponding to the given Hamming weight & spin multiplicity
    if spin_multiplicity != None:
        selected_indices = get_indices_by_hamming_weight_and_spin(n, w, spin_multiplicity)
    else:
        selected_indices = get_indices_by_hamming_weight(n, w)
    print(len(selected_indices), flush = True)
    submatrix = H[selected_indices, :][:, selected_indices].toarray()

    # Step 2: Compute eigenvalues & eigenvectors in the subspace
    #num_eigen = min(k_eigen, submatrix.shape[0])  # Ensure k_eigen is not larger than the submatrix
    eigenvalues, sub_eigenvectors = onp_eigh(submatrix)  # Smallest eigenvalues

    # Step 3: Extend eigenvectors back to full space
    #eigenvectors = extend_eigenvectors(sub_eigenvectors, selected_indices, H.shape[0])

    # Step 4: Apply frequency shift
    eigenvalues += freq_shift
    psi_0_sel = onp.array(psi_0[selected_indices])
    # Step 5: Transform initial state into eigenbasis
    psi_0_in_eigenbasis = onp.array(sub_eigenvectors.T.conj() @ psi_0_sel)

    # Compute |c_i|^2 for each eigencomponent
    coefficients_squared = onp.abs(psi_0_in_eigenbasis) ** 2

    # Define the autocorrelation function
    def autocorrelation(t):
        evolution_factors = jnp.exp(-1j * eigenvalues * t / hbar)
        return jnp.sum(coefficients_squared * evolution_factors)

    # Use JAX lax.scan for efficient computation over time
    def scan_fn(carry, t):
        autocorr_val = autocorrelation(t)
        return carry, autocorr_val

    _, autocorrelation_values = jax.lax.scan(scan_fn, None, times)
    return autocorrelation_values

def compute_autocorrelation_sparse3(psi_0, H, times, n, w, spin_multiplicity, hbar=1.0, freq_shift=0, k_eigen=1000):
    """
    Computes the autocorrelation function C(t) = <psi_0 | psi(t)> using sparse matrix diagonalization,
    restricting the Hamiltonian to a subspace with a given Hamming weight and spin multiplicity.
    
    Args:
        psi_0: Initial wavefunction as a 1D complex NumPy array.
        H: Time-independent sparse Hamiltonian (scipy.sparse matrix).
        times: 1D array of times at which to compute the autocorrelation function.
        n: Length of binary strings (size of Hilbert space is 2^n).
        w: Hamming weight for subspace selection.
        spin_multiplicity: Desired total Sz value.
        hbar: Reduced Planck constant (default is 1.0, natural units).
        freq_shift: Frequency shift to adjust the eigenvalues.
        k_eigen: Number of eigenvalues to compute (None means full diagonalization, otherwise uses eigsh).

    Returns:
        A 1D NumPy array representing the autocorrelation function at each time.
    """
    # Step 1: Extract submatrix corresponding to the given Hamming weight & spin multiplicity
    selected_indices = get_indices_by_hamming_weight_and_spin(n, w, spin_multiplicity)
    print(len(selected_indices))
    submatrix = H[selected_indices, :][:, selected_indices]

    # Step 2: Compute eigenvalues & eigenvectors in the subspace
    num_eigen = min(k_eigen, submatrix.shape[0])  # Ensure k_eigen is not larger than the submatrix
    eigenvalues, sub_eigenvectors = eigsh(submatrix, k=num_eigen, which='SA')  # Smallest eigenvalues

    # Step 3: Extend eigenvectors back to full space
    eigenvectors = extend_eigenvectors(sub_eigenvectors, selected_indices, H.shape[0])

    # Step 4: Apply frequency shift
    eigenvalues += freq_shift

    # Step 5: Transform initial state into eigenbasis
    psi_0_in_eigenbasis = eigenvectors.T.conj() @ psi_0

    # Compute |c_i|^2 for each eigencomponent
    coefficients_squared = np.abs(psi_0_in_eigenbasis) ** 2

    # Define the autocorrelation function
    def autocorrelation(t):
        evolution_factors = jnp.exp(-1j * eigenvalues * t / hbar)
        return jnp.sum(coefficients_squared * evolution_factors)

    # Use JAX lax.scan for efficient computation over time
    def scan_fn(carry, t):
        autocorr_val = autocorrelation(t)
        return carry, autocorr_val

    _, autocorrelation_values = jax.lax.scan(scan_fn, None, times)
    return autocorrelation_values

def compute_autocorrelation_sparse2(psi_0, H, times, n, w, hbar=1.0, freq_shift=0, k_eigen=400):
    """
    Computes the autocorrelation function C(t) = <psi_0 | psi(t)> using sparse matrix diagonalization,
    restricting the Hamiltonian to a subspace with a given Hamming weight.
    
    Args:
        psi_0: Initial wavefunction as a 1D complex NumPy array.
        H: Time-independent sparse Hamiltonian (scipy.sparse matrix).
        times: 1D array of times at which to compute the autocorrelation function.
        n: Length of binary strings (size of Hilbert space is 2^n).
        w: Hamming weight for subspace selection.
        hbar: Reduced Planck constant (default is 1.0, natural units).
        freq_shift: Frequency shift to adjust the eigenvalues.
        k_eigen: Number of eigenvalues to compute (None means full diagonalization, otherwise uses eigsh).

    Returns:
        A 1D NumPy array representing the autocorrelation function at each time.
    """
    # Step 1: Extract submatrix corresponding to the given Hamming weight
    selected_indices = get_indices_by_hamming_weight(n, w)
    submatrix = H[selected_indices, :][:, selected_indices]

    # Step 2: Compute eigenvalues & eigenvectors in the subspace
    num_eigen = min(k_eigen, submatrix.shape[0])  # Ensure k_eigen is not larger than the submatrix
    eigenvalues, sub_eigenvectors = eigsh(submatrix, k=num_eigen, which='SA')  # Smallest eigenvalues
    #print(eigenvalues[0])

    # Step 3: Extend eigenvectors back to full space
    eigenvectors = extend_eigenvectors(sub_eigenvectors, selected_indices, H.shape[0])

    # Step 4: Apply frequency shift
    eigenvalues += freq_shift

    # Step 5: Transform initial state into eigenbasis
    psi_0_in_eigenbasis = eigenvectors.T.conj() @ psi_0

    # Compute |c_i|^2 for each eigencomponent
    coefficients_squared = np.abs(psi_0_in_eigenbasis) ** 2

    # Define the autocorrelation function
    def autocorrelation(t):
        evolution_factors = jnp.exp(-1j * eigenvalues * t / hbar)
        return jnp.sum(coefficients_squared * evolution_factors)

    # Use JAX lax.scan for efficient computation over time
    def scan_fn(carry, t):
        autocorr_val = autocorrelation(t)
        return carry, autocorr_val

    _, autocorrelation_values = jax.lax.scan(scan_fn, None, times)
    return autocorrelation_values


def compute_autocorrelation_sparse(psi_0, H, times, hbar=1.0, freq_shift=0, k_eigen=400):
    """
    Computes the autocorrelation function C(t) = <psi_0 | psi(t)> using sparse matrix diagonalization.

    Args:
        psi_0: Initial wavefunction as a 1D complex NumPy array.
        H: Time-independent sparse Hamiltonian (scipy.sparse matrix).
        times: 1D array of times at which to compute the autocorrelation function.
        hbar: Reduced Planck constant (default is 1.0, natural units).
        freq_shift: Frequency shift to adjust the eigenvalues.
        k_eigen: Number of eigenvalues to compute (None means full diagonalization, otherwise uses eigsh).

    Returns:
        A 1D NumPy array representing the autocorrelation function at each time.
    """
    # Compute eigenvalues & eigenvectors using sparse diagonalization
    if k_eigen is None or k_eigen >= H.shape[0]:  # Full diagonalization if needed
        eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())  # Convert to dense for full diag
    else:
        eigenvalues, eigenvectors = eigsh(H, k=k_eigen, which='SA')  # Smallest eigenvalues
    print(eigenvalues[0])
    # Apply frequency shift
    eigenvalues += freq_shift

    # Transform initial state into eigenbasis
    psi_0_in_eigenbasis = eigenvectors.T.conj() @ psi_0

    # Compute |c_i|^2 for each eigencomponent
    coefficients_squared = np.abs(psi_0_in_eigenbasis) ** 2

    def autocorrelation(t):
        evolution_factors = jnp.exp(-1j * eigenvalues * t / hbar)
        return jnp.sum(coefficients_squared * evolution_factors)

    autocorrelation_values = []

    for t in times:
        autocorrelation_values.append(autocorrelation(t))
    return autocorrelation_values

def compute_autocorrelation_low_memory_sparse(psi_0, H, times, hbar=1.0):
    
    psi_0_np = np.asarray(psi_0)
    results = []
    def autocorrelation_numpy(t):
        ham = H.multiply(-1j * (t / hbar))
        psi_t = expm_multiply(ham, psi_0_np)
        return jnp.vdot(psi_0, np.array(psi_t))  # Convert back to JAX
    for t in times:
        results.append(autocorrelation_numpy(t))
        gc.collect()
    return jnp.array(results)


def compute_autocorrelation_low_memory(psi_0, H, times, hbar=1.0, freq_shift=8):
    """
    Computes the autocorrelation function C(t) = <psi_0 | psi(t)> for a wavefunction
    under a time-independent Hamiltonian.
    
    Args:
        psi_0: Initial wavefunction as a 1D complex JAX array (in the original basis).
        H: Time-independent Hamiltonian as a 2D Hermitian JAX array.
        times: 1D array of times at which to compute the autocorrelation function.
        hbar: Reduced Planck constant (default is 1.0, natural units).
        freq_shift: Frequency shift applied to the eigenvalues (default is 8).
        
    Returns:
        A 1D JAX array representing the autocorrelation function at each time.
    """

    # Diagonalize the Hamiltonian: H = V * Lambda * V^-1
    eigenvalues, eigenvectors = eigh(H)  # eigenvalues and eigenvectors of H
    eigenvalues = eigenvalues + freq_shift

    # Represent the initial state in the eigenbasis: psi_0' = V^-1 * psi_0
    psi_0_in_eigenbasis = eigenvectors.T.conj() @ psi_0

    # Compute |c_i|^2 for each component in the eigenbasis
    coefficients_squared = jnp.abs(psi_0_in_eigenbasis) ** 2

    # Define the autocorrelation function for a single time.
    def autocorrelation(t):
        evolution_factors = jnp.exp(-1j * eigenvalues * t / hbar)
        return jnp.sum(coefficients_squared * evolution_factors)

    # Use lax.scan to iterate over the times.
    # Here, the "carry" is unused (set to None), and we collect the autocorrelation at each time.
    def scan_fn(carry, t):
        autocorr_val = autocorrelation(t)
        return carry, autocorr_val

    # The scan returns a tuple (carry, results). We only need the results.
    _, autocorrelation_values = jax.lax.scan(scan_fn, None, times)
    return autocorrelation_values



def compute_autocorrelation(psi_0, H, times, hbar=1.0, freq_shift = 8):
    """
    Computes the autocorrelation function C(t) = <psi_0 | psi(t)> for a wavefunction
    under a time-independent Hamiltonian.
    
    Args:
        psi_0: Initial wavefunction as a 1D complex JAX array (in the original basis).
        H: Time-independent Hamiltonian as a 2D Hermitian JAX array.
        times: 1D array of times at which to compute the autocorrelation function.
        hbar: Reduced Planck constant (default is 1.0, natural units).
        
    Returns:
        A 1D JAX array representing the autocorrelation function at each time.
    """
    
    # Diagonalize the Hamiltonian: H = V * Lambda * V^-1
    eigenvalues, eigenvectors = onp_eigh(H)  # eigenvalues and eigenvectors of H
    eigenvalues = eigenvalues + freq_shift
    
    # Represent the initial state in the eigenbasis: psi_0' = V^-1 * psi_0
    psi_0_in_eigenbasis = eigenvectors.T.conj() @ psi_0  # Coefficients in eigenbasis
    
    # Compute |c_i|^2 for each component in the eigenbasis
    coefficients_squared = onp.abs(psi_0_in_eigenbasis) ** 2
    
    # Compute the autocorrelation function C(t)
#    def autocorrelation(t):
#        evolution_factors = onp.exp(-1j * eigenvalues * t / hbar)
#        return onp.sum(coefficients_squared * evolution_factors)

    def autocorrelation(t):
        evolution_factors = jnp.exp(-1j * eigenvalues * t / hbar)
        return jnp.sum(coefficients_squared * evolution_factors)

    # Use JAX lax.scan for efficient computation over time
    def scan_fn(carry, t):
        autocorr_val = autocorrelation(t)
        return carry, autocorr_val

    _, autocorrelation_values = jax.lax.scan(scan_fn, None, times)


    # Vectorize over the times
#    autocorrelation = np.array([autocorrelation(t) for t in times])
    #autocorrelation_vec = jax.vmap(autocorrelation)
    #autocorrelation = autocorrelation_vec(times)
    #autocorrelation_np = onp.asarray(autocorrelation)
    autocorr = onp.array(autocorrelation_values)
    return autocorr


def compute_evolution2(nstep, hamiltonian, initial_state, dt):

    def body(carry, j):
        
        
        step_forward = jsp.linalg.expm(-1j * hamiltonian * dt)
        carry = jnp.dot(carry, step_forward)
        psi_t = carry @ initial_state
        autocorr = jnp.dot(jnp.conj(jnp.transpose(initial_state)), psi_t)
        

        return carry, autocorr

    # Initialize total_evolution matrix
    total_evolution = jnp.eye(jnp.shape(hamiltonian)[0], dtype = jnp.complex128)

    # Create a sequence of step indices
    step_indices = jnp.arange(nstep)

    # Use lax.scan to perform the loop
    total_evolution, autocorr = lax.scan(body, total_evolution, step_indices)
    
    return total_evolution, autocorr


def Hamadard_test(p,M):

    val = 0
    for m in range(M):
        r = rnp.random.uniform(0,1)
        if(r < p):
            val = val + 1
        else:
            val = val - 1

    return val/M


### the following one can be jitted

def sample_with_hadamard(signal, M):
    hadamard_sampled_signal = jnp.zeros(len(signal))
    k = 0
    for signal_temp in signal:
        real = signal_temp.real 
        imag = signal_temp.imag
        real2 = Hamadard_test(0.5*(1+real),M)
        imag2 = Hamadard_test(0.5*(1+imag),M)
        hadamard_sampled_signal = hadamard_sampled_signal.at[k].set(real2 + 1j * imag2)
        k += 1
    return hadamard_sampled_signal

def randomly_select_points(signal, num_points):
    """
    Randomly selects a specified number of points from the given signal.

    Args:
    - signal (array_like): The input signal.
    - num_points (int): The number of points to select.

    Returns:
    - selected_points (array_like): The selected points from the signal.
    - selected_indices (array_like): The indices of the selected points.
    """
    signal_length = len(signal)
    key = jrandom.PRNGKey(int(time.time()))
    selected_indices = jrandom.choice(key, signal_length, (num_points, ), replace=False)
    selected_points = signal[selected_indices]
    return selected_points, selected_indices

def extract_points(vector, indices):
    """
    Extracts the values of the vector corresponding to the specified indices.

    Args:
    - vector (array_like): The input vector.
    - indices (array_like): The indices of the points to extract.

    Returns:
    - extracted_points (array_like): The values of the vector corresponding to the specified indices.
    """
    return vector[indices]


def extract_chunk(vec, N, M):
    """
    Extracts the chunk of `vec` that ranges from -M to M.

    Args:
        vec: 1D JAX array with values from -2N to 2N.
        N: The original range limit (i.e., vec spans -2N to 2N).
        M: The desired range limit (M < 2N).

    Returns:
        A JAX array containing elements from -M to M.
    """
    assert M < 2*N, "M must be smaller than 2N"
    
    mask = jnp.logical_and(vec >= -M, vec <= M)  # Create a boolean mask
    return vec[mask]  # Extract the values



def equally_spaced_points(points, M):
    """
    Generate `M` equally spaced points from the input `points`.

    Args:
    - points: array-like, shape (K,)
        Input vector of `K` points.
    - M: int
        Number of equally spaced points to generate (M < K).

    Returns:
    - out_points: array-like, shape (M,)
        Vector of `M` equally spaced points.
    """
    K = points.shape[0]
    step = (K) / (M)
    indices = jnp.arange(0, K, step)
    return jnp.take(points, indices.astype(int)), indices

def evenly_spaced_elements(arr, x):
    """
    Returns x elements evenly spaced from the input numpy array arr.
    
    Parameters:
    arr (numpy array): Input array from which to select elements.
    x (int): Number of evenly spaced elements to select.
    
    Returns:
    numpy array: Numpy array of x evenly spaced elements from arr.
    """
    if x <= 0:
        return np.array([])
    
    if x > len(arr):
        raise ValueError("Number of elements requested is greater than the length of the input array.")
    
    indices = np.linspace(0, len(arr) - 1, x, dtype=int)
    return arr[indices]


def transform_signal(signal):
    new_signal = onp.hstack((onp.conj(signal[::-1]), signal[1:]))
    # conj_signal = jnp.conj(signal)
    # middle_value = jnp.array([1.0 + 0.0j])  # Ensure the middle value is complex
    # new_signal = jnp.concatenate((conj_signal, middle_value, signal))
    return new_signal

def mirror_axis(signal):
    new_signal = onp.hstack((-onp.conj(signal[::-1]), signal[1:]))
    # conj_signal = jnp.conj(signal)
    # middle_value = jnp.array([1.0 + 0.0j])  # Ensure the middle value is complex
    # new_signal = jnp.concatenate((conj_signal, middle_value, signal))
    return new_signal

def reversed_axis(signal):
    rev_sign = onp.hstack((-onp.conj(signal[::-1]), signal[1:]))
    return rev_sign

###### set of functions to compute energy ######

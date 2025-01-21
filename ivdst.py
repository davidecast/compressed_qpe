import numpy as np
import jax.numpy as jnp
import jax
#from jax.scipy.linalg import eigh
from scipy.linalg import eigh

from jax import jit
jax.config.update("jax_enable_x64", True)
import numba as nb
from utils_ivdst import guess_autocorr, randomly_select_points


def load_data_old(molecule, samples, steps):
    """
    Load the signal vector and the indices of non-null values.
    """
    z = np.load(f"{molecule}hadamard_signal_cf0.3{samples}_{steps}steps.npy", allow_pickle=True)[0]  
    non_null_indices = np.real(np.load(f"{molecule}hadamard_signal_random_measurement_cf0.3_{steps}steps.npy")[0][1])
    return z, non_null_indices

def load_data(file_name, samples = 50):
    z = np.load(file_name)
    _, non_null_indices = randomly_select_points(z, samples)
    return z, non_null_indices



def create_modified_identity_matrix(size, zero_indexes):
    """
    Create an identity matrix with selected diagonal elements set to 0.
    """
    I = np.zeros((size, size))
    I[zero_indexes, zero_indexes] = 1
    return I


# Define a helper function to process one diagonal
def process_diagonal(first_column, first_row, index, matrix):

    first_column = jnp.append(first_column, jnp.mean(extract_diagonal(matrix, k =  -abs(index))))
    first_row = jnp.append(first_row, jnp.mean(extract_diagonal(matrix, k =  index)))

    return first_column, first_row

@nb.jit
def compute(mat, n):
    #n = mat.shape[0]
    output = np.zeros(n*2-1, dtype=np.complex64)
    for i in range(n-1, -1, -1):
        output[i:i+n] += mat[n-1-i]
    output[0:n] /= np.arange(1, n+1, 1, dtype=np.complex64)
    output[n:]  /= np.arange(n-1, 0, -1, dtype=np.complex64)
    return output

def make_toeplitz_matrix(matrix):
    # Get the shape of the input matrix
    n, m = matrix.shape
    
    row_and_column = compute(matrix, n)

    first_column = jnp.flip(row_and_column[:n])
    first_row = row_and_column[n-1:]

    toeplitz_matrix = jax.scipy.linalg.toeplitz(first_column, first_row)
    
    return toeplitz_matrix


def toeplitz_to_hankel(toeplitz_matrix):

    hankel_matrix = jnp.flipud(toeplitz_matrix)
    
    return hankel_matrix

def initialization_from_file(filename, samples = 20):
    z, non_null_indices = load_data(filename, samples)
    mask = create_modified_identity_matrix(len(z), non_null_indices)
    masked_z = jnp.dot(jnp.transpose(mask), z)

    r0 = jnp.outer(masked_z, jnp.conj(masked_z))

    T0 = make_toeplitz_matrix(r0)

    ni0 = jnp.trace(T0) / jnp.shape(r0)[0]

    return z, T0, ni0, non_null_indices

def initialization_theoretical_guess_from_file(filename, guess_freq, samples = 20):  #### here we have to ensure that we are taking the gradients w.r.t. measurements only
    z, non_null_indices = load_data(filename, samples)

    artificial_z = guess_autocorr(guess_freq, len(z))

    r0 = jnp.outer(artificial_z, jnp.conj(artificial_z))

    T0 = make_toeplitz_matrix(r0)

    ni0 = jnp.trace(T0) / jnp.shape(r0)[0]
    return z, T0, ni0, non_null_indices

def initialization(molecule, samples, steps):
    z, non_null_indices = load_data(molecule, samples, steps)
    r0 = jnp.outer(z, jnp.conj(z))

    T0 = make_toeplitz_matrix(r0)

    ni0 = jnp.trace(T0) / jnp.shape(r0)[0]

    return z, T0, ni0, non_null_indices

def initialization_theoretical_guess(molecule, samples, steps, guess_freq):
    z, non_null_indices = load_data(molecule, samples, steps)

    artificial_z = guess_autocorr(guess_freq, len(z))

    r0 = jnp.outer(artificial_z, jnp.conj(artificial_z))

    T0 = make_toeplitz_matrix(r0)

    ni0 = jnp.trace(T0) / jnp.shape(r0)[0]
    return z, T0, ni0, non_null_indices


def one_step_iteration(measurements, mask, variables, previous_variables, iteration_numb, last_momentum_coeff, step_size, lam_threshold):

    mod_variables, old_momentum_coeff = apply_momentum(variables, previous_variables, iteration_numb, last_momentum_coeff)

    descended_variables = apply_gradient_descent(mod_variables, measurements, mask, step_size)

    new_variables = apply_proximal_mapping(descended_variables, lam_threshold)

    convergence, new_iteration_numb  = check_convergence(new_variables[1], variables[1], iteration_numb)
    
    return new_variables, variables, new_iteration_numb, old_momentum_coeff, convergence


def apply_momentum(variables, previous_variables, iteration_numb, last_momentum_coeff):

    if iteration_numb != 0:

        new_momentum_coeff = 1 + jnp.sqrt(4 * jnp.power(last_momentum_coeff, 2) + 1) * 0.5

        for element1, element2 in zip(variables, previous_variables):
            element1 = element1 + (element1 - element2) * ( (last_momentum_coeff - 1) / new_momentum_coeff )
        
        new_variables = variables

    else:

        new_variables = variables

        new_momentum_coeff = 1 + jnp.sqrt(5) * 0.5

    return new_variables, new_momentum_coeff


@jit
def toeplitz_vandermonde_decomposition(toeplitz_matrix, rcond = 1e-3): ### this is working only for symmetric toeplitz matrices, ours are hermitian :( !!!! It is now working, before we were facing numerical stability issues. 

    n = jnp.shape(toeplitz_matrix)[0]

    v = jnp.ones(n, dtype='complex128')

    T_regularized = toeplitz_matrix #+ 0.0175 * jnp.eye(n)  ## in the original algorithm there is no regularization here
    
    a = jnp.linalg.solve(T_regularized, v) 
    #a = jax.scipy.linalg.solve(T_regularized, v, assume_a = "her") #### not clear which one is the fastest option

    vander_coeff = jnp.roots(a, strip_zeros = False) ### strip_zeros = False for jitting-compatibility

    vander_coeff = jnp.append(jnp.array([1]), vander_coeff)

    V = jnp.vander(vander_coeff, increasing = True)

    D = jnp.linalg.inv(jnp.conj(jnp.transpose(V))) @ toeplitz_matrix @ jnp.linalg.inv(V)

    #reconstruction = jnp.conj(V.T) @ D @ V 

    return V, D#, reconstruction


@jit
def build_Z_matrix(variables):

    r, T, t_reshaped = variables

    r_reshaped = jnp.reshape(r, (len(r), 1))
    r_H_reshaped = jnp.transpose(jnp.append(jnp.conj(r_reshaped), t_reshaped))

    first_row = jnp.hstack([T, r_reshaped])
    #second_row = jnp.hstack([r_H_reshaped, t_reshaped])
    second_row = r_H_reshaped

    Z = jnp.vstack([first_row, second_row])

    return Z

@jit
def apply_gradient_descent(variables, z, P, step_size = 5e-3):
    
    signal, T_matrix, scalar_ni = variables
    measurements = jnp.dot(P, z)
    updated_signal = signal - step_size * jnp.dot(P, (jnp.dot(P, signal) - measurements))

    return (updated_signal, T_matrix, scalar_ni)

@jit
def soft_threshold_psd_svd(matrix, lam):
    """
    Ensure the matrix is positive semidefinite and apply soft thresholding using SVD.
    
    Args:
        matrix (jnp.ndarray): Input Hermitian matrix (complex or real).
        lam (float): Threshold value for soft thresholding.
    
    Returns:
        jnp.ndarray: Processed matrix after ensuring PSD and applying soft thresholding.
    """
    # Step 1: Symmetrize to ensure Hermitian property
    hermitian_matrix = 0.5 * (matrix + matrix.conj().T)
    
    # Step 2: Perform SVD
    U, S, Vh = jnp.linalg.svd(hermitian_matrix, full_matrices=False)
    
    # Step 3: Soft threshold the singular values (equivalent to eigenvalues for Hermitian)
    S_shrink = jnp.maximum(S - lam, 0)
    
    # Step 4: Reconstruct the matrix using SVD components
    processed_matrix = U @ jnp.diag(S_shrink) @ U.conj().T
    
    return processed_matrix, S_shrink

@jit
def project_onto_psd_top_k(X, K):
    """
    Projects a Hermitian matrix X onto the PSD cone, retaining only the top K largest positive eigenvalues.

    Parameters:
    X (jax.numpy.ndarray): Hermitian matrix.
    K (int): Number of largest positive eigenvalues to retain.

    Returns:
    jax.numpy.ndarray: PSD matrix closest to X with rank at most K.
    """
    # Step 1: Eigen decomposition
    d, V = jnp.linalg.eigh(X)  # d: eigenvalues, V: eigenvectors

    # Step 2: Ensure positive eigenvalues and sort them in descending order
    idx = jnp.argsort(d)[::-1]  # Indices for sorting eigenvalues in descending order
    d_sorted = d[idx]  # Sorted eigenvalues
    V_sorted = V[:, idx]  # Corresponding eigenvectors

    # Step 3: Retain the top K positive eigenvalues
    d_top_k = jnp.where((jnp.arange(len(d_sorted)) < K) & (d_sorted > 0), d_sorted, 0)

    # Step 4: Reconstruct the matrix
    D_top_k = jnp.diag(d_top_k)  # Diagonal matrix of top K positive eigenvalues
    X_proj = V_sorted @ D_top_k @ jnp.conj(V_sorted.T)

    return X_proj


def apply_proximal_mapping(variables, lam_threshold):
    
    signal, T_matrix, scalar_ni = variables

    T_low_rank, D_threshold = soft_threshold_psd_svd(T_matrix, lam_threshold)

    new_scalar_ni = jnp.sum(D_threshold)

    Z = build_Z_matrix((signal, T_low_rank, new_scalar_ni))

    Z_np = np.array(Z)

    num_eigvecs = int(jnp.count_nonzero(D_threshold) + 1)
    
    eigvals, eigvecs = eigh(Z_np, subset_by_index = [Z_np.shape[0] - num_eigvecs, Z_np.shape[0] - 1])
    D = jnp.diag(eigvals)
    D_threshold = jnp.where(D > 0, D, 0)
    Z_thresholded = eigvecs @ D_threshold @ jnp.conj(jnp.transpose(eigvecs))

    N = len(signal)

    new_T_matrix = make_toeplitz_matrix(Z_thresholded[:N, :N])

    new_signal = Z_thresholded[:N, -1]

    new_scalar_ni = Z_thresholded[-1, -1]

    return (new_signal, new_T_matrix, new_scalar_ni)


def check_convergence(new_T_matrix, old_T_matrix, iteration_numb):
    
    convergence = jnp.linalg.norm(new_T_matrix - old_T_matrix) / jnp.linalg.norm(old_T_matrix)

    iteration_numb = iteration_numb + 1
    return convergence, iteration_numb

def run_ivdst_from_file(filename, frequencies = None, samples = 50):
    if frequencies == None:
        z, T2, ni0, non_null_indices = initialization_from_file(filename, samples)
    else:
        z, T2, ni0, non_null_indices = initialization_theoretical_guess_from_file(filename, frequencies, samples)

    variables = (z, T2, ni0)

    mask = create_modified_identity_matrix(len(z), non_null_indices)

    new_variables, variables, new_iteration_numb, old_momentum_coeff, convergence = one_step_iteration(z, mask, variables, None, 0, None, 0.5, 1e-3)
    print("Convergence at iter. num " + str(new_iteration_numb) + " is : " + str(convergence))

    conv = convergence

    while conv > 7e-5 and new_iteration_numb < 1000:

        new_variables, variables, new_iteration_numb, old_momentum_coeff, convergence = one_step_iteration(z, mask, new_variables, variables, new_iteration_numb, old_momentum_coeff, 1e-2, 1e-3)
        conv = convergence
        if (new_iteration_numb % 20) == 0:
            print("Convergence at iter. num " + str(new_iteration_numb) + " is : " + str(convergence))

    return new_variables[0], non_null_indices

def run_ivdst(molecule, samples, steps, frequencies = None):
    if frequencies == None:
        z, T2, ni0, non_null_indices = initialization(molecule, samples, steps)
    else:
        z, T2, ni0, non_null_indices = initialization_theoretical_guess(molecule, samples, steps, frequencies)

    variables = (z, T2, ni0)

    mask = create_modified_identity_matrix(len(z), non_null_indices)

#start = time.time()

    new_variables, variables, new_iteration_numb, old_momentum_coeff, convergence = one_step_iteration(z, mask, variables, None, 0, None, 1e-2, 1e-3)
    print("Convergence at iter. num " + str(new_iteration_numb) + " is : " + str(convergence))

    conv = convergence

    while conv > 1e-6:

        new_variables, variables, new_iteration_numb, old_momentum_coeff, convergence = one_step_iteration(z, mask, new_variables, variables, new_iteration_numb, old_momentum_coeff, 1e-2, 1e-3)
        conv = convergence
        print("Convergence at iter. num " + str(new_iteration_numb) + " is : " + str(convergence))

    return new_variables[0], non_null_indices


def ivdst_algorithm(hadamard_signal, sampled_indices):
    z = hadamard_signal
    r0 = jnp.outer(z, jnp.conj(z))

    T0 = make_toeplitz_matrix(r0)

    ni0 = jnp.trace(T0) / jnp.shape(r0)[0]

    variables = (z, T0, ni0)
    mask = create_modified_identity_matrix(len(z), sampled_indices)

    new_variables, variables, new_iteration_numb, old_momentum_coeff, convergence = one_step_iteration(z, mask, variables, None, 0, None, 1e-2, 5e-3)
    print("Convergence at iter. num " + str(new_iteration_numb) + " is : " + str(convergence))

    conv = convergence

    while conv > 1e-6:

        new_variables, variables, new_iteration_numb, old_momentum_coeff, convergence = one_step_iteration(z, mask, new_variables, variables, new_iteration_numb, old_momentum_coeff, 1e-2, 5e-3)
        conv = convergence
        print("Convergence at iter. num " + str(new_iteration_numb) + " is : " + str(convergence))

    return new_variables[0]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:39:51 2024

@author: davide
"""


import jax.numpy as jnp
import jax.random as jrandom
import pennylane.numpy as np
import os
import time





###### set of functions to generate unitary evolution ######


def compute_cost(indices, shots, N):

    def adjust_indices(indices):
        return [element if element < N // 2 else element // 2 for element in indices]
    
    indices = adjust_indices(indices)
    max_t = np.max(indices)


    total_runtime = shots * np.sum(indices)
    return max_t, total_runtime, len(indices)

def save_or_append_to_npy(file_path, new_vector):
    """
    Check if a .npy file exists. If it does, append the new vector as a row.
    Otherwise, create the file and save the vector.

    Parameters:
        file_path (str): The path to the .npy file.
        new_vector (numpy.ndarray): The vector to append or save.
    """
    if os.path.exists(file_path):
        # File exists, load the existing matrix
        matrix = np.load(file_path, allow_pickle = True)
        
        # Check if it's a matrix (2D) or single vector (1D)
        if matrix.ndim == 1:
            # Convert 1D vector to 2D matrix
            matrix = np.expand_dims(matrix, axis=0)
        
        # Append the new vector as a row
        matrix = np.vstack((matrix, new_vector))
    else:
        # File does not exist, create matrix from the vector
        matrix = np.expand_dims(new_vector, axis=0)

    # Save the updated matrix back to the file
    np.save(file_path, matrix)

def guess_autocorr(frequencies, N):
    """
    Generate a normalized signal composed of complex exponential oscillations.

    Parameters:
    - frequencies: List or array of frequencies (in Hz or arbitrary units).
    - N: Length of the output signal.

    Returns:
    - signal: A normalized numpy array of length N with the complex oscillating signal.
    """
    # Generate the time indices
    t = np.arange(0, (N // 2) + 1)
    t = jnp.hstack((-jnp.conj(t[::-1]), t[1:]))
    # Initialize the signal
    signal = np.zeros(N, dtype=complex)
    # Add contributions from each frequency
    for freq in frequencies:
        signal += 1/(np.sqrt(len(frequencies))) * np.exp(1j * freq * t)  # Complex exponential
    
    signal /= np.max(signal)

    # Normalize the signal
    #signal /= np.linalg.norm(signal)

    return signal


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


def transform_signal(signal):
    new_signal = jnp.hstack((jnp.conj(signal[::-1]), signal[1:]))
    # conj_signal = jnp.conj(signal)
    # middle_value = jnp.array([1.0 + 0.0j])  # Ensure the middle value is complex
    # new_signal = jnp.concatenate((conj_signal, middle_value, signal))
    return new_signal

def reversed_axis(signal):
    rev_sign = jnp.hstack((-jnp.conj(signal[::-1]), signal[1:]))
    return rev_sign


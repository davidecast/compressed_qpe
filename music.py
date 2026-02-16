#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  12 17:51:17 2024

@author: davide
"""

import numpy as np
import jax.numpy as jnp
import math
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import qeep_prony as prony


class MusicProcessor:
    """
    MUSIC (Multiple Signal Classification) spectrum processor.
    
    This class encapsulates the MUSIC algorithm for frequency estimation
    from autocorrelation signals.
    """
    
    def __init__(self, hamiltonian_norm, num_singular_values, batch_size=600000):
        """
        Initialize the MUSIC processor.
        
        Args:
            hamiltonian_norm: Normalization factor for the Hamiltonian
            num_singular_values: Number of singular values to use
            batch_size: Batch size for spectrum computation
        """
        self.hamiltonian_norm = hamiltonian_norm
        self.num_singular_values = num_singular_values
        self.batch_size = batch_size
        self.projector = None
        self.dim = None
        
    def construct_projector(self, S, dim):
        """
        Construct the projection matrix using the first `num_singular_values` singular vectors.
        
        Args:
            S: Singular vectors matrix
            dim: Dimension of the projector
            
        Returns:
            Projection matrix
        """
        P = np.zeros((dim, dim), dtype=complex)
        for i in range(self.num_singular_values):
            v = S[:, i]
            P += prony.projector(v, dim)
        return P
    
    def compute_batch(self, P2, freqs_batch, dim):
        """
        Compute MUSIC spectrum for a batch of frequencies.
        
        Args:
            P2: Null space projector
            freqs_batch: Batch of frequencies
            dim: Signal dimension
            
        Returns:
            Tuple of (X_batch, Y2_batch)
        """
        l = jnp.arange(dim)
        V_temp_batch = jnp.exp(1j * 2 * jnp.pi * freqs_batch[:, None] * l)
        V_transformed_batch = jnp.dot(P2, V_temp_batch.T).T  # Shape: (batch_size, dim)
        
        # Compute squared norm for each row (vector) using einsum
        norm_squared = jnp.einsum('ij,ij->i', V_transformed_batch, jnp.conj(V_transformed_batch))
        
        X_batch = -freqs_batch * self.hamiltonian_norm * 2 * jnp.pi
        Y2_batch = 1 / jnp.sqrt(jnp.abs(norm_squared) / dim)
        
        return X_batch, Y2_batch
    
    def compute_music_spectrum(self, P2, dim):
        """
        Compute MUSIC spectrum using batching to fit within memory constraints.
        
        Args:
            P2: Null space projector
            dim: Signal dimension
            
        Returns:
            Tuple of (X_axis, Y2_axis)
        """
        N = 3600000
        freqs = jnp.fft.rfftfreq(2 * N, d=1 / 2)[:-1]
        
        num_batches = N // self.batch_size
        results = []
        
        for i in range(num_batches):
            freqs_batch = freqs[i * self.batch_size : (i + 1) * self.batch_size]
            X_batch, Y2_batch = self.compute_batch(P2, freqs_batch, dim)
            results.append((X_batch, Y2_batch))
        
        # Handle last remaining batch if N is not divisible by batch_size
        remainder = N % self.batch_size
        if remainder > 0:
            freqs_batch = freqs[-remainder:]
            X_batch, Y2_batch = self.compute_batch(P2, freqs_batch, dim)
            results.append((X_batch, Y2_batch))
        
        # Concatenate final results
        X_axis, Y2_axis = zip(*results)
        return jnp.concatenate(X_axis), jnp.concatenate(Y2_axis)
    
    def estimate_frequencies(self, Y2_axis, X_axis, known_frequencies, threshold=2):
        """
        Estimate frequencies from the MUSIC spectrum and compute the error.
        
        Args:
            Y2_axis: MUSIC spectrum values
            X_axis: Frequency axis
            known_frequencies: Known true frequencies
            threshold: Threshold for peak detection
            
        Returns:
            Tuple of (estimated_frequencies, error, average_error)
        """
        peaks, _ = find_peaks(Y2_axis, height=threshold)
        estimated_frequencies = X_axis[peaks]
        estimated_frequencies = np.sort(estimated_frequencies)
        known_frequencies = np.sort(known_frequencies)
        
        if len(estimated_frequencies) >= len(known_frequencies):
            error = np.abs(estimated_frequencies[:len(known_frequencies)] - known_frequencies)
        else:
            error = np.abs(known_frequencies[:len(estimated_frequencies)] - estimated_frequencies)
        
        average_error = np.mean(error)
        
        return estimated_frequencies, error, average_error
    
    def process_signal(self, signal, dim):
        """
        Process a single signal using the MUSIC algorithm.
        
        Args:
            signal: Input autocorrelation signal
            dim: Dimension for Hankel matrix
            
        Returns:
            Tuple of (X_axis, Y2_axis)
        """
        H = prony.Hankel(signal, 0, dim)
        S, _, _ = np.linalg.svd(H)
        
        P1 = self.construct_projector(S, dim)
        P2 = np.identity(dim) - P1
        
        X_axis, Y2_axis = self.compute_music_spectrum(P2, dim)
        
        return X_axis, Y2_axis
    
    def average_spectra(self, spectra_list):
        """
        Compute the average spectrum from a list of spectra.
        
        Args:
            spectra_list: List of spectrum arrays
            
        Returns:
            Average spectrum
        """
        return np.mean(spectra_list, axis=0)
    
    def jackknife_error(self, spectra_list, known_frequencies, total_spectrum, X_axis):
        """
        Compute jackknife error of the average frequency.
        
        Args:
            spectra_list: List of spectra
            known_frequencies: Known true frequencies
            total_spectrum: Total spectrum
            X_axis: Frequency axis
            
        Returns:
            Tuple of (jackknife_mean, jackknife_std)
        """
        num_samples = len(spectra_list)
        jackknife_errors = []
        
        for i in range(num_samples):
            reduced_set = np.delete(spectra_list, i, axis=0)
            avg_reduced_spectrum = self.average_spectra(reduced_set)
            _, error, _ = self.estimate_frequencies(avg_reduced_spectrum, X_axis, known_frequencies)
            jackknife_errors.append(error)
        
        jackknife_errors = np.array(jackknife_errors)
        jackknife_mean = np.mean(jackknife_errors, axis=0)
        jackknife_std = np.std(jackknife_errors, axis=0)
        
        return jackknife_mean, jackknife_std
    
    def plot_spectrum(self, X_axis, Y2_axis, energies, f1, f2, nameplot):
        """
        Plot the computed MUSIC spectrum along with estimated energies.
        
        Args:
            X_axis: Frequency axis
            Y2_axis: MUSIC spectrum
            energies: Estimated energies
            f1: First reference frequency
            f2: Second reference frequency
            nameplot: Output plot filename
        """
        plt.figure(figsize=(10, 8))
        plt.plot(X_axis, Y2_axis)
        plt.plot(np.ones(int(np.max(Y2_axis))) * f1, np.arange(int(np.max(Y2_axis))), 
                 '--', c='red', label=r'E $|GS\rangle$')
        plt.plot(np.ones(int(np.max(Y2_axis))) * f2, np.arange(int(np.max(Y2_axis))), 
                 '--', c='green', label=r'E $|S_1\rangle$')
        
        plt.xlabel(r"$E$ [Ha]", fontsize=12)
        plt.ylabel(r"$J(E)$", fontsize=12)
        
        plt.xlim(f1 - 1.5, f1 + 1.5)
        plt.legend()
        plt.tight_layout()
        
        fig_width = 7  # inches
        fig_height = fig_width / 1.618  # Golden ratio
        plt.gcf().set_size_inches(fig_width, fig_height)
        
        plt.savefig(nameplot + '.png', dpi=300, bbox_inches='tight')


# Legacy functions for backward compatibility
def construct_projector(S, dim, num_singular_values):
    """Construct the projection matrix using the first `num_singular_values` singular vectors."""
    P = np.zeros((dim, dim), dtype=complex)
    for i in range(num_singular_values):
        v = S[:, i]
        P += prony.projector(v, dim)
    return P

def compute_batch(P2, freqs_batch, dim, hamiltonian_norm):
    """Compute MUSIC spectrum for a batch of frequencies."""
    l = jnp.arange(dim)
    V_temp_batch = jnp.exp(1j * 2 * jnp.pi * freqs_batch[:, None] * l)
    V_transformed_batch = jnp.dot(P2, V_temp_batch.T).T  # Shape: (batch_size, dim)

    # Compute squared norm for each row (vector) using einsum
    norm_squared = jnp.einsum('ij,ij->i', V_transformed_batch, jnp.conj(V_transformed_batch))

    X_batch = -freqs_batch * hamiltonian_norm * 2 * jnp.pi
    Y2_batch = 1 / jnp.sqrt(jnp.abs(norm_squared) / dim)

    return X_batch, Y2_batch

def compute_music_spectrum(P2, dim, hamiltonian_norm, batch_size=600000):
    """Compute MUSIC spectrum using batching to fit within 8GB of memory."""
    N = 3600000
    freqs = jnp.fft.rfftfreq(2 * N, d=1 / 2)[:-1]

    num_batches = N // batch_size
    results = []

    for i in range(num_batches):
        freqs_batch = freqs[i * batch_size : (i + 1) * batch_size]
        X_batch, Y2_batch = compute_batch(P2, freqs_batch, dim, hamiltonian_norm)
        results.append((X_batch, Y2_batch))

    # Handle last remaining batch if N is not divisible by batch_size
    remainder = N % batch_size
    if remainder > 0:
        freqs_batch = freqs[-remainder:]
        X_batch, Y2_batch = compute_batch(P2, freqs_batch, dim, hamiltonian_norm)
        results.append((X_batch, Y2_batch))

    # Concatenate final results
    X_axis, Y2_axis = zip(*results)
    return jnp.concatenate(X_axis), jnp.concatenate(Y2_axis)

def estimate_frequencies_and_compute_error(Y2_axis, X_axis, known_frequencies, threshold=2):
    """Estimate frequencies from the MUSIC spectrum and compute the error with respect to known frequencies."""
    peaks, _ = find_peaks(Y2_axis, height=threshold)
    estimated_frequencies = X_axis[peaks]
    estimated_frequencies = np.sort(estimated_frequencies)
    known_frequencies = np.sort(known_frequencies)

    if len(estimated_frequencies) >= len(known_frequencies):
        error = np.abs(estimated_frequencies[:len(known_frequencies)] - known_frequencies)
    else:
        error = np.abs(known_frequencies[:len(estimated_frequencies)] - estimated_frequencies)

    average_error = np.mean(error)

    return estimated_frequencies, error, average_error

def process_signal(signal, dim, hamiltonian_norm, num_singular_values):
    """Process a single signal using the MUSIC algorithm."""
    H = prony.Hankel(signal, 0, dim)
    S, _, _ = np.linalg.svd(H)

    P1 = construct_projector(S, dim, num_singular_values)
    P2 = np.identity(dim) - P1

    X_axis, Y2_axis = compute_music_spectrum(P2, dim, hamiltonian_norm)

    return X_axis, Y2_axis

def average_spectra(spectra_list):
    """Compute the average spectrum from a list of spectra."""
    return np.mean(spectra_list, axis=0)

def jackknife_error(spectra_list, known_frequencies, total_spectrum, X_axis):
    """Compute jackknife error of the average frequency compared to the total signal."""
    num_samples = len(spectra_list)
    jackknife_errors = []

    for i in range(num_samples):
        reduced_set = np.delete(spectra_list, i, axis=0)
        avg_reduced_spectrum = average_spectra(reduced_set)
        _, error = estimate_frequencies_and_compute_error(avg_reduced_spectrum, X_axis, known_frequencies)
        jackknife_errors.append(error)

    jackknife_errors = np.array(jackknife_errors)
    jackknife_mean = np.mean(jackknife_errors, axis=0)
    jackknife_std = np.std(jackknife_errors, axis=0)
    
    return jackknife_mean, jackknife_std

def plot_spectrum(X_axis, Y2_axis, energies, f1, f2, nameplot):
    """Plot the computed MUSIC spectrum along with estimated energies."""
    plt.figure(figsize=(10, 8))
    plt.plot(X_axis, Y2_axis)
    plt.plot(np.ones(int(np.max(Y2_axis))) * f1, np.arange(int(np.max(Y2_axis))), '--', c='red', label=r'E $|GS\rangle$')
    plt.plot(np.ones(int(np.max(Y2_axis))) * f2, np.arange(int(np.max(Y2_axis))), '--', c='green', label=r'E $|S_1\rangle$')

    plt.xlabel(r"$E$ [Ha]", fontsize=12)
    plt.ylabel(r"$J(E)$", fontsize=12)

    plt.xlim(f1 - 1.5, f1 + 1.5)
    plt.legend()
    plt.tight_layout()

    fig_width = 7  # inches
    fig_height = fig_width / 1.618  # Golden ratio, adjust as needed
    plt.gcf().set_size_inches(fig_width, fig_height)

    plt.savefig(nameplot + '.png', dpi=300, bbox_inches='tight')

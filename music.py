#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  12 17:51:17 2024

@author: davide
"""

import numpy as np
import math
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import qeep_prony as prony

def construct_projector(S, dim, num_singular_values):
    """Construct the projection matrix using the first `num_singular_values` singular vectors."""
    P = np.zeros((dim, dim), dtype=complex)
    for i in range(num_singular_values):
        v = S[:, i]
        P += prony.projector(v, dim)
    return P

def compute_music_spectrum(P2, dim, hamiltonian_norm):
    """Compute the MUSIC spectrum."""
    N = 180000
    X_axis = np.zeros(N)
    Y2_axis = np.zeros(N)

    freqs = np.fft.rfftfreq(2 * N, d=1/2)[:-1]
    l = np.arange(dim)
    for n in range(N):
        w = freqs[n]
        V_temp = np.zeros(dim, dtype=complex)
        # for l in range(dim):
        #     V_temp[l] = math.cos(2 * np.pi * w * l) + 1j * math.sin(2 * np.pi * w * l)
        V_temp = np.exp(1j * 2 * np.pi * w * l)

        V_temp = P2.dot(V_temp)

        X_axis[n] = -w * hamiltonian_norm * 2 * np.pi 
        Y2_axis[n] = 1 / math.sqrt(abs(np.vdot(V_temp, V_temp)) / dim)

    return X_axis, Y2_axis

def estimate_frequencies_and_compute_error(Y2_axis, X_axis, known_frequencies, threshold=5):
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


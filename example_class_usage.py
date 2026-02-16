#!/usr/bin/env python3
"""
Example usage of the refactored class-based API for compressed QPE.

This script demonstrates how to use the new class-based interfaces for:
- IVDSTSolver: Signal reconstruction from compressed measurements
- MusicProcessor: Frequency estimation using MUSIC algorithm
- PronyMethod: Polynomial operations and matrix utilities
"""

import numpy as np
from ivdst import IVDSTSolver
from music import MusicProcessor
from qeep_prony import PronyMethod


def example_ivdst_solver():
    """Example of using the IVDSTSolver class."""
    print("=" * 60)
    print("Example: IVDSTSolver class usage")
    print("=" * 60)
    
    # Create solver instance with custom parameters
    solver = IVDSTSolver(
        step_size=0.5,
        lam_threshold=1e-3,
        max_iterations=800
    )
    
    # Example: Process a file
    # filename = "lih_hadamard_measurements_npcs_500_steps_5_shots_better_integration.npy"
    # reconstructed_signal, indices = solver.solve_from_file(
    #     filename,
    #     guess_frequencies=None,
    #     samples=50
    # )
    
    print("IVDSTSolver instance created successfully!")
    print(f"  - Step size: {solver.step_size}")
    print(f"  - Lambda threshold: {solver.lam_threshold}")
    print(f"  - Max iterations: {solver.max_iterations}")
    print()


def example_music_processor():
    """Example of using the MusicProcessor class."""
    print("=" * 60)
    print("Example: MusicProcessor class usage")
    print("=" * 60)
    
    # Create MUSIC processor instance
    processor = MusicProcessor(
        hamiltonian_norm=8,
        num_singular_values=4,
        batch_size=600000
    )
    
    # Example: Process a signal
    # signal = np.load("reconstructed_signal.npy")
    # dim = len(signal) // 2
    # x_axis, y_axis = processor.process_signal(signal, dim)
    
    # Example: Estimate frequencies
    # true_freqs = [-7.880873948, -7.75139657, -7.32777980, -7.236480038]
    # estimated_frequencies, error, avg_error = processor.estimate_frequencies(
    #     y_axis, x_axis, true_freqs
    # )
    
    print("MusicProcessor instance created successfully!")
    print(f"  - Hamiltonian norm: {processor.hamiltonian_norm}")
    print(f"  - Num singular values: {processor.num_singular_values}")
    print(f"  - Batch size: {processor.batch_size}")
    print()


def example_prony_method():
    """Example of using the PronyMethod class."""
    print("=" * 60)
    print("Example: PronyMethod class usage")
    print("=" * 60)
    
    # PronyMethod is a static utility class
    prony = PronyMethod()
    
    # Example: Create a Hankel matrix
    signal = np.random.randn(100) + 1j * np.random.randn(100)
    size = 10
    hankel_matrix = PronyMethod.Hankel(signal, 0, size)
    
    print("PronyMethod instance created successfully!")
    print(f"  - Created Hankel matrix of size: {hankel_matrix.shape}")
    
    # Example: Create a random normalized vector
    dim = 5
    vec = PronyMethod.rand_vec(dim)
    print(f"  - Random vector norm: {np.linalg.norm(vec):.6f} (should be 1.0)")
    
    # Example: Create a projector
    projector = PronyMethod.projector(vec, dim)
    print(f"  - Projector matrix shape: {projector.shape}")
    print()


def example_backward_compatibility():
    """Example showing backward compatibility with legacy functions."""
    print("=" * 60)
    print("Example: Backward compatibility")
    print("=" * 60)
    
    # Legacy function-based API still works
    from ivdst import load_data, make_toeplitz_matrix
    from music import process_signal, estimate_frequencies_and_compute_error
    from qeep_prony import Hankel, projector
    
    print("Legacy functions are still available for backward compatibility:")
    print("  - ivdst.load_data()")
    print("  - ivdst.make_toeplitz_matrix()")
    print("  - music.process_signal()")
    print("  - music.estimate_frequencies_and_compute_error()")
    print("  - qeep_prony.Hankel()")
    print("  - qeep_prony.projector()")
    print()
    print("This ensures existing code continues to work without modifications.")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  Compressed QPE - Class-Based API Examples".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")
    
    example_ivdst_solver()
    example_music_processor()
    example_prony_method()
    example_backward_compatibility()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()

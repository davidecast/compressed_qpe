# Compressed QPE - Class-Based Refactoring

This repository contains a refactored version of the compressed quantum phase estimation (QPE) code, now organized using object-oriented programming principles with classes.

## Overview of Changes

The codebase has been refactored to use classes while maintaining **full backward compatibility** with the original function-based API.

### Refactored Modules

#### 1. `ivdst.py` - IVDSTSolver Class

The IVDST (Iterative Vandermonde Decomposition with Soft Thresholding) algorithm is now encapsulated in the `IVDSTSolver` class.

**New Class-Based API:**
```python
from ivdst import IVDSTSolver

# Create solver instance
solver = IVDSTSolver(
    step_size=0.5,
    lam_threshold=1e-3,
    max_iterations=800
)

# Initialize and solve from file
reconstructed_signal, indices = solver.solve_from_file(
    filename="data.npy",
    guess_frequencies=None,
    samples=50
)

# Or initialize and solve from molecule data
reconstructed_signal, indices = solver.solve(
    molecule="lih",
    samples=120,
    steps=600,
    guess_frequencies=[-7.9, -7.7, -7.3, -7.2],
    convergence_threshold=1e-6
)
```

**Legacy Function-Based API (still works):**
```python
from ivdst import run_ivdst_from_file, run_ivdst

# Function-based approach still works
reconstructed_signal, indices = run_ivdst_from_file(filename, frequencies, samples)
```

**Key Features:**
- Encapsulates algorithm state (T_matrix, scalar_ni, mask, etc.)
- Methods for initialization, iteration, and solving
- Configurable parameters (step_size, lam_threshold, max_iterations)
- Supports both file-based and molecule-based initialization
- Optional frequency guess for better convergence

#### 2. `music.py` - MusicProcessor Class

The MUSIC (Multiple Signal Classification) algorithm is now encapsulated in the `MusicProcessor` class.

**New Class-Based API:**
```python
from music import MusicProcessor

# Create MUSIC processor
processor = MusicProcessor(
    hamiltonian_norm=8,
    num_singular_values=4,
    batch_size=600000
)

# Process signal to get spectrum
x_axis, y_axis = processor.process_signal(signal, dim)

# Estimate frequencies
true_freqs = [-7.880873948, -7.75139657, -7.32777980, -7.236480038]
estimated_frequencies, error, avg_error = processor.estimate_frequencies(
    y_axis, x_axis, true_freqs, threshold=2
)

# Plot spectrum
processor.plot_spectrum(x_axis, y_axis, energies, f1, f2, "output_plot")
```

**Legacy Function-Based API (still works):**
```python
from music import process_signal, estimate_frequencies_and_compute_error

# Function-based approach still works
x_axis, y_axis = process_signal(signal, dim, hamiltonian_norm, num_singular_values)
estimated_frequencies, error, avg_error = estimate_frequencies_and_compute_error(
    y_axis, x_axis, true_freqs
)
```

**Key Features:**
- Encapsulates MUSIC algorithm parameters
- Batch processing for memory efficiency
- Methods for spectrum computation and frequency estimation
- Jackknife error estimation
- Plotting capabilities

#### 3. `qeep_prony.py` - PronyMethod Class

Prony method utilities are now organized in the `PronyMethod` class.

**New Class-Based API:**
```python
from qeep_prony import PronyMethod

# Static utility class
hankel_matrix = PronyMethod.Hankel(signal, init_idx=0, size=10)
projector = PronyMethod.projector(vec, dim)
random_vec = PronyMethod.rand_vec(dim)
toeplitz_matrix = PronyMethod.Toeplitz(signal, init_idx, L)
```

**Legacy Function-Based API (still works):**
```python
from qeep_prony import Hankel, projector, rand_vec

# Function-based approach still works
hankel_matrix = Hankel(signal, 0, 10)
```

**Key Features:**
- Static methods for matrix operations (Hankel, Toeplitz)
- Polynomial operations (product, inner product, etc.)
- Vector utilities (projector, random vector generation)
- All methods accessible as static methods

## Backward Compatibility

**Important:** All original function-based APIs are preserved for backward compatibility. Existing code using the old API will continue to work without any modifications.

The refactoring adds new class-based interfaces while keeping all legacy functions intact.

## Benefits of Class-Based Approach

1. **Better Organization**: Related functionality is grouped together in classes
2. **State Management**: Classes encapsulate algorithm state and parameters
3. **Easier Testing**: Classes can be instantiated with different configurations
4. **Extensibility**: New methods can be added to classes easily
5. **Documentation**: Class structure makes the API more discoverable
6. **Reusability**: Multiple instances can be created with different parameters

## Dependencies

The code requires the following Python packages:
- numpy
- jax
- scipy
- numba
- pennylane
- matplotlib

## Example Usage

See `example_class_usage.py` for comprehensive examples of the new class-based API.

```python
# Example: Complete workflow with classes
from ivdst import IVDSTSolver
from music import MusicProcessor

# Step 1: Reconstruct signal using IVDST
solver = IVDSTSolver(step_size=0.5, lam_threshold=1e-3, max_iterations=800)
reconstructed_signal, indices = solver.solve_from_file(
    filename="lih_data.npy",
    samples=50
)

# Step 2: Process signal with MUSIC algorithm
processor = MusicProcessor(hamiltonian_norm=8, num_singular_values=4)
x_axis, y_axis = processor.process_signal(
    reconstructed_signal,
    dim=len(reconstructed_signal) // 2
)

# Step 3: Estimate frequencies
true_freqs = [-7.880873948, -7.75139657, -7.32777980, -7.236480038]
estimated_frequencies, error, avg_error = processor.estimate_frequencies(
    y_axis, x_axis, true_freqs
)

print(f"Average error: {avg_error}")
print(f"Estimated frequencies: {estimated_frequencies}")
```

## Testing

The refactored code maintains the same functionality as the original. All existing tests should pass without modification.

To verify the refactoring:
1. Run existing test files (e.g., `test.py`)
2. Run the main entry point (`compressed_qpe.py`)
3. Run the example script (`example_class_usage.py`)

## Migration Guide

If you want to migrate from the function-based API to the class-based API:

### Before (Function-based):
```python
from ivdst import run_ivdst_from_file
from music import process_signal, estimate_frequencies_and_compute_error

reconstructed_signal, indices = run_ivdst_from_file(filename, frequencies, samples)
x_axis, y_axis = process_signal(reconstructed_signal, dim, norm, num_sv)
est_freqs, error, avg_error = estimate_frequencies_and_compute_error(y_axis, x_axis, true_freqs)
```

### After (Class-based):
```python
from ivdst import IVDSTSolver
from music import MusicProcessor

solver = IVDSTSolver()
reconstructed_signal, indices = solver.solve_from_file(filename, frequencies, samples)

processor = MusicProcessor(hamiltonian_norm=norm, num_singular_values=num_sv)
x_axis, y_axis = processor.process_signal(reconstructed_signal, dim)
est_freqs, error, avg_error = processor.estimate_frequencies(y_axis, x_axis, true_freqs)
```

## Structure

```
compressed_qpe/
├── ivdst.py              # IVDSTSolver class + legacy functions
├── music.py              # MusicProcessor class + legacy functions
├── qeep_prony.py         # PronyMethod class + legacy functions
├── compressed_qpe.py     # Main entry point (unchanged)
├── test.py               # Tests (unchanged)
├── utils.py              # Utility functions
├── utils_ivdst.py        # IVDST utilities
├── source.py             # Source utilities
├── example_class_usage.py # Example usage of new API
└── README.md             # This file
```

## License

[Original license applies]

## Authors

Original code by davidecast
Refactored to class-based structure while maintaining full backward compatibility

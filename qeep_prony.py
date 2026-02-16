import numpy  as np
import scipy
import math
from scipy import linalg
from matplotlib import pyplot as plt
import source as srs


class PronyMethod:
    """
    Prony method for signal processing and polynomial operations.
    
    This class encapsulates Prony-related operations for frequency estimation
    and polynomial manipulation.
    """
    
    def __init__(self):
        """Initialize the Prony method processor."""
        pass
    
    @staticmethod
    def projector(vec, dim):
        """
        Create a projection matrix from a vector.
        
        Args:
            vec: Input vector
            dim: Dimension of the projector
            
        Returns:
            Projection matrix
        """
        P = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            for j in range(dim):
                P[i][j] = vec[i] * (np.conj(vec[j]))
        return P
    
    @staticmethod
    def rand_vec(dim):
        """
        Generate a random normalized vector.
        
        Args:
            dim: Dimension of the vector
            
        Returns:
            Normalized random vector
        """
        vec = np.zeros(dim)
        rand_vec = np.random.normal(0, 1, dim)
        
        norm = math.sqrt(abs(np.vdot(rand_vec, rand_vec)))
        
        for i in range(dim):
            vec[i] = rand_vec[i] / norm
        
        return vec
    
    @staticmethod
    def Hankel(M, init_idx, size):
        """
        Create a Hankel matrix from a signal.
        
        Args:
            M: Input signal
            init_idx: Initial index
            size: Size of the Hankel matrix
            
        Returns:
            Hankel matrix
        """
        H = np.zeros((size, size), dtype=complex)
        for i in range(size):
            for j in range(size):
                H[i][j] = M[init_idx + i + j]
        return H
    
    @staticmethod
    def Toeplitz(M, init_idx, L):
        """
        Create a Toeplitz matrix.
        
        Args:
            M: Input signal
            init_idx: Initial index
            L: Size of the matrix
            
        Returns:
            Toeplitz matrix
        """
        T = np.zeros((L, L), dtype=complex)
        for i in range(L):
            for j in range(L):
                T[i][j] = M[i - j + L]
        return T
    
    @staticmethod
    def moment_vec(M, init_idx, size):
        """
        Extract a moment vector from a signal.
        
        Args:
            M: Input signal
            init_idx: Initial index
            size: Size of the vector
            
        Returns:
            Moment vector
        """
        vec = np.zeros(size, dtype=complex)
        for i in range(size):
            vec[i] = M[init_idx + i]
        return vec
    
    @staticmethod
    def norm_of_polynomial(vec_poly1, size1, vec_poly2, size2, M):
        """
        Compute the norm of a polynomial product.
        
        Args:
            vec_poly1: First polynomial coefficients
            size1: Size of first polynomial
            vec_poly2: Second polynomial coefficients
            size2: Size of second polynomial
            M: Moment matrix
            
        Returns:
            Norm value
        """
        length = size1 + size2 - 1
        coefficients = np.zeros(length, dtype=complex)
        for i in range(size1):
            for j in range(size2):
                idx = i + j
                coefficients[idx] = coefficients[idx] + vec_poly1[i] * vec_poly2[j]
        norm = 0
        for k in range(length):
            norm = norm + coefficients[k] * M[k]
        
        return norm.real
    
    @staticmethod
    def vec_add_one(vec, size):
        """
        Add a 1 at the end of a vector.
        
        Args:
            vec: Input vector
            size: Original size
            
        Returns:
            Extended vector
        """
        v_temp = np.zeros(size + 1, dtype=complex)
        for i in range(size):
            v_temp[i] = vec[i]
        v_temp[size] = 1
        
        return v_temp
    
    @staticmethod
    def polynomial_fix_size(vec, fixed_size):
        """
        Fix polynomial to a specific size by padding with zeros.
        
        Args:
            vec: Input polynomial coefficients
            fixed_size: Target size
            
        Returns:
            Fixed-size polynomial
        """
        vec_size = vec.size
        
        poly_temp = np.zeros(fixed_size, dtype=complex)
        if vec_size > fixed_size:
            print('error')
            return poly_temp
        else:
            for i in range(vec_size):
                poly_temp[i] = vec[i]
            return poly_temp
    
    @staticmethod
    def polynomial_product(vec1, vec2, fixed_size):
        """
        Compute the product of two polynomials.
        
        Args:
            vec1: First polynomial coefficients
            vec2: Second polynomial coefficients
            fixed_size: Maximum size for result
            
        Returns:
            Product polynomial
        """
        vec_size1 = vec1.size
        vec_size2 = vec2.size
        
        poly_temp = np.zeros(fixed_size, dtype=complex)
        if vec_size1 + vec_size2 > fixed_size:
            print('error')
            return poly_temp
        else:
            for i in range(vec_size1):
                for j in range(vec_size2):
                    poly_temp[i + j] = poly_temp[i + j] + vec1[i] * vec2[j]
            
            return poly_temp
    
    @staticmethod
    def polynomial_cut_end(vec, fixed_size):
        """
        Remove trailing zeros from a polynomial.
        
        Args:
            vec: Polynomial coefficients
            fixed_size: Current size
            
        Returns:
            Trimmed polynomial
        """
        length = 0
        for i in range(fixed_size):
            if vec[fixed_size - 1 - i] != 0:
                length = fixed_size - i
                break
        
        poly_new = np.zeros(length, dtype=complex)
        for j in range(length):
            poly_new[j] = vec[j]
        
        return poly_new
    
    @staticmethod
    def inner_product(v1, v2, fixed_size):
        """
        Compute inner product of two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            fixed_size: Size of vectors
            
        Returns:
            Inner product value
        """
        sum = 0 + 0 * 1j
        for i in range(fixed_size):
            sum = sum + v1[i] * v2[i]
        return sum
    
    @staticmethod
    def S_inner_product(poly1_pre, poly2_pre, S, L):
        """
        Compute S-inner product of two polynomials.
        
        Args:
            poly1_pre: First polynomial
            poly2_pre: Second polynomial
            S: S matrix
            L: Size
            
        Returns:
            S-inner product value
        """
        poly_new = PronyMethod.polynomial_product(poly1_pre, poly2_pre, L)
        return PronyMethod.inner_product(poly_new, S, L)
    
    @staticmethod
    def move_forward(vec_poly):
        """
        Shift polynomial coefficients forward (multiply by x).
        
        Args:
            vec_poly: Polynomial coefficients
            
        Returns:
            Shifted polynomial
        """
        vec_size = vec_poly.size
        vec_temp = np.zeros(vec_size + 1, dtype=complex)
        for i in range(1, vec_size + 1):
            vec_temp[i] = vec_poly[i - 1]
        return vec_temp
    
    @staticmethod
    def orthogonal_poly(n, x, S):
        """
        Compute orthogonal polynomial.
        
        Args:
            n: Polynomial degree
            x: Variable
            S: S matrix
            
        Returns:
            Orthogonal polynomial value
        """
        H1 = PronyMethod.Hankel(S, i, n + 1)
        D1 = np.linalg.det(H1)
        H0 = PronyMethod.Hankel(S, i, n)
        D0 = np.linalg.det(H0)
        
        OP = np.zeros((n + 1, n + 1), dtype=complex)
        for i in range(n):
            for j in range(n + 1):
                OP[i][j] = H1[i][j]
        
        for j in range(n + 1):
            OP[n][j] = x ** j
        
        return np.linalg.det(OP) / math.sqrt(D0 * D1)


# Legacy functions for backward compatibility
def projector(vec, dim):
    P = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            P[i][j] = vec[i] * (np.conj(vec[j]))
    
    return P

def rand_vec(dim):
    
    vec = np.zeros(dim)
    rand_vec = np.random.normal(0, 1, dim)
    
    norm = math.sqrt(abs(np.vdot(rand_vec, rand_vec)))
    
    for i in range(dim):
        vec[i] = rand_vec[i] / norm
    
    return vec

def Hankel(M, init_idx, size):
    
    H = np.zeros((size, size), dtype=complex)
    for i in range(size):
        for j in range(size):
            H[i][j] = M[init_idx + i + j]
    
    return H

def Toeplitz(M, init_idx, L):
    T = np.zeros((L, L), dtype=complex)
    for i in range(L):
        for j in range(L):
            T[i][j] = M[i - j + L]
    
    return T

def moment_vec(M, init_idx, size):
    
    vec = np.zeros(size, dtype=complex)
    for i in range(size):
        vec[i] = M[init_idx + i]
    return vec

def norm_of_polynomial(vec_poly1, size1, vec_poly2, size2, M):
    
    length = size1 + size2 - 1
    coefficients = np.zeros(length, dtype=complex)
    for i in range(size1):
        for j in range(size2):
            idx = i + j
            coefficients[idx] = coefficients[idx] + vec_poly1[i] * vec_poly2[j]
    norm = 0
    for k in range(length):
        norm = norm + coefficients[k] * M[k]
    
    return norm.real

def vec_add_one(vec, size):
    
    v_temp = np.zeros(size + 1, dtype=complex)
    for i in range(size):
        v_temp[i] = vec[i]
    v_temp[size] = 1
    
    return v_temp

def polynomial_fix_size(vec, fixed_size):
    vec_size = vec.size
    
    poly_temp = np.zeros(fixed_size, dtype=complex)
    if vec_size > fixed_size:
        print('error')
        return poly_temp
    else:
        for i in range(vec_size):
            poly_temp[i] = vec[i]
        return poly_temp

def polynomial_product(vec1, vec2, fixed_size):
    
    vec_size1 = vec1.size
    vec_size2 = vec2.size
    
    poly_temp = np.zeros(fixed_size, dtype=complex)
    if vec_size1 + vec_size2 > fixed_size:
        print('error')
        return poly_temp
    else:
        for i in range(vec_size1):
            for j in range(vec_size2):
                poly_temp[i + j] = poly_temp[i + j] + vec1[i] * vec2[j]
        
        return poly_temp

def polynomial_cut_end(vec, fixed_size):
    
    length = 0
    for i in range(fixed_size):
        if vec[fixed_size - 1 - i] != 0:
            length = fixed_size - i
            break
    
    poly_new = np.zeros(length, dtype=complex)
    for j in range(length):
        poly_new[j] = vec[j]
    
    return poly_new

def inner_product(v1, v2, fixed_size):
    
    sum = 0 + 0 * 1j
    for i in range(fixed_size):
        sum = sum + v1[i] * v2[i]
    return sum

def S_inner_product(poly1_pre, poly2_pre, S, L):
    poly_new = polynomial_product(poly1_pre, poly2_pre, L)
    return inner_product(poly_new, S, L)

def move_forward(vec_poly):
    
    vec_size = vec_poly.size
    vec_temp = np.zeros(vec_size + 1, dtype=complex)
    for i in range(1, vec_size + 1):
        vec_temp[i] = vec_poly[i - 1]
    return vec_temp


def orthogonal_poly(n, x, S):
    
    H1 = Hankel(S, i, n + 1)
    D1 = np.linalg.det(H1)
    H0 = Hankel(S, i, n)
    D0 = np.linalg.det(H0)
    
    OP = np.zeros((n + 1, n + 1), dtype=complex)
    for i in range(n):
        for j in range(n + 1):
            OP[i][j] = H1[i][j]
    
    for j in range(n + 1):
        OP[n][j] = x ** j
    
    return np.linalg.det(OP) / math.sqrt(D0 * D1)

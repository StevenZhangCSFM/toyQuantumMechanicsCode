from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator


class Hamiltonian:
    """
    Discretized 1D Hamiltonian with Dirichlet boundary conditions.

    The unknown vector stores interior grid values only. Boundary values are
    enforced through odd extension when centered finite-difference stencils
    reach outside the domain.
    """

    def __init__(self, N_grid: int, h: float, fd_order: int, V_interior: np.ndarray):
        self.N_grid = int(N_grid)
        self.h = float(h)
        self.fd_order = int(fd_order)
        self.V_interior = np.asarray(V_interior, dtype=float)
        self.N_interior = self.N_grid - 2

        if self.N_grid < 3:
            raise ValueError("N_grid must be at least 3.")
        if self.h <= 0.0:
            raise ValueError("Grid spacing h must be positive.")
        if self.fd_order not in (2, 4, 6, 8, 10):
            raise ValueError("fd_order must be one of 2, 4, 6, 8, 10.")
        if self.V_interior.shape != (self.N_interior,):
            raise ValueError(
                f"V_interior shape {self.V_interior.shape} does not match ({self.N_interior},)."
            )

        self._weights = self._generate_fd_weights_second_derivative()

    def make_operator(self) -> LinearOperator:
        def matvec(u: np.ndarray) -> np.ndarray:
            return self._apply_hamiltonian(np.asarray(u, dtype=float))

        return LinearOperator(
            shape=(self.N_interior, self.N_interior),
            matvec=matvec,
            dtype=float,
        )

    def _generate_fd_weights_second_derivative(self) -> np.ndarray:
        """
        Generate centered finite-difference weights for the second derivative.

        fd_order is the accuracy order: 2, 4, 6, 8, 10.
        The stencil half-width is p = fd_order // 2.
        """
        p = self.fd_order // 2
        offsets = np.arange(-p, p + 1, dtype=float)
        n = 2 * p + 1

        A = np.zeros((n, n), dtype=float)
        b = np.zeros(n, dtype=float)

        for k in range(n):
            A[k, :] = offsets**k
        b[2] = 2.0

        coeffs = np.linalg.solve(A, b)

        if not np.allclose(coeffs, coeffs[::-1], atol=1e-12, rtol=1e-12):
            raise ValueError("Generated FD coefficients failed symmetry check.")

        return coeffs

    def _odd_extended_value(self, u: np.ndarray, j: int) -> float:
        """
        Return the full-grid value phi_j under odd extension across both boundaries.

        Full-grid indices are 0..N_grid-1, with Dirichlet boundary values at
        j=0 and j=N_grid-1. The unknown vector u stores interior values for
        full-grid indices 1..N_grid-2.
        """
        sign = 1.0
        j_mapped = int(j)

        while j_mapped < 0 or j_mapped > self.N_grid - 1:
            if j_mapped < 0:
                j_mapped = -j_mapped
                sign = -sign
            elif j_mapped > self.N_grid - 1:
                j_mapped = 2 * (self.N_grid - 1) - j_mapped
                sign = -sign

        if j_mapped == 0 or j_mapped == self.N_grid - 1:
            return 0.0

        return sign * float(u[j_mapped - 1])

    def _apply_hamiltonian(self, u: np.ndarray) -> np.ndarray:
        if u.shape != (self.N_interior,):
            raise ValueError(
                f"Input vector shape {u.shape} does not match ({self.N_interior},)."
            )

        p = (len(self._weights) - 1) // 2
        Hu = np.zeros_like(u, dtype=float)
        inv_h2 = 1.0 / (self.h * self.h)

        for k in range(self.N_interior):
            full_i = k + 1
            lap_sum = 0.0

            for m in range(-p, p + 1):
                coeff = self._weights[m + p]
                val = self._odd_extended_value(u, full_i + m)
                lap_sum += coeff * val

            kinetic = -0.5 * inv_h2 * lap_sum
            potential = self.V_interior[k] * u[k]
            Hu[k] = kinetic + potential

        return Hu

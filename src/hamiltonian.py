from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse.linalg import LinearOperator


class HamiltonianBase(ABC):
    """Abstract Hamiltonian interface."""

    operator_dtype = np.float64

    @abstractmethod
    def make_operator(self) -> LinearOperator:
        raise NotImplementedError

    @staticmethod
    def generate_fd_weights_second_derivative(fd_order: int) -> np.ndarray:
        if fd_order not in (2, 4, 6, 8, 10):
            raise ValueError("fd_order must be one of 2, 4, 6, 8, 10.")

        p = fd_order // 2
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


class OneElectronHamiltonianBase(HamiltonianBase):
    operator_dtype = np.float64

    def __init__(self, N_grid: int, h: float, fd_order: int, V_interior: np.ndarray):
        self.N_grid = int(N_grid)
        self.h = float(h)
        self.fd_order = int(fd_order)
        self.V_interior = np.asarray(V_interior, dtype=self.operator_dtype)
        self.N_interior = int(self.expected_n_interior())

        if self.N_grid < 3:
            raise ValueError("N_grid must be at least 3.")
        if self.h <= 0.0:
            raise ValueError("Grid spacing h must be positive.")
        if self.V_interior.shape != (self.N_interior,):
            raise ValueError(
                f"V_interior shape {self.V_interior.shape} does not match ({self.N_interior},)."
            )

        self._weights = self.generate_fd_weights_second_derivative(self.fd_order)

    @abstractmethod
    def expected_n_interior(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def full_index_from_unknown_index(self, k: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def extended_value(self, u: np.ndarray, j: int):
        raise NotImplementedError

    def make_operator(self) -> LinearOperator:
        def matvec(u: np.ndarray) -> np.ndarray:
            return self._apply_hamiltonian(np.asarray(u, dtype=self.operator_dtype))

        return LinearOperator(
            shape=(self.N_interior, self.N_interior),
            matvec=matvec,
            dtype=self.operator_dtype,
        )

    def apply_kinetic_only(self, u: np.ndarray) -> np.ndarray:
        if u.shape != (self.N_interior,):
            raise ValueError(
                f"Input vector shape {u.shape} does not match ({self.N_interior},)."
            )

        p = (len(self._weights) - 1) // 2
        Hu = np.zeros(self.N_interior, dtype=self.operator_dtype)
        inv_h2 = 1.0 / (self.h * self.h)

        for k in range(self.N_interior):
            full_i = self.full_index_from_unknown_index(k)
            lap_sum = self.operator_dtype(0.0)
            for m in range(-p, p + 1):
                coeff = self._weights[m + p]
                val = self.extended_value(u, full_i + m)
                lap_sum += coeff * val
            Hu[k] = -0.5 * inv_h2 * lap_sum
        return Hu

    def _apply_hamiltonian(self, u: np.ndarray) -> np.ndarray:
        return self.apply_kinetic_only(u) + self.V_interior * u


class OneElectronDirichletHamiltonian(OneElectronHamiltonianBase):
    operator_dtype = np.float64

    def expected_n_interior(self) -> int:
        return self.N_grid - 2

    def full_index_from_unknown_index(self, k: int) -> int:
        return k + 1

    def extended_value(self, u: np.ndarray, j: int) -> float:
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


class OneElectronPeriodicHamiltonian(OneElectronHamiltonianBase):
    operator_dtype = np.float64

    def expected_n_interior(self) -> int:
        return self.N_grid - 1

    def full_index_from_unknown_index(self, k: int) -> int:
        return k

    def extended_value(self, u: np.ndarray, j: int) -> float:
        period = self.N_grid - 1
        return float(u[int(j) % period])


class OneElectronBlochHamiltonian(OneElectronHamiltonianBase):
    operator_dtype = np.complex128

    def __init__(self, N_grid: int, h: float, fd_order: int, V_interior: np.ndarray, kpt: float):
        self.kpt = float(kpt)
        super().__init__(N_grid=N_grid, h=h, fd_order=fd_order, V_interior=V_interior)

    def expected_n_interior(self) -> int:
        return self.N_grid - 1

    def full_index_from_unknown_index(self, k: int) -> int:
        return k

    def extended_value(self, u: np.ndarray, j: int) -> complex:
        period = self.N_grid - 1
        j_int = int(j)
        wrap_count = j_int // period
        j_mapped = j_int % period
        phase = np.exp(1j * self.kpt * (period * self.h) * wrap_count)
        return complex(u[j_mapped] * phase)


class ManyElectronHamiltonianBase(HamiltonianBase):
    operator_dtype = np.float64

    def __init__(self, problem_size: int):
        self._problem_size = int(problem_size)
        if self._problem_size < 1:
            raise ValueError("problem_size must be positive.")

    @property
    def problem_size(self) -> int:
        return self._problem_size

    def make_operator(self) -> LinearOperator:
        def matvec(u: np.ndarray) -> np.ndarray:
            return self._apply_hamiltonian(np.asarray(u, dtype=self.operator_dtype))

        return LinearOperator(
            shape=(self.problem_size, self.problem_size),
            matvec=matvec,
            dtype=self.operator_dtype,
        )

    @abstractmethod
    def _apply_hamiltonian(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TwoElectronDirichletHamiltonian(ManyElectronHamiltonianBase):
    operator_dtype = np.float64

    def __init__(
        self,
        N_grid: int,
        h: float,
        fd_order: int,
        V_interior: np.ndarray,
        x_interior: np.ndarray,
        interaction_softening: float,
        spin_symmetry: str,
    ):
        self.N_grid = int(N_grid)
        self.h = float(h)
        self.fd_order = int(fd_order)
        self.V_interior = np.asarray(V_interior, dtype=float)
        self.x_interior = np.asarray(x_interior, dtype=float)
        self.interaction_softening = float(interaction_softening)
        self.spin_symmetry = str(spin_symmetry)
        self.N_interior = self.N_grid - 2

        if self.V_interior.shape != (self.N_interior,):
            raise ValueError(
                f"V_interior shape {self.V_interior.shape} does not match ({self.N_interior},)."
            )
        if self.x_interior.shape != (self.N_interior,):
            raise ValueError(
                f"x_interior shape {self.x_interior.shape} does not match ({self.N_interior},)."
            )
        if self.interaction_softening <= 0.0:
            raise ValueError("interaction_softening must be positive.")
        if self.spin_symmetry not in ("singlet", "triplet"):
            raise ValueError("spin_symmetry must be 'singlet' or 'triplet'.")

        self.one_body = OneElectronDirichletHamiltonian(
            N_grid=self.N_grid,
            h=self.h,
            fd_order=self.fd_order,
            V_interior=np.zeros_like(self.V_interior),
        )
        x1 = self.x_interior[:, None]
        x2 = self.x_interior[None, :]
        self._pair_potential = (
            self.V_interior[:, None]
            + self.V_interior[None, :]
            + 1.0 / np.sqrt((x1 - x2) ** 2 + self.interaction_softening**2)
        )

        if self.spin_symmetry == "singlet":
            self._basis_pairs = [(i, j) for i in range(self.N_interior) for j in range(i, self.N_interior)]
        else:
            self._basis_pairs = [(i, j) for i in range(self.N_interior) for j in range(i + 1, self.N_interior)]
        super().__init__(problem_size=len(self._basis_pairs))

    def expand_reduced_vector_to_matrix(self, u: np.ndarray) -> np.ndarray:
        if u.shape != (self.problem_size,):
            raise ValueError(f"Input vector shape {u.shape} does not match ({self.problem_size},).")
        psi = np.zeros((self.N_interior, self.N_interior), dtype=float)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        for coeff, (i, j) in zip(u, self._basis_pairs):
            if i == j:
                psi[i, j] += coeff
            elif self.spin_symmetry == "singlet":
                amp = coeff * inv_sqrt2
                psi[i, j] += amp
                psi[j, i] += amp
            else:
                amp = coeff * inv_sqrt2
                psi[i, j] += amp
                psi[j, i] -= amp
        return psi

    def restrict_matrix_to_reduced_vector(self, psi: np.ndarray) -> np.ndarray:
        if psi.shape != (self.N_interior, self.N_interior):
            raise ValueError(
                f"Input matrix shape {psi.shape} does not match ({self.N_interior}, {self.N_interior})."
            )
        u = np.zeros(self.problem_size, dtype=float)
        sqrt2 = np.sqrt(2.0)
        for idx, (i, j) in enumerate(self._basis_pairs):
            if i == j:
                u[idx] = psi[i, j]
            elif self.spin_symmetry == "singlet":
                u[idx] = sqrt2 * psi[i, j]
            else:
                u[idx] = sqrt2 * psi[i, j]
        return u

    def _apply_full_hamiltonian_to_matrix(self, psi: np.ndarray) -> np.ndarray:
        Hpsi = np.zeros_like(psi)
        for j in range(self.N_interior):
            Hpsi[:, j] += self.one_body.apply_kinetic_only(psi[:, j])
        for i in range(self.N_interior):
            Hpsi[i, :] += self.one_body.apply_kinetic_only(psi[i, :])
        Hpsi += self._pair_potential * psi
        return Hpsi

    def _apply_hamiltonian(self, u: np.ndarray) -> np.ndarray:
        psi = self.expand_reduced_vector_to_matrix(u)
        Hpsi = self._apply_full_hamiltonian_to_matrix(psi)
        return self.restrict_matrix_to_reduced_vector(Hpsi)


class Hamiltonian:
    def __new__(
        cls,
        N_grid: int,
        h: float,
        fd_order: int,
        V_interior: np.ndarray,
        boundary: str = "dirichlet",
        kpt: float = 0.0,
        num_electrons: int = 1,
        x_interior: np.ndarray | None = None,
        interaction_softening: float = 1.0,
        spin_symmetry: str = "singlet",
    ):
        boundary = str(boundary)
        num_electrons = int(num_electrons)

        if num_electrons == 1:
            if boundary == "dirichlet":
                return OneElectronDirichletHamiltonian(N_grid=N_grid, h=h, fd_order=fd_order, V_interior=V_interior)
            if boundary == "periodic":
                return OneElectronPeriodicHamiltonian(N_grid=N_grid, h=h, fd_order=fd_order, V_interior=V_interior)
            if boundary == "bloch":
                return OneElectronBlochHamiltonian(N_grid=N_grid, h=h, fd_order=fd_order, V_interior=V_interior, kpt=kpt)
            raise ValueError("boundary must be 'dirichlet', 'periodic', or 'bloch'.")

        if num_electrons == 2:
            if boundary != "dirichlet":
                raise ValueError("Two-electron exact solver currently supports boundary='dirichlet' only.")
            if x_interior is None:
                raise ValueError("x_interior is required for the two-electron Hamiltonian.")
            return TwoElectronDirichletHamiltonian(
                N_grid=N_grid,
                h=h,
                fd_order=fd_order,
                V_interior=V_interior,
                x_interior=x_interior,
                interaction_softening=interaction_softening,
                spin_symmetry=spin_symmetry,
            )

        raise ValueError("num_electrons must be 1 or 2.")

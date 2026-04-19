from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.sparse.linalg import eigsh

from hamiltonian import Hamiltonian
from qm1d_setup import Grid1D, PreparedInputs, QM1DError, QM1DConfig


class QM1DExactSolver:
    def __init__(self, prepared: PreparedInputs, result_cls: type[Any]):
        self.prepared = prepared
        self.result_cls = result_cls
        self.config = prepared.config
        self.grid = prepared.grid
        self.potential_fn = prepared.potential_fn
        self.V_full = prepared.V_full
        self.V_interior = prepared.V_interior
        self.kpt_reduced = prepared.kpt_reduced
        self.effective_spin_symmetry = prepared.effective_spin_symmetry

        self.hamiltonian = Hamiltonian(
            N_grid=self.grid.N_grid,
            h=self.grid.h,
            fd_order=self.config.fd_order,
            V_interior=self.V_interior,
            boundary=self.config.boundary,
            kpt=self.kpt_reduced,
            num_electrons=self.config.num_electrons,
            x_interior=self.grid.x_interior,
            interaction_softening=self.config.interaction_softening,
            spin_symmetry=self.effective_spin_symmetry,
            many_body_method=self.config.many_body_method,
        )
        self.operator = self.hamiltonian.make_operator()

    @staticmethod
    def _normalize_eigenvectors(eigenvectors: np.ndarray, h: float) -> np.ndarray:
        normalized = np.array(eigenvectors, copy=True)
        for s in range(normalized.shape[1]):
            vec = normalized[:, s]
            norm = math.sqrt(float(np.sum(np.abs(vec) ** 2) * h))
            if norm == 0.0:
                raise QM1DError(f"Encountered zero norm for eigenvector {s}.")
            normalized[:, s] = vec / norm
        return normalized

    @staticmethod
    def _normalize_two_electron_wavefunctions(wavefunctions: np.ndarray, h: float) -> np.ndarray:
        normalized = np.array(wavefunctions, copy=True)
        for s in range(normalized.shape[2]):
            psi = normalized[:, :, s]
            norm = math.sqrt(float(np.sum(np.abs(psi) ** 2) * (h**2)))
            if norm == 0.0:
                raise QM1DError(f"Encountered zero norm for two-electron wavefunction {s}.")
            normalized[:, :, s] = psi / norm
        return normalized

    @staticmethod
    def _symmetrize_matrix(psi: np.ndarray) -> np.ndarray:
        return 0.5 * (psi + psi.T)

    @staticmethod
    def _antisymmetrize_matrix(psi: np.ndarray) -> np.ndarray:
        return 0.5 * (psi - psi.T)

    @classmethod
    def _project_exchange_symmetry(cls, psi: np.ndarray, spin_symmetry: str) -> np.ndarray:
        if spin_symmetry == "singlet":
            return cls._symmetrize_matrix(psi)
        if spin_symmetry == "triplet":
            return cls._antisymmetrize_matrix(psi)
        raise QM1DError("spin_symmetry must be 'singlet' or 'triplet'.")

    @classmethod
    def _build_initial_vector_for_two_electron(cls, grid: Grid1D, spin_symmetry: str) -> np.ndarray:
        x = grid.x_interior
        phi0 = np.exp(-0.5 * (x / max(grid.h, 1.0)) ** 2)
        phi1 = x * phi0
        psi = np.outer(phi0, phi0) if spin_symmetry == "singlet" else np.outer(phi0, phi1) - np.outer(phi1, phi0)
        psi = cls._project_exchange_symmetry(psi, spin_symmetry)
        norm = math.sqrt(float(np.sum(np.abs(psi) ** 2) * (grid.h**2)))
        if norm == 0.0:
            raise QM1DError("Failed to build a nonzero initial vector in the requested symmetry sector.")
        return (psi / norm).reshape(-1)

    @staticmethod
    def _reconstruct_full_orbitals(
        interior_vectors: np.ndarray,
        N_grid: int,
        boundary: str = "dirichlet",
        kpt: float = 0.0,
        L: float = 0.0,
    ) -> np.ndarray:
        n_states = interior_vectors.shape[1]
        full = np.zeros((N_grid, n_states), dtype=interior_vectors.dtype)
        if boundary == "periodic":
            full[:-1, :] = interior_vectors
            full[-1, :] = interior_vectors[0, :]
        elif boundary == "bloch":
            full[:-1, :] = interior_vectors
            full[-1, :] = interior_vectors[0, :] * np.exp(1j * kpt * L)
        else:
            full[1:-1, :] = interior_vectors
        return full

    @classmethod
    def _reconstruct_full_two_electron_wavefunctions(
        cls,
        interior_wavefunctions: np.ndarray,
        N_grid: int,
    ) -> np.ndarray:
        n_states = interior_wavefunctions.shape[2]
        full = np.zeros((N_grid, N_grid, n_states), dtype=interior_wavefunctions.dtype)
        full[1:-1, 1:-1, :] = interior_wavefunctions
        return full

    @staticmethod
    def _compute_two_electron_one_particle_density(
        interior_wavefunctions: np.ndarray,
        h: float,
        N_grid: int,
    ) -> np.ndarray:
        n_states = interior_wavefunctions.shape[2]
        rho_full = np.zeros((N_grid, n_states), dtype=float)
        for s in range(n_states):
            psi = interior_wavefunctions[:, :, s]
            rho_interior = 2.0 * np.sum(np.abs(psi) ** 2, axis=1) * h
            rho_full[1:-1, s] = rho_interior
        return rho_full

    @staticmethod
    def _check_exchange_symmetry(psi: np.ndarray, spin_symmetry: str, tol: float = 1e-8) -> None:
        if spin_symmetry == "singlet":
            err = np.linalg.norm(psi - psi.T)
        else:
            err = np.linalg.norm(psi + psi.T)
            diag_err = np.linalg.norm(np.diag(psi))
            err = max(err, diag_err)
        if err > tol:
            raise QM1DError(
                f"Computed two-electron wavefunction violates the requested {spin_symmetry} symmetry (error={err:.3e})."
            )

    def _solve_one_electron(self):
        n = self.grid.N_interior
        n_states = self.config.n_states

        ncv = self.config.ncv
        if ncv is None:
            ncv = min(n, max(2 * n_states + 1, 20))
        if ncv <= n_states:
            ncv = min(n, n_states + 2)

        eigenvalues, eigenvectors = eigsh(
            self.operator,
            k=n_states,
            which="SA",
            tol=self.config.tol,
            maxiter=self.config.maxiter,
            ncv=ncv,
        )

        order = np.argsort(eigenvalues)
        eigenvalues = np.asarray(eigenvalues[order], dtype=float)
        eigenvectors = np.asarray(eigenvectors[:, order])

        eigenvectors = self._normalize_eigenvectors(eigenvectors, self.grid.h)
        wavefunctions_full = self._reconstruct_full_orbitals(
            eigenvectors,
            self.grid.N_grid,
            self.config.boundary,
            self.kpt_reduced,
            self.config.L,
        )
        densities_full = np.abs(wavefunctions_full) ** 2

        return self.result_cls(
            config=self.config,
            grid=self.grid,
            eigenvalues=eigenvalues,
            wavefunctions_full=wavefunctions_full,
            densities_full=densities_full,
            V_full=self.V_full,
            V_interior=self.V_interior,
            orbitals_full=wavefunctions_full,
            kpt_reduced=self.kpt_reduced,
        )

    def _solve_two_electron_exact(self):
        dim = self.hamiltonian.problem_size
        n_states = self.config.n_states

        ncv = self.config.ncv
        if ncv is None:
            ncv = min(dim, max(2 * n_states + 1, 20))
        if ncv <= n_states:
            ncv = min(dim, n_states + 2)

        v0 = self.hamiltonian.restrict_matrix_to_reduced_vector(
            self._project_exchange_symmetry(
                self._build_initial_vector_for_two_electron(self.grid, self.effective_spin_symmetry).reshape(
                    self.grid.N_interior, self.grid.N_interior
                ),
                self.effective_spin_symmetry,
            )
        )
        eigenvalues, eigenvectors = eigsh(
            self.operator,
            k=n_states,
            which="SA",
            tol=self.config.tol,
            maxiter=self.config.maxiter,
            ncv=ncv,
            v0=v0,
        )

        order = np.argsort(eigenvalues)
        eigenvalues = np.asarray(eigenvalues[order], dtype=float)
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)

        wavefunctions_interior = np.zeros((self.grid.N_interior, self.grid.N_interior, n_states), dtype=float)
        for s in range(n_states):
            psi = self.hamiltonian.expand_reduced_vector_to_matrix(eigenvectors[:, s])
            psi = self._project_exchange_symmetry(psi, self.effective_spin_symmetry)
            wavefunctions_interior[:, :, s] = psi

        wavefunctions_interior = self._normalize_two_electron_wavefunctions(wavefunctions_interior, self.grid.h)
        for s in range(n_states):
            self._check_exchange_symmetry(wavefunctions_interior[:, :, s], self.effective_spin_symmetry)

        wavefunctions_full = self._reconstruct_full_two_electron_wavefunctions(
            wavefunctions_interior,
            self.grid.N_grid,
        )
        densities_full = np.abs(wavefunctions_full) ** 2
        one_particle_densities_full = self._compute_two_electron_one_particle_density(
            wavefunctions_interior,
            self.grid.h,
            self.grid.N_grid,
        )

        return self.result_cls(
            config=self.config,
            grid=self.grid,
            eigenvalues=eigenvalues,
            wavefunctions_full=wavefunctions_full,
            densities_full=densities_full,
            V_full=self.V_full,
            V_interior=self.V_interior,
            one_particle_densities_full=one_particle_densities_full,
            total_density_full=one_particle_densities_full[:, 0] if one_particle_densities_full.shape[1] > 0 else None,
            kpt_reduced=self.kpt_reduced,
        )

    def solve(self):
        if self.config.num_electrons == 1:
            return self._solve_one_electron()
        return self._solve_two_electron_exact()

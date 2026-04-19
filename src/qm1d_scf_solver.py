from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.sparse.linalg import eigsh

from hamiltonian import Hamiltonian
from qm1d_setup import PreparedInputs, QM1DError


class QM1DSCFSolver:
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

    def _orthonormalize_orbitals(self, orbitals: np.ndarray) -> np.ndarray:
        q = np.array(orbitals, dtype=float, copy=True)
        n_orb = q.shape[1]
        for i in range(n_orb):
            for j in range(i):
                overlap = float(np.vdot(q[:, j], q[:, i]).real * self.grid.h)
                q[:, i] -= overlap * q[:, j]
            norm = math.sqrt(float(np.sum(np.abs(q[:, i]) ** 2) * self.grid.h))
            if norm <= 1e-14:
                raise QM1DError("Failed to orthonormalize the orbital set.")
            q[:, i] /= norm
        return q

    def _build_initial_hf_orbitals(self) -> np.ndarray:
        x = self.grid.x_interior
        alpha = 1.0 / max(self.config.L, 1.0)
        phi0 = np.exp(-alpha * x * x)
        if self.effective_spin_symmetry == "singlet":
            orbitals = phi0.reshape(-1, 1)
        else:
            phi1 = x * np.exp(-alpha * x * x)
            orbitals = np.column_stack([phi0, phi1])
        return self._orthonormalize_orbitals(orbitals)

    def _get_hf_occupations(self) -> np.ndarray:
        if self.effective_spin_symmetry == "singlet":
            return np.array([2.0], dtype=float)
        return np.array([1.0, 1.0], dtype=float)

    def _build_hf_density(self, occupied_orbitals: np.ndarray) -> np.ndarray:
        occupations = self._get_hf_occupations()
        density = np.zeros(self.grid.N_interior, dtype=float)
        for occ, phi in zip(occupations, occupied_orbitals.T):
            density += occ * np.abs(phi) ** 2
        return density

    def _normalize_eigenvectors(self, eigenvectors: np.ndarray) -> np.ndarray:
        normalized = np.array(eigenvectors, copy=True)
        for s in range(normalized.shape[1]):
            vec = normalized[:, s]
            norm = math.sqrt(float(np.sum(np.abs(vec) ** 2) * self.grid.h))
            if norm == 0.0:
                raise QM1DError(f"Encountered zero norm for eigenvector {s}.")
            normalized[:, s] = vec / norm
        return normalized

    def _solve_one_electron_operator(self, operator, n_states: int) -> tuple[np.ndarray, np.ndarray]:
        n = self.grid.N_interior
        ncv = self.config.ncv
        if ncv is None:
            ncv = min(n, max(2 * n_states + 1, 20))
        if ncv <= n_states:
            ncv = min(n, n_states + 2)
        eigenvalues, eigenvectors = eigsh(
            operator,
            k=n_states,
            which="SA",
            tol=self.config.tol,
            maxiter=self.config.maxiter,
            ncv=ncv,
        )
        order = np.argsort(eigenvalues)
        eigenvalues = np.asarray(eigenvalues[order], dtype=float)
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)
        eigenvectors = self._normalize_eigenvectors(eigenvectors)
        return eigenvalues, eigenvectors

    def _select_hf_occupied_orbitals(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_occ = 1 if self.effective_spin_symmetry == "singlet" else 2
        return eigenvalues[:n_occ].copy(), self._orthonormalize_orbitals(eigenvectors[:, :n_occ].copy())

    def _mix_density(self, density_old: np.ndarray, density_new: np.ndarray) -> np.ndarray:
        alpha = self.config.scf_mixing
        return (1.0 - alpha) * density_old + alpha * density_new

    def _reconstruct_hf_two_electron_wavefunction(self, occupied_orbitals: np.ndarray) -> np.ndarray:
        psi_full = np.zeros((self.grid.N_grid, self.grid.N_grid, 1), dtype=float)
        if self.effective_spin_symmetry == "singlet":
            phi = occupied_orbitals[:, 0]
            psi_interior = np.outer(phi, phi)
        else:
            phi1 = occupied_orbitals[:, 0]
            phi2 = occupied_orbitals[:, 1]
            psi_interior = (np.outer(phi1, phi2) - np.outer(phi2, phi1)) / np.sqrt(2.0)

        norm = math.sqrt(float(np.sum(np.abs(psi_interior) ** 2) * (self.grid.h**2)))
        if norm == 0.0:
            raise QM1DError("HF reconstruction produced a zero two-electron wavefunction.")
        psi_full[1:-1, 1:-1, 0] = psi_interior / norm
        return psi_full

    def _reconstruct_full_orbitals(self, interior_orbitals: np.ndarray) -> np.ndarray:
        full = np.zeros((self.grid.N_grid, interior_orbitals.shape[1]), dtype=interior_orbitals.dtype)
        full[1:-1, :] = interior_orbitals
        return full

    def _compute_total_density_full(self, density_interior: np.ndarray) -> np.ndarray:
        rho_full = np.zeros(self.grid.N_grid, dtype=float)
        rho_full[1:-1] = density_interior
        return rho_full

    def solve(self):
        if self.config.many_body_method != "HF":
            raise NotImplementedError("QM1DSCFSolver currently implements HF only.")

        occupations = self._get_hf_occupations()
        occupied_orbitals = self._build_initial_hf_orbitals()
        density = self._build_hf_density(occupied_orbitals)

        converged = False
        density_change = math.inf
        final_orbital_energies: np.ndarray | None = None
        final_orbitals_all: np.ndarray | None = None
        final_occ_energies: np.ndarray | None = None
        final_occupied_orbitals: np.ndarray | None = None

        for iteration in range(1, self.config.scf_max_iter + 1):
            hamiltonian = Hamiltonian(
                N_grid=self.grid.N_grid,
                h=self.grid.h,
                fd_order=self.config.fd_order,
                V_interior=self.V_interior,
                boundary=self.config.boundary,
                num_electrons=1,
                x_interior=self.grid.x_interior,
                interaction_softening=self.config.interaction_softening,
                spin_symmetry=self.effective_spin_symmetry,
                many_body_method="HF",
                occupied_orbitals=occupied_orbitals,
                orbital_occupations=occupations,
            )
            operator = hamiltonian.make_operator()
            orbital_energies, orbitals_all = self._solve_one_electron_operator(operator, self.config.n_states)
            occ_energies, new_occupied_orbitals = self._select_hf_occupied_orbitals(orbital_energies, orbitals_all)
            new_density = self._build_hf_density(new_occupied_orbitals)
            density_change = math.sqrt(float(np.sum((new_density - density) ** 2) * self.grid.h))

            final_orbital_energies = orbital_energies
            final_orbitals_all = orbitals_all
            final_occ_energies = occ_energies
            final_occupied_orbitals = new_occupied_orbitals

            if density_change < self.config.scf_tol:
                occupied_orbitals = new_occupied_orbitals
                density = new_density
                converged = True
                break

            mixed_density = self._mix_density(density, new_density)
            scale = np.sqrt(np.maximum(mixed_density, 1e-15) / np.maximum(new_density, 1e-15))
            occupied_orbitals = self._orthonormalize_orbitals(new_occupied_orbitals * scale[:, None])
            density = self._build_hf_density(occupied_orbitals)

        if final_orbital_energies is None or final_orbitals_all is None or final_occupied_orbitals is None:
            raise QM1DError("HF SCF loop failed to produce a solution.")

        wavefunctions_full = self._reconstruct_hf_two_electron_wavefunction(final_occupied_orbitals)
        densities_full = np.abs(wavefunctions_full) ** 2
        total_density_full = self._compute_total_density_full(self._build_hf_density(final_occupied_orbitals))
        one_particle_densities_full = total_density_full.reshape(-1, 1)
        orbitals_full = self._reconstruct_full_orbitals(final_orbitals_all)

        return self.result_cls(
            config=self.config,
            grid=self.grid,
            eigenvalues=final_orbital_energies,
            wavefunctions_full=wavefunctions_full,
            densities_full=densities_full,
            V_full=self.V_full,
            V_interior=self.V_interior,
            one_particle_densities_full=one_particle_densities_full,
            orbitals_full=orbitals_full,
            many_body_method="HF",
            scf_iterations=iteration,
            scf_converged=converged,
            scf_density_change=density_change,
            total_density_full=total_density_full,
            orbital_occupations=occupations,
            occupied_orbital_energies=final_occ_energies,
            kpt_reduced=self.kpt_reduced,
        )

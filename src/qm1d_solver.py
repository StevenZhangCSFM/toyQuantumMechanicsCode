from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from scipy.sparse.linalg import eigsh

from hamiltonian import Hamiltonian


_ALLOWED_MATH_FUNCS = {
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "abs": np.abs,
    "pi": np.pi,
}


@dataclass
class QM1DConfig:
    """
    Configuration for the 1D stationary Schrodinger solver.

    Units:
    - Length: Bohr
    - Energy / potential: Hartree
    """

    L: float
    h_target: float
    fd_order: int
    n_states: int
    potential_expr: str
    parameters: Dict[str, float] = field(default_factory=dict)
    tol: float = 1e-10
    maxiter: Optional[int] = None
    ncv: Optional[int] = None


@dataclass
class Grid1D:
    N_grid: int
    N_interior: int
    h: float
    x_full: np.ndarray
    x_interior: np.ndarray


@dataclass
class QM1DResult:
    config: QM1DConfig
    grid: Grid1D
    eigenvalues: np.ndarray
    orbitals_full: np.ndarray
    densities_full: np.ndarray
    V_full: np.ndarray
    V_interior: np.ndarray


class QM1DError(ValueError):
    pass


class QM1DSolver:
    def __init__(self, config: QM1DConfig):
        self.config = config
        self._validate_basic_input(config)

        self.grid = self._make_grid(config.L, config.h_target)
        self._validate_grid_vs_fd_order(self.grid, config.fd_order, config.n_states)

        self.potential_fn = self._build_potential_function(
            expr=config.potential_expr,
            params=config.parameters,
        )
        self.V_full = self._evaluate_potential_on_grid(self.potential_fn, self.grid.x_full)
        self._validate_potential_values(self.V_full)
        self.V_interior = self.V_full[1:-1].copy()

        self.hamiltonian = Hamiltonian(
            N_grid=self.grid.N_grid,
            h=self.grid.h,
            fd_order=config.fd_order,
            V_interior=self.V_interior,
        )
        self.operator = self.hamiltonian.make_operator()

    @staticmethod
    def _validate_basic_input(config: QM1DConfig) -> None:
        if not (0.0 < config.L <= 50.0):
            raise QM1DError("L must be in (0, 50].")
        if config.h_target <= 0.0:
            raise QM1DError("h_target must be positive.")
        if config.fd_order not in (2, 4, 6, 8, 10):
            raise QM1DError("fd_order must be one of 2, 4, 6, 8, 10.")
        if config.n_states < 1:
            raise QM1DError("n_states must be at least 1.")
        if not config.potential_expr or not config.potential_expr.strip():
            raise QM1DError("potential_expr must be a non-empty string.")

    @staticmethod
    def _make_grid(L: float, h_target: float) -> Grid1D:
        N_grid = int(round(L / h_target)) + 1
        N_grid = max(N_grid, 3)

        h = L / (N_grid - 1)
        x_full = np.linspace(-0.5 * L, 0.5 * L, N_grid, dtype=float)
        x_interior = x_full[1:-1].copy()

        return Grid1D(
            N_grid=N_grid,
            N_interior=N_grid - 2,
            h=h,
            x_full=x_full,
            x_interior=x_interior,
        )

    @staticmethod
    def _validate_grid_vs_fd_order(grid: Grid1D, fd_order: int, n_states: int) -> None:
        p = fd_order // 2
        if grid.N_interior < 2 * p + 1:
            raise QM1DError(
                f"Grid too small for fd_order={fd_order}. "
                f"Need at least {2 * p + 1} interior points, got {grid.N_interior}."
            )
        if n_states >= grid.N_interior:
            raise QM1DError(
                f"n_states must be smaller than the number of interior points ({grid.N_interior})."
            )

    @staticmethod
    def _build_potential_function(
        expr: str, params: Optional[Dict[str, float]] = None
    ) -> Callable[[np.ndarray], np.ndarray]:
        params = dict(params or {})
        namespace = dict(_ALLOWED_MATH_FUNCS)
        namespace.update(params)

        code = compile(expr, "<potential_expr>", "eval")

        def V(x: np.ndarray | float) -> np.ndarray:
            local_ns = dict(namespace)
            local_ns["x"] = x
            return eval(code, {"__builtins__": {}}, local_ns)

        return V

    @staticmethod
    def _evaluate_potential_on_grid(
        V: Callable[[np.ndarray], np.ndarray], x_full: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(V(x_full), dtype=float)
        if values.shape != x_full.shape:
            raise QM1DError(
                f"Potential evaluation returned shape {values.shape}, expected {x_full.shape}."
            )
        return values

    @staticmethod
    def _validate_potential_values(V_full: np.ndarray) -> None:
        if not np.all(np.isfinite(V_full)):
            raise QM1DError("Potential contains NaN or Inf on the grid.")
        if np.iscomplexobj(V_full):
            raise QM1DError("Potential must be real-valued.")

    @staticmethod
    def _normalize_eigenvectors(eigenvectors: np.ndarray, h: float) -> np.ndarray:
        normalized = np.array(eigenvectors, dtype=float, copy=True)
        for s in range(normalized.shape[1]):
            vec = normalized[:, s]
            norm = math.sqrt(float(np.sum(np.abs(vec) ** 2) * h))
            if norm == 0.0:
                raise QM1DError(f"Encountered zero norm for eigenvector {s}.")
            normalized[:, s] = vec / norm
        return normalized

    @staticmethod
    def _reconstruct_full_orbitals(interior_vectors: np.ndarray, N_grid: int) -> np.ndarray:
        n_states = interior_vectors.shape[1]
        full = np.zeros((N_grid, n_states), dtype=float)
        full[1:-1, :] = interior_vectors
        return full

    def solve(self) -> QM1DResult:
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
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)

        eigenvectors = self._normalize_eigenvectors(eigenvectors, self.grid.h)
        orbitals_full = self._reconstruct_full_orbitals(eigenvectors, self.grid.N_grid)
        densities_full = np.abs(orbitals_full) ** 2

        return QM1DResult(
            config=self.config,
            grid=self.grid,
            eigenvalues=eigenvalues,
            orbitals_full=orbitals_full,
            densities_full=densities_full,
            V_full=self.V_full,
            V_interior=self.V_interior,
        )


def solve_qm1d(config: QM1DConfig) -> QM1DResult:
    solver = QM1DSolver(config)
    return solver.solve()

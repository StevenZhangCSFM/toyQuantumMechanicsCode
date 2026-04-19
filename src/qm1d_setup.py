from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional

import numpy as np

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
    Raw configuration for the 1D stationary Schrodinger solver.

    Units:
    - Length: Bohr
    - Energy / potential: Hartree
    """

    L: float
    h_target: float
    fd_order: int
    n_states: int
    potential_expr: str
    boundary: str = "dirichlet"
    kpt: float = 0.0
    relative_kpt: Optional[float] = None
    num_electrons: int = 1
    spin_symmetry: Optional[str] = None
    interaction_softening: float = 1.0
    many_body_method: Optional[str] = None
    parameters: Dict[str, float] = field(default_factory=dict)
    tol: float = 1e-10
    maxiter: Optional[int] = None
    ncv: Optional[int] = None
    scf_tol: float = 1e-8
    scf_max_iter: int = 100
    scf_mixing: float = 0.3


@dataclass
class Grid1D:
    N_grid: int
    N_interior: int
    h: float
    x_full: np.ndarray
    x_interior: np.ndarray


@dataclass
class PreparedInputs:
    config: QM1DConfig
    grid: Grid1D
    potential_fn: Callable[[np.ndarray | float], np.ndarray]
    V_full: np.ndarray
    V_interior: np.ndarray
    kpt_reduced: float
    effective_spin_symmetry: str


class QM1DError(ValueError):
    pass


def validate_config(config: QM1DConfig) -> None:
    if not (0.0 < config.L <= 50.0):
        raise QM1DError("L must be in (0, 50].")
    if config.h_target <= 0.0:
        raise QM1DError("h_target must be positive.")
    if config.fd_order not in (2, 4, 6, 8, 10):
        raise QM1DError("fd_order must be one of 2, 4, 6, 8, 10.")
    if config.n_states < 1:
        raise QM1DError("n_states must be at least 1.")
    if config.boundary not in ("dirichlet", "periodic", "bloch"):
        raise QM1DError("boundary must be one of: dirichlet, periodic, bloch.")
    if config.num_electrons not in (1, 2):
        raise QM1DError("num_electrons must be 1 or 2.")
    if config.many_body_method not in (None, "HF", "DFT"):
        raise QM1DError("many_body_method must be None, 'HF', or 'DFT'.")
    if config.scf_tol <= 0.0:
        raise QM1DError("scf_tol must be positive.")
    if config.scf_max_iter < 1:
        raise QM1DError("scf_max_iter must be at least 1.")
    if not (0.0 < config.scf_mixing <= 1.0):
        raise QM1DError("scf_mixing must be in (0, 1].")

    if config.num_electrons == 2 and config.boundary != "dirichlet" and config.many_body_method in (None, "HF", "DFT"):
        raise QM1DError("Two-electron solvers currently support boundary='dirichlet' only.")

    if config.num_electrons == 2:
        if config.spin_symmetry not in ("singlet", "triplet"):
            raise QM1DError("For num_electrons=2, spin_symmetry must be 'singlet' or 'triplet'.")
        if abs(config.kpt) > 0.0 or config.relative_kpt is not None:
            raise QM1DError("kpt and relative_kpt are not supported for num_electrons=2.")
        if config.interaction_softening <= 0.0:
            raise QM1DError("interaction_softening must be positive.")
    else:
        if config.spin_symmetry is not None:
            raise QM1DError("spin_symmetry is only supported when num_electrons=2.")
        if config.many_body_method is not None:
            raise QM1DError("many_body_method is currently supported only when num_electrons=2.")
        if config.boundary != "bloch":
            if abs(config.kpt) > 0.0:
                raise QM1DError("kpt may only be provided when boundary='bloch'.")
            if config.relative_kpt is not None:
                raise QM1DError("relative_kpt may only be provided when boundary='bloch'.")
        if config.boundary == "bloch" and config.relative_kpt is not None and abs(config.kpt) > 0.0:
            raise QM1DError("Provide at most one of kpt and relative_kpt.")

    if config.many_body_method in ("HF", "DFT"):
        if config.num_electrons != 2:
            raise QM1DError("HF/DFT modes currently require num_electrons=2.")
        if config.boundary != "dirichlet":
            raise QM1DError("HF/DFT modes currently require boundary='dirichlet'.")

    if not np.isfinite(config.kpt):
        raise QM1DError("kpt must be finite.")
    if config.relative_kpt is not None and not np.isfinite(config.relative_kpt):
        raise QM1DError("relative_kpt must be finite.")
    if not config.potential_expr or not config.potential_expr.strip():
        raise QM1DError("potential_expr must be a non-empty string.")


def reduce_k_to_first_bz(
    kpt: float,
    L: float,
    relative_kpt: Optional[float] = None,
) -> float:
    G = 2.0 * np.pi / L
    if relative_kpt is not None:
        kpt = float(relative_kpt) * G
    k_reduced = float(kpt) % G
    if k_reduced > 0.5 * G:
        k_reduced -= G
    upper_edge = np.pi / L
    if np.isclose(k_reduced, upper_edge, atol=1e-14, rtol=0.0):
        k_reduced = -upper_edge
    return float(k_reduced)


def make_grid(L: float, h_target: float, boundary: str = "dirichlet") -> Grid1D:
    N_grid = int(round(L / h_target)) + 1
    N_grid = max(N_grid, 3)

    h = L / (N_grid - 1)
    x_full = np.linspace(-0.5 * L, 0.5 * L, N_grid, dtype=float)
    if boundary in ("periodic", "bloch"):
        x_interior = x_full[:-1].copy()
        N_interior = N_grid - 1
    else:
        x_interior = x_full[1:-1].copy()
        N_interior = N_grid - 2

    return Grid1D(
        N_grid=N_grid,
        N_interior=N_interior,
        h=h,
        x_full=x_full,
        x_interior=x_interior,
    )


def validate_grid_vs_fd_order(
    grid: Grid1D,
    fd_order: int,
    n_states: int,
    config: QM1DConfig,
) -> None:
    p = fd_order // 2
    if grid.N_interior < 2 * p + 1:
        raise QM1DError(
            f"Grid too small for fd_order={fd_order}. "
            f"Need at least {2 * p + 1} interior points, got {grid.N_interior}."
        )

    if config.many_body_method in ("HF", "DFT"):
        dim = grid.N_interior
    elif config.num_electrons == 1:
        dim = grid.N_interior
    elif config.spin_symmetry == "triplet":
        dim = grid.N_interior * (grid.N_interior - 1) // 2
    else:
        dim = grid.N_interior * (grid.N_interior + 1) // 2

    if n_states >= dim:
        raise QM1DError(
            f"n_states must be smaller than the accessible Hilbert-space dimension ({dim})."
        )

    if config.many_body_method == "HF" and config.spin_symmetry == "triplet" and grid.N_interior < 2:
        raise QM1DError("Triplet HF requires at least two one-electron basis functions.")


def validate_potential_expression(
    expr: str,
    params: Optional[Dict[str, float]] = None,
) -> None:
    params = dict(params or {})
    allowed_names = {"x", *(_ALLOWED_MATH_FUNCS.keys()), *(params.keys())}

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise QM1DError(f"Invalid potential expression syntax: {exc.msg}.") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise QM1DError(
                "Invalid symbol in potential_expr: "
                f"'{node.id}'. Only 'x', parameter names, and allowed math names "
                f"{sorted(_ALLOWED_MATH_FUNCS.keys())} are permitted."
            )


def build_potential_function(
    expr: str,
    params: Optional[Dict[str, float]] = None,
) -> Callable[[np.ndarray | float], np.ndarray]:
    params = dict(params or {})
    namespace = dict(_ALLOWED_MATH_FUNCS)
    namespace.update(params)

    code = compile(expr, "<potential_expr>", "eval")

    def V(x: np.ndarray | float) -> np.ndarray:
        local_ns = dict(namespace)
        local_ns["x"] = x
        return eval(code, {"__builtins__": {}}, local_ns)

    return V


def evaluate_potential_on_grid(
    V: Callable[[np.ndarray | float], np.ndarray],
    x_full: np.ndarray,
) -> np.ndarray:
    values = np.asarray(V(x_full), dtype=float)
    if values.shape != x_full.shape:
        raise QM1DError(
            f"Potential evaluation returned shape {values.shape}, expected {x_full.shape}."
        )
    return values


def validate_potential_values(V_full: np.ndarray) -> None:
    if not np.all(np.isfinite(V_full)):
        raise QM1DError("Potential contains NaN or Inf on the grid.")
    if np.iscomplexobj(V_full):
        raise QM1DError("Potential must be real-valued.")


def warn_if_nonperiodic_bloch_potential(config: QM1DConfig, V_full: np.ndarray) -> None:
    if config.boundary != "bloch":
        return
    if not np.isclose(V_full[0], V_full[-1], atol=1e-10, rtol=1e-8):
        print(
            "Warning: boundary='bloch' assumes a periodic potential over the cell, "
            "but V(-L/2) and V(L/2) differ on the current grid."
        )


def prepare_solver_inputs(config: QM1DConfig) -> PreparedInputs:
    validate_config(config)

    effective_kpt = config.kpt
    if config.relative_kpt is not None:
        effective_kpt = float(config.relative_kpt) * (2.0 * np.pi / config.L)

    prepared_config = replace(config, kpt=effective_kpt)
    grid = make_grid(prepared_config.L, prepared_config.h_target, prepared_config.boundary)
    validate_grid_vs_fd_order(grid, prepared_config.fd_order, prepared_config.n_states, prepared_config)

    validate_potential_expression(prepared_config.potential_expr, prepared_config.parameters)
    potential_fn = build_potential_function(prepared_config.potential_expr, prepared_config.parameters)
    V_full = evaluate_potential_on_grid(potential_fn, grid.x_full)
    validate_potential_values(V_full)
    warn_if_nonperiodic_bloch_potential(prepared_config, V_full)

    if prepared_config.boundary == "dirichlet":
        V_interior = V_full[1:-1].copy()
    else:
        V_interior = V_full[:-1].copy()

    return PreparedInputs(
        config=prepared_config,
        grid=grid,
        potential_fn=potential_fn,
        V_full=V_full,
        V_interior=V_interior,
        kpt_reduced=reduce_k_to_first_bz(
            prepared_config.kpt,
            prepared_config.L,
            prepared_config.relative_kpt,
        ),
        effective_spin_symmetry=prepared_config.spin_symmetry or "singlet",
    )

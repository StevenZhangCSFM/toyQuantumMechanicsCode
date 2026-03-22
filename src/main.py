from __future__ import annotations

import argparse

import numpy as np

from plotting import plot_potential_and_orbitals
from qm1d_solver import QM1DConfig, QM1DResult, solve_qm1d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="1D stationary Schrodinger solver demos")
    parser.add_argument(
        "--test_zero",
        action="store_true",
        help="Run the zero-potential-in-a-box regression test.",
    )
    parser.add_argument(
        "--test_harmonic",
        action="store_true",
        help="Run the harmonic-oscillator test.",
    )
    parser.add_argument(
        "--potential_expr",
        type=str,
        help=(
            "User-provided 1D potential expression, such as "
            "'0.5*x**2' or 'sin(x) + 0.1*x**2'. Required unless "
            "--test_zero or --test_harmonic is specified."
        ),
    )
    parser.add_argument(
        "--L",
        type=float,
        help="Half-box length L in Bohr for user-input runs.",
    )
    parser.add_argument(
        "--h_target",
        type=float,
        help="Target grid spacing in Bohr for user-input runs.",
    )
    parser.add_argument(
        "--fd_order",
        type=int,
        help="Finite-difference order for user-input runs.",
    )
    parser.add_argument(
        "--n_states",
        type=int,
        help="Number of eigenstates to compute for user-input runs.",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="Display the plot window in addition to saving the PDF.",
    )
    parser.add_argument(
        "--output_pdf",
        default="user_input_potential_orbitals.pdf",
        help="Output PDF filename for the saved plot.",
    )
    return parser


def make_default_config() -> QM1DConfig:
    """Return the default configuration for user-input runs.

    This reuses the numerical settings of the harmonic-oscillator reference
    case, but leaves potential_expr blank so the user must provide it.
    """
    return QM1DConfig(
        L=10.0,
        h_target=0.05,
        fd_order=6,
        n_states=4,
        potential_expr="",
        parameters={},
        tol=1e-12,
    )


def make_zero_potential_test_config() -> QM1DConfig:
    config = make_default_config()
    config.potential_expr = "0.0*x"
    return config


def make_harmonic_oscillator_config() -> QM1DConfig:
    config = make_default_config()
    config.potential_expr = "0.5*x**2"
    return config


def make_user_input_config(args: argparse.Namespace) -> QM1DConfig:
    """Build a QM1DConfig from command-line input for a user-defined potential."""
    config = make_default_config()

    if not args.potential_expr:
        raise ValueError(
            "Missing required argument --potential_expr when not running "
            "--test_zero or --test_harmonic."
        )

    config.potential_expr = args.potential_expr

    if args.L is not None:
        config.L = args.L
    if args.h_target is not None:
        config.h_target = args.h_target
    if args.fd_order is not None:
        config.fd_order = args.fd_order
    if args.n_states is not None:
        config.n_states = args.n_states

    return config


def _harmonic_oscillator_theoretical_eigenvalues(n_states: int) -> np.ndarray:
    """Return exact full-line eigenvalues for H = -1/2 d^2/dx^2 + 1/2 x^2.

    In atomic units with m = omega = hbar = 1, the spectrum is
        E_n = n + 1/2,   n = 0, 1, 2, ...
    """
    n = np.arange(n_states, dtype=float)
    return n + 0.5


def print_summary(result: QM1DResult) -> None:
    print("1D stationary Schrodinger solver")
    print(f"L               = {result.config.L:.10g} Bohr")
    print(f"h_target        = {result.config.h_target:.10g} Bohr")
    print(f"h_actual        = {result.grid.h:.10g} Bohr")
    print(f"N_grid          = {result.grid.N_grid}")
    print(f"N_interior      = {result.grid.N_interior}")
    print(f"fd_order        = {result.config.fd_order}")
    print(f"n_states        = {result.config.n_states}")
    print(f"potential_expr  = {result.config.potential_expr}")
    print("Eigenvalues (Hartree):")
    for i, ev in enumerate(result.eigenvalues, start=1):
        print(f"  state {i:2d}: {ev:.14f}")


def print_zero_box_comparison(result: QM1DResult) -> None:
    """Print comparison to the infinite square well energies.

    For a particle in a 1D infinite square well of width L with Dirichlet
    boundary conditions at the two ends, the exact eigenvalues are

        E_n = n^2 pi^2 / (2 L^2),   n = 1, 2, 3, ...
    """
    config = result.config
    print("\nComparison to exact box energies:")
    for n, E_num in enumerate(result.eigenvalues, start=1):
        E_exact = (n * n * np.pi * np.pi) / (2.0 * config.L * config.L)
        err = E_num - E_exact
        print(f"  n={n:2d}: numerical={E_num:.14f}, exact={E_exact:.14f}, error={err:.3e}")


def print_harmonic_oscillator_comparison(result: QM1DResult) -> None:
    """Print comparison to exact full-line harmonic-oscillator energies.

    For H = -1/2 d^2/dx^2 + 1/2 x^2 in atomic units, the exact spectrum is

        E_n = n + 1/2,   n = 0, 1, 2, ...

    The finite interval with Dirichlet boundary conditions restricts the trial
    space relative to the full-line problem, so the boxed continuum problem is
    expected to have energies above the full-line values. The present finite-
    difference discretization is not strictly variational, so the discrete
    eigenvalues need not be exact upper bounds state by state.
    """
    E_theory = _harmonic_oscillator_theoretical_eigenvalues(len(result.eigenvalues))

    print("\nComparison to exact harmonic-oscillator energies:")
    print("  Model Hamiltonian: H = -1/2 d^2/dx^2 + 1/2 x^2")
    print("  Exact full-line spectrum: E_n = n + 1/2")
    for n, (E_num, E_exact) in enumerate(zip(result.eigenvalues, E_theory)):
        err = E_num - E_exact
        print(
            f"  n={n:2d}: numerical={E_num:.14f}, exact={E_exact:.14f}, numerical-exact={err:.3e}"
        )


def run_zero_potential_test() -> QM1DResult:
    result = solve_qm1d(make_zero_potential_test_config())
    print_summary(result)
    print_zero_box_comparison(result)
    return result


def run_harmonic_oscillator_test(
    show_plot: bool = False,
    output_pdf: str = "harmonic_oscillator_orbitals.pdf",
) -> QM1DResult:
    result = solve_qm1d(make_harmonic_oscillator_config())
    print_summary(result)
    print_harmonic_oscillator_comparison(result)
    saved_path = plot_potential_and_orbitals(result, output_pdf=output_pdf, show=show_plot)
    print(f"\nSaved figure to: {saved_path}")
    return result


def run_user_input(
    args: argparse.Namespace,
    show_plot: bool = False,
    output_pdf: str = "user_input_potential_orbitals.pdf",
) -> QM1DResult:
    """Solve and plot a user-supplied 1D potential from parsed CLI arguments."""
    result = solve_qm1d(make_user_input_config(args))
    print_summary(result)
    saved_path = plot_potential_and_orbitals(result, output_pdf=output_pdf, show=show_plot)
    print(f"\nSaved figure to: {saved_path}")
    return result


def main() -> QM1DResult | None:
    args = build_parser().parse_args()

    if args.test_zero:
        return run_zero_potential_test()

    if args.test_harmonic:
        return run_harmonic_oscillator_test(show_plot=args.show_plot, output_pdf=args.output_pdf)

    return run_user_input(args=args, show_plot=args.show_plot, output_pdf=args.output_pdf)


if __name__ == "__main__":
    main()

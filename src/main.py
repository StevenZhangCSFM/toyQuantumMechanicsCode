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
    parser.add_argument("--L", type=float, help="Half-box length L in Bohr for user-input runs.")
    parser.add_argument("--h_target", type=float, help="Target grid spacing in Bohr for user-input runs.")
    parser.add_argument("--fd_order", type=int, help="Finite-difference order for user-input runs.")
    parser.add_argument("--n_states", type=int, help="Number of eigenstates to compute for user-input runs.")
    parser.add_argument(
        "--boundary",
        type=str,
        help="Boundary condition: dirichlet, periodic, or bloch.",
    )
    parser.add_argument(
        "--kpt",
        type=float,
        help="Scalar Bloch wavevector in 1/Bohr. Used only when --boundary bloch.",
    )
    parser.add_argument(
        "--relative_kpt",
        type=float,
        help=(
            "Relative Bloch k-point coordinate without unit. The real wavevector is "
            "relative_kpt * (2*pi/L). Requires --boundary bloch."
        ),
    )
    parser.add_argument(
        "--num_electrons",
        type=int,
        help="Number of electrons. Currently supported values are 1 and 2.",
    )
    parser.add_argument(
        "--spin_symmetry",
        type=str,
        help="For the exact two-electron solver: singlet or triplet.",
    )
    parser.add_argument(
        "--interaction_softening",
        type=float,
        help="Softening parameter a in the two-electron soft-Coulomb interaction 1/sqrt((x1-x2)^2+a^2).",
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
    return QM1DConfig(
        L=10.0,
        h_target=0.05,
        fd_order=6,
        n_states=4,
        potential_expr="",
        boundary="dirichlet",
        kpt=0.0,
        relative_kpt=None,
        num_electrons=1,
        spin_symmetry=None,
        interaction_softening=1.0,
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
    if args.boundary is not None:
        config.boundary = args.boundary
    if args.kpt is not None:
        config.kpt = args.kpt
    if args.relative_kpt is not None:
        config.relative_kpt = args.relative_kpt
    if args.num_electrons is not None:
        config.num_electrons = args.num_electrons
    if args.spin_symmetry is not None:
        config.spin_symmetry = args.spin_symmetry
    if args.interaction_softening is not None:
        config.interaction_softening = args.interaction_softening

    return config


def _harmonic_oscillator_theoretical_eigenvalues(n_states: int) -> np.ndarray:
    n = np.arange(n_states, dtype=float)
    return n + 0.5


def print_summary(result: QM1DResult) -> None:
    print("1D stationary Schrodinger solver")
    print(f"num_electrons   = {result.config.num_electrons}")
    print(f"L               = {result.config.L:.10g} Bohr")
    print(f"h_target        = {result.config.h_target:.10g} Bohr")
    print(f"h_actual        = {result.grid.h:.10g} Bohr")
    print(f"N_grid          = {result.grid.N_grid}")
    print(f"N_interior      = {result.grid.N_interior}")
    print(f"fd_order        = {result.config.fd_order}")
    print(f"n_states        = {result.config.n_states}")
    print(f"potential_expr  = {result.config.potential_expr}")
    print(f"boundary        = {result.config.boundary}")
    if result.config.num_electrons == 2:
        print(f"spin_symmetry   = {result.config.spin_symmetry}")
        print(f"softening_a     = {result.config.interaction_softening:.10g} Bohr")
    if result.config.boundary == "bloch":
        if result.config.relative_kpt is not None:
            print(f"relative_kpt    = {result.config.relative_kpt:.14f}")
        print(f"kpt_input       = {result.config.kpt:.14f} 1/Bohr")
        print(f"kpt_reduced     = {result.config.kpt_reduced:.14f} 1/Bohr")
        print(f"-kpt_reduced    = {-result.config.kpt_reduced:.14f} 1/Bohr")
    print("Eigenvalues (Hartree):")
    for i, ev in enumerate(result.eigenvalues, start=1):
        print(f"  state {i:2d}: {ev:.14f}")


def print_zero_box_comparison(result: QM1DResult) -> None:
    config = result.config
    print("\nComparison to exact box energies:")
    for n, E_num in enumerate(result.eigenvalues, start=1):
        E_exact = (n * n * np.pi * np.pi) / (2.0 * config.L * config.L)
        err = E_num - E_exact
        print(f"  n={n:2d}: numerical={E_num:.14f}, exact={E_exact:.14f}, error={err:.3e}")


def print_harmonic_oscillator_comparison(result: QM1DResult) -> None:
    E_theory = _harmonic_oscillator_theoretical_eigenvalues(len(result.eigenvalues))
    print("\nComparison to exact harmonic-oscillator energies:")
    print("  Model Hamiltonian: H = -1/2 d^2/dx^2 + 1/2 x^2")
    print("  Exact full-line spectrum: E_n = n + 1/2")
    for n, (E_num, E_exact) in enumerate(zip(result.eigenvalues, E_theory)):
        err = E_num - E_exact
        print(f"  n={n:2d}: numerical={E_num:.14f}, exact={E_exact:.14f}, numerical-exact={err:.3e}")


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

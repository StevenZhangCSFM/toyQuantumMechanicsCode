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
        help="Run the zero-potential-in-a-box regression test instead of the harmonic oscillator demo.",
    )
    parser.add_argument(
        "--test_harmonic",
        action="store_true",
        help="Run the harmonic-oscillator test.",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="Display the plot window in addition to saving the PDF.",
    )
    parser.add_argument(
        "--output_pdf",
        default="harmonic_oscillator_orbitals.pdf",
        help="Output PDF filename for the harmonic-oscillator plot.",
    )
    return parser


def make_zero_potential_test_config() -> QM1DConfig:
    return QM1DConfig(
        L=10.0,
        h_target=0.05,
        fd_order=6,
        n_states=4,
        potential_expr="0.0*x",
        parameters={},
        tol=1e-12,
    )


def make_harmonic_oscillator_config() -> QM1DConfig:
    return QM1DConfig(
        L=10.0,
        h_target=0.05,
        fd_order=6,
        n_states=4,
        potential_expr="0.5*x**2",
        parameters={},
        tol=1e-12,
    )


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
    """
    Compare numerical eigenvalues against the infinite square well on [0, L].

    With atomic units and zero potential inside the interval, the exact energies are

        E_n = n^2 pi^2 / (2 L^2),   n = 1, 2, 3, ...

    for Dirichlet boundary conditions at the two endpoints.
    """
    config = result.config
    print("\nComparison to exact box energies:")
    for n, E_num in enumerate(result.eigenvalues, start=1):
        E_exact = (n * n * np.pi * np.pi) / (2.0 * config.L * config.L)
        err = E_num - E_exact
        print(f"  n={n:2d}: numerical={E_num:.14f}, exact={E_exact:.14f}, error={err:.3e}")


def _harmonic_oscillator_theoretical_eigenvalues(n_states: int) -> np.ndarray:
    """
    Harmonic-oscillator eigenvalues for the convention used here:

        H = -1/2 d^2/dx^2 + 1/2 x^2

    In atomic units, this is the standard omega = 1 harmonic oscillator, so

        E_n = n + 1/2,   n = 0, 1, 2, ...
    """
    n = np.arange(n_states, dtype=float)
    return n + 0.5


def print_harmonic_oscillator_comparison(result: QM1DResult) -> None:
    theory = _harmonic_oscillator_theoretical_eigenvalues(len(result.eigenvalues))

    print("\nComparison to theoretical harmonic-oscillator eigenvalues:")
    print("  Hamiltonian solved here: H = -1/2 d^2/dx^2 + 1/2 x^2")
    print("  Exact full-line spectrum: E_n = n + 1/2")
    print()
    print(f"{'n':>3s}  {'numerical':>18s}  {'theory':>18s}  {'num-theory':>14s}")
    for n, (E_num, E_th) in enumerate(zip(result.eigenvalues, theory)):
        diff = E_num - E_th
        print(f"{n:3d}  {E_num:18.12f}  {E_th:18.12f}  {diff:14.6e}")

    diffs = result.eigenvalues - theory
    if np.all(diffs > 0.0):
        print("\nAll reported numerical eigenvalues are above the full-line theoretical values.")
    else:
        print("\nThe reported numerical eigenvalues are not all above the full-line theoretical values.")

    print(
        "This upward shift is expected because the computation is performed in a restricted "
        "trial space: the wavefunctions are forced to satisfy Dirichlet boundary conditions "
        "at x = ±L/2. The true harmonic-oscillator eigenfunctions live on the whole line and "
        "do not vanish exactly at these finite endpoints. Restricting the admissible linear "
        "space raises the variational eigenvalues, so the boxed problem gives energies that "
        "are slightly higher than the exact full-line harmonic-oscillator spectrum."
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



def main() -> QM1DResult:
    args = build_parser().parse_args()
    if args.test_zero:
        return run_zero_potential_test()
    
    if args.test_harmonic:
        return run_harmonic_oscillator_test(show_plot=args.show_plot, output_pdf=args.output_pdf)
    raise SystemExit("Please choose a test to run: --test_zero or --test_harmonic")


if __name__ == "__main__":
    main()

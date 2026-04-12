# QM1D Toy Solver

A small educational Python code for learning 1D quantum mechanics and simple solid-state physics ideas.

## Overview

This project solves simple **1D stationary Schrödinger problems** on a finite real-space grid using a finite-difference Hamiltonian and a sparse eigensolver. It is meant as a **toy / learning code**, not a production simulation package.

The code uses the **atomic unit system**:
- length: **Bohr**
- energy and potential: **Hartree**

## Current functionality and limits

This code is intentionally small and simplified. It is designed specifically for **1D stationary Schrödinger problems** and currently supports the following two solver modes.

### One-electron feature

The one-electron solver handles a single particle in a user-supplied 1D external potential. It computes several low-lying eigenvalues and eigenstates on a finite-difference real-space grid, and it can generate a PDF plot of the potential, orbitals, and eigenvalue spectrum.

Supported boundary conditions for the one-electron solver:
- `dirichlet`
- `periodic`
- `bloch`

For Bloch calculations, you may specify either:
- `--kpt` in units of 1/Bohr, or
- `--relative_kpt`, where the physical wavevector is `relative_kpt * (2*pi/L)`

Example:

```bash
python main.py --potential_expr "sin(x*2*pi/12)" --n_states 12 --L 12.0 --h_target 0.05 --boundary bloch --relative_kpt 0.25
```

Current one-electron limits:
- only the stationary Schrödinger equation is supported
- the potential must be given through the implemented expression parser and allowed names

### Two-electron feature

The two-electron solver is an **exact grid-based solver for two interacting electrons in 1D**. It works with a spatial two-electron wavefunction
`Ψ(x₁, x₂)` and solves the Hamiltonian consisting of:
- kinetic energy of electron 1
- kinetic energy of electron 2
- the external one-body potential acting on each electron
- a softened electron-electron interaction

The electron-electron repulsion is modeled as:

```text
1 / sqrt((x1 - x2)^2 + a^2)
```

where `a` is the softening parameter set by `--interaction_softening`.

The two-electron solver supports two exchange-symmetry sectors through `--spin_symmetry`:
- `singlet`: spatial wavefunction is symmetric
- `triplet`: spatial wavefunction is antisymmetric

Example:

```bash
python main.py --potential_expr "0.5*x**2" --num_electrons 2 --spin_symmetry triplet --interaction_softening 1.0
```

Current two-electron limits:
- only `boundary=dirichlet` is currently implemented for the exact two-electron solver
- only `num_electrons = 2` is supported in this mode
- `spin_symmetry` must be provided as `singlet` or `triplet`
- Bloch and periodic boundary conditions are not currently supported in the exact two-electron solver

## Code structure

- `main.py`  
  Command-line entry point. Defines the built-in examples, parses input flags, and launches runs.

- `qm1d_solver.py`  
  Main solver logic. Contains configuration classes, grid generation, potential-expression validation, eigensolver workflow, normalization, and orbital reconstruction.

- `hamiltonian.py`  
  Builds the finite-difference Hamiltonian operator for the supported boundary-condition implementations.

- `plotting.py`  
  Generates the figure showing the potential, orbitals, and eigenvalue spectrum, and saves it as a PDF.

## Requirements

Recommended environment:
- Python 3.10+

Required packages:
- `numpy`
- `scipy`
- `matplotlib`

## Installation

Install the required Python packages with:

```bash
pip install numpy scipy matplotlib
```

## How to run

The main entry point is:

```bash
python main.py [flags]
```

### Built-in examples

Run the zero-potential box example:

```bash
python main.py --test_zero
```

Run the harmonic-oscillator example and save the figure:

```bash
python main.py --test_harmonic --output_pdf harmonic_oscillator_orbitals.pdf
```

### User-defined potential

Run with a custom potential expression:

```bash
python main.py --potential_expr "0.5*x**2"
```

Example with extra numerical settings:

```bash
python main.py --potential_expr "sin(x) + 0.1*x**2" --L 10 --h_target 0.05 --fd_order 6 --n_states 4 --boundary dirichlet --output_pdf user_input_potential_orbitals.pdf
```

## Input arguments

This README does not list every flag in detail.

For the full command-line interface, use:

```bash
python main.py --help
```

You can also read the argument definitions directly in:
- `main.py` for CLI flags and defaults

### Potential expression input

The potential is provided as a Python-style expression string through `--potential_expr`.

Examples:

```bash
python main.py --potential_expr "sin(x) + 0.1*x**2"
```

This README does not enumerate every allowed function or validation rule.

For the exact implementation details, see:
- `_ALLOWED_MATH_FUNCS` in `qm1d_solver.py`
- potential-expression validation and parsing logic in `qm1d_solver.py`

## Output

A typical run produces:
- a terminal summary of the numerical setup
- eigenvalues printed in **Hartree**
- for built-in tests, a comparison against the corresponding reference energies
- for one-electron runs, a PDF figure containing the potential, wavefunctions, and eigenvalue spectrum
- for two-electron runs, a PDF figure containing a two-particle density heat map, one-particle densities, and the eigenvalue spectrum

Use `--show_plot` if you also want the plot window to appear interactively.

## Notes

- The user-input run requires `--potential_expr` unless you are using `--test_zero` or `--test_harmonic`.
- The computational interval is built from `L` on a symmetric grid from `-L/2` to `L/2`.

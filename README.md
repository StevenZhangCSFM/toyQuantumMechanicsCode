# QM1D Toy Solver

A small educational Python code for learning 1D quantum mechanics and simple solid-state physics ideas.

## Overview

This project solves simple **1D stationary Schrödinger problems** on a finite real-space grid using a finite-difference Hamiltonian and a sparse eigensolver. It is meant as a **toy / learning code**, not a production simulation package.

The code uses the **atomic unit system**:
- length: **Bohr**
- energy and potential: **Hartree**

## Current functionality

The current code can:
- solve 1D stationary eigenvalue problems for a user-supplied potential
- compute several low-lying eigenvalues and eigenstates
- plot the potential, orbitals, and eigenvalue spectrum
- save the plot to a PDF file

## Current limits

This code is intentionally small and simplified. Current limits include:

- **1D only**
- **stationary** Schrödinger equation only
- intended for learning and experimentation rather than research-scale calculations
- potential input is limited to the syntax and allowed names implemented in the source code

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

## Potential expression input

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
- a PDF plot containing the potential, orbitals, and eigenvalue spectrum

Use `--show_plot` if you also want the plot window to appear interactively.

## Notes

- The user-input run requires `--potential_expr` unless you are using `--test_zero` or `--test_harmonic`.
- The computational interval is built from `L` on a symmetric grid from `-L/2` to `L/2`.

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from qm1d_solver import QM1DResult


def plot_potential_and_orbitals(
    result: QM1DResult,
    output_pdf: str = "harmonic_oscillator_orbitals.pdf",
    show: bool = False,
) -> str:
    x = result.grid.x_full
    V = result.V_full
    orbitals = result.orbitals_full

    fig, ax_pot = plt.subplots(figsize=(9, 6))
    ax_orb = ax_pot.twinx()

    ax_pot.plot(x, V, color="red", linewidth=2.5, label="Potential V(x)")

    for idx in range(orbitals.shape[1]):
        ax_orb.plot(
            x,
            orbitals[:, idx],
            linestyle="--",
            linewidth=1.6,
            label=f"Orbital {idx + 1}",
        )

    ax_pot.set_xlabel("x (Bohr)")
    ax_pot.set_ylabel("Potential energy (Hartree)", color="red")
    ax_orb.set_ylabel("Orbital value")
    ax_pot.set_title("1D Harmonic Oscillator: Potential and Orbitals")

    ax_pot.tick_params(axis="y", labelcolor="red")
    ax_pot.grid(True, alpha=0.3)

    handles_pot, labels_pot = ax_pot.get_legend_handles_labels()
    handles_orb, labels_orb = ax_orb.get_legend_handles_labels()
    ax_pot.legend(handles_pot + handles_orb, labels_pot + labels_orb, loc="best")

    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_pdf

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from qm1d_solver import QM1DResult


def plot_potential_and_orbitals(
    result: QM1DResult,
    output_pdf: str = "user_input_potential_orbitals.pdf",
    show: bool = False,
) -> str:
    x = result.grid.x_full
    V = result.V_full
    orbitals = result.orbitals_full
    eigenvalues = np.asarray(result.eigenvalues, dtype=float)

    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.5, 1.0], hspace=0.08)

    ax_pot = fig.add_subplot(gs[0])
    ax_orb = ax_pot.twinx()
    ax_spec = fig.add_subplot(gs[1])

    ax_pot.plot(x, V, color="red", linewidth=2.5, label="Potential V(x)")

    orbital_colors: list[str] = []
    for idx in range(orbitals.shape[1]):
        (line,) = ax_orb.plot(
            x,
            orbitals[:, idx],
            linestyle="--",
            linewidth=1.6,
            label=f"Orbital {idx + 1}",
        )
        orbital_colors.append(line.get_color())

    for idx, eig in enumerate(eigenvalues):
        color = orbital_colors[idx] if idx < len(orbital_colors) else None
        ax_spec.vlines(eig, 0.2, 0.95, linewidth=2.0, color=color)
        ax_spec.plot(eig, 0.95, marker="o", markersize=4, color=color)

    ax_pot.set_ylabel("Potential energy (Hartree)", color="red")
    ax_orb.set_ylabel("Orbital value")
    ax_pot.set_title(output_pdf)
    ax_pot.tick_params(axis="y", labelcolor="red")
    ax_pot.grid(True, alpha=0.3)
    ax_pot.tick_params(axis="x", labelbottom=False)

    ax_spec.set_xlabel("Eigenvalue / Energy (Hartree)")
    ax_spec.set_yticks([])
    ax_spec.set_ylim(0.0, 1.1)
    ax_spec.grid(True, axis="x", alpha=0.3)
    ax_spec.spines["left"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)
    ax_spec.spines["top"].set_visible(False)

    if eigenvalues.size > 0:
        e_min = float(np.min(eigenvalues))
        e_max = float(np.max(eigenvalues))
        if np.isclose(e_min, e_max):
            pad = max(1.0, 0.05 * max(abs(e_min), 1.0))
        else:
            pad = 0.08 * (e_max - e_min)
        ax_spec.set_xlim(e_min - pad, e_max + pad)

    handles_pot, labels_pot = ax_pot.get_legend_handles_labels()
    handles_orb, labels_orb = ax_orb.get_legend_handles_labels()
    ax_pot.legend(handles_pot + handles_orb, labels_pot + labels_orb, loc="best")

    fig.align_ylabels([ax_pot, ax_orb])
    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_pdf

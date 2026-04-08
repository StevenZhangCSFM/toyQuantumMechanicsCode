from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from qm1d_solver import QM1DResult


class PlotterBase:
    def plot(self, result: QM1DResult, output_pdf: str, show: bool = False) -> str:
        raise NotImplementedError

    @staticmethod
    def _finalize(fig: plt.Figure, output_pdf: str, show: bool) -> str:
        fig.tight_layout()
        fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return output_pdf


class OneElectronPlotter(PlotterBase):
    def plot(self, result: QM1DResult, output_pdf: str, show: bool = False) -> str:
        x = result.grid.x_full
        V = result.V_full
        wavefunctions = result.wavefunctions_full
        eigenvalues = np.asarray(result.eigenvalues, dtype=float)

        fig = plt.figure(figsize=(9, 7))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.5, 1.0], hspace=0.08)

        ax_pot = fig.add_subplot(gs[0])
        ax_orb = ax_pot.twinx()
        ax_spec = fig.add_subplot(gs[1])

        ax_pot.plot(x, V, color="red", linewidth=2.5, label="Potential V(x)")

        boundary = getattr(result.config, "boundary", "dirichlet")
        bc_marker = "s" if boundary == "dirichlet" else "o"
        ax_pot.plot(
            [0.0, 1.0],
            [0.0, 0.0],
            linestyle="None",
            marker=bc_marker,
            markersize=7,
            color="black",
            transform=ax_pot.transAxes,
            clip_on=False,
        )

        wf_colors: list[str] = []
        for idx in range(wavefunctions.shape[1]):
            y = np.real(wavefunctions[:, idx]) if np.iscomplexobj(wavefunctions) else wavefunctions[:, idx]
            label = f"Re(Wavefunction {idx + 1})" if boundary == "bloch" else f"Wavefunction {idx + 1}"
            (line,) = ax_orb.plot(x, y, linestyle="--", linewidth=1.6, label=label)
            wf_colors.append(line.get_color())

        for idx, eig in enumerate(eigenvalues):
            color = wf_colors[idx] if idx < len(wf_colors) else None
            ax_spec.vlines(eig, 0.2, 0.95, linewidth=2.0, color=color)
            ax_spec.plot(eig, 0.95, marker="o", markersize=4, color=color)

        ax_pot.set_ylabel("Potential energy (Hartree)", color="red")
        ax_orb.set_ylabel("Re(wavefunction value)" if boundary == "bloch" else "Wavefunction value")
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
            pad = max(1.0, 0.05 * max(abs(e_min), 1.0)) if np.isclose(e_min, e_max) else 0.08 * (e_max - e_min)
            ax_spec.set_xlim(e_min - pad, e_max + pad)

        handles_pot, labels_pot = ax_pot.get_legend_handles_labels()
        handles_wf, labels_wf = ax_orb.get_legend_handles_labels()
        ax_pot.legend(handles_pot + handles_wf, labels_pot + labels_wf, loc="best")

        fig.align_ylabels([ax_pot, ax_orb])
        return self._finalize(fig, output_pdf, show)


class TwoElectronPlotter(PlotterBase):
    def plot(self, result: QM1DResult, output_pdf: str, show: bool = False) -> str:
        x = result.grid.x_full
        V = result.V_full
        densities_2d = result.densities_full
        one_particle = result.one_particle_densities_full
        eigenvalues = np.asarray(result.eigenvalues, dtype=float)

        fig = plt.figure(figsize=(11, 8))
        gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[4.0, 1.0], width_ratios=[1.15, 1.0], hspace=0.2, wspace=0.28)

        ax_heat = fig.add_subplot(gs[0, 0])
        ax_dens = fig.add_subplot(gs[0, 1])
        ax_spec = fig.add_subplot(gs[1, :])

        state_idx = 0
        heat = ax_heat.imshow(
            densities_2d[:, :, state_idx].T,
            origin="lower",
            extent=[x[0], x[-1], x[0], x[-1]],
            aspect="auto",
        )
        ax_heat.set_xlabel(r"$x_1$ (Bohr)")
        ax_heat.set_ylabel(r"$x_2$ (Bohr)")
        ax_heat.set_title(rf"$|\Psi_{{{state_idx + 1}}}(x_1,x_2)|^2$")
        fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04)

        ax_pot = ax_dens.twinx()
        density_colors: list[str] = []
        if one_particle is not None:
            for idx in range(one_particle.shape[1]):
                (line,) = ax_dens.plot(x, one_particle[:, idx], linewidth=1.8, label=f"Density {idx + 1}")
                density_colors.append(line.get_color())
        ax_pot.plot(x, V, color="red", linewidth=2.2, label="Potential V(x)")
        ax_dens.set_xlabel("x (Bohr)")
        ax_dens.set_ylabel("One-particle density")
        ax_pot.set_ylabel("Potential energy (Hartree)", color="red")
        ax_pot.tick_params(axis="y", labelcolor="red")
        ax_dens.grid(True, alpha=0.3)
        ax_dens.set_title("Marginal densities and potential")

        for idx, eig in enumerate(eigenvalues):
            color = density_colors[idx] if idx < len(density_colors) else None
            ax_spec.vlines(eig, 0.2, 0.95, linewidth=2.0, color=color)
            ax_spec.plot(eig, 0.95, marker="o", markersize=4, color=color)
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
            pad = max(1.0, 0.05 * max(abs(e_min), 1.0)) if np.isclose(e_min, e_max) else 0.08 * (e_max - e_min)
            ax_spec.set_xlim(e_min - pad, e_max + pad)

        handles_dens, labels_dens = ax_dens.get_legend_handles_labels()
        handles_pot, labels_pot = ax_pot.get_legend_handles_labels()
        ax_dens.legend(handles_dens + handles_pot, labels_dens + labels_pot, loc="best")
        fig.suptitle(output_pdf)
        return self._finalize(fig, output_pdf, show)



def plot_result(
    result: QM1DResult,
    output_pdf: str = "user_input_potential_orbitals.pdf",
    show: bool = False,
) -> str:
    plotter: PlotterBase = TwoElectronPlotter() if result.config.num_electrons == 2 else OneElectronPlotter()
    return plotter.plot(result, output_pdf=output_pdf, show=show)


# Backward-compatible name used by main.py.
def plot_potential_and_orbitals(
    result: QM1DResult,
    output_pdf: str = "user_input_potential_orbitals.pdf",
    show: bool = False,
) -> str:
    return plot_result(result, output_pdf=output_pdf, show=show)

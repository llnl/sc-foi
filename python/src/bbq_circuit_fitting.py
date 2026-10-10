"""
Perform transformations between a black box circuit and another circuit model
"""

import matplotlib.pyplot as plt
import numpy as np

from file_io import get_delimiter_from_file


# Abstract base classes, filled in later
class BlackBoxCircuit:
    pass


class BlackBoxAdmittanceModel:
    hfss_frequencies: np.ndarray
    hfss_admittances: np.ndarray

    mode_frequencies: np.ndarray
    mode_impedances: np.ndarray

    def __init__(
        self,
        admittance_filename: str,
        ljs: np.ndarray,
        n_modes: int,
        cj: float | None = None,
        plot_scales: dict[str, str] | None = None,
    ) -> None:
        """
        Convert HFSS admittances to mode parameters: capacitances, inductances,
        frequencies, and impedances as a function of junction inductances (ljs).

        admittance_filename: str
            path to the admittance frequency data

        ljs: np.ndarray
            array of junction inductances [H]

        n_modes: int
            number of modes to fit mode impedances

        cj: float | None (default: None)
            junction shunt self capacitance [F]

        plot_scales: dict
            dictionary of plot scales for x, y axes
        """

        self.ljs = ljs
        self.cj = cj if cj is not None else 0.0
        self.n_modes = n_modes
        self.plot_scales = (
            plot_scales if plot_scales is not None else {"x": "log", "y": "linear"}
        )
        self._read_admittance_file(admittance_filename)

    def _read_admittance_file(self, filename: str) -> None:
        """Read the real and imaginary parts of the admittance data from a file."""
        delimiter = get_delimiter_from_file(filename)
        data = np.genfromtxt(filename, delimiter=delimiter, names=True, autostrip=True)

        frequencies = data["frequency"]
        self.fscale = 1.0 if max(frequencies) > 1e9 else 1e9

        self.hfss_frequencies = self.fscale * frequencies
        self.hfss_admittances = data["rey11"] + 1j * data["imy11"]

    def fit_admittances(self) -> None:
        """Fit the admittances to extract the mode frequencies and impedances as
        function of the junction inductances."""
        zp = []
        wp = []

        w = 2 * np.pi * self.hfss_frequencies
        y_hfss = self.hfss_admittances

        for lj in self.ljs:
            yj = 1 / (1j * w * lj)
            y = y_hfss + yj + (1j * w * self.cj)

            # Zero crossings from the those with positive derivative
            zero_crossing_idxs = np.where(np.diff(np.sign(y.imag)) > 0)[0][
                : self.n_modes
            ]
            wp_j = w[zero_crossing_idxs]
            wp.append(wp_j)

            # Derivative of admittances at mode frequencies
            # gives the mode capacitances, inductances, impedances
            y_grad = np.gradient(y, np.median(np.diff(w)))
            cp = 0.5 * np.abs(y_grad.imag[zero_crossing_idxs])
            lp = 1.0 / (wp_j**2 * cp)
            zp.append(np.sqrt(lp / cp))

        self.mode_impedances = np.asarray(zp)
        self.mode_frequencies = np.asarray(wp)

    def plot_mode_impedances_vs_lj(
        self,
    ) -> None:

        _, ax = plt.subplots(1, 1, tight_layout=True)

        for p in range(self.n_modes):
            ax.plot(self.ljs, self.mode_impedances[p], label=rf"$Z_{p}$")

        ax.set_xlabel("Junction inductances [H]")
        ax.set_ylabel(r"Mode impedance [$\Omega$]")
        ax.set_xscale(self.plot_scales["x"])
        ax.set_yscale(self.plot_scales["y"])
        ax.legend()

    def run(self) -> None:

        self.fit_admittances()
        self.plot_mode_impedances_vs_lj()


class InductivelyCoupledSQUIDCircuit:
    """
    Couple a single SQUID shunted to ground to multiple resonators through
    inductive couplings.

    Parameters:
    ----------

    lr: np.ndarray
        resonator inductances [h]
    cr: np.ndarray
        resonator capacitances [f]
    lrq: np.ndarray
        resonator-qubit inductive couplings [f]
    lq: np.narray
        SQUID (juntion) inductances [H]
    cq: float
        SQUID (junction) capacitances [F]

    """

    def __init__(
        self, lr: np.ndarray, cr: np.ndarray, lrq: np.ndarray, lq: np.ndarray, cq: float
    ) -> None:

        self.lr = lr
        self.cr = cr
        self.lq = lq
        self.lrq = lrq
        self.cq = cq

    def _get_c_matrix(self) -> np.ndarray:
        return np.diag(np.concatenate((self.cr, self.cq)).flatten())

    def _get_inverse_l_matrix_lq(self, lq: float) -> np.ndarray:
        # Resonator diagonal terms
        lr_inv = np.diag(np.concatenate((1 / self.lr + 1 / self.lrq), 0).flatten())

        # Off diagonal terms
        lr_inv[-1, :-1] = -1 / self.lrq
        lr_inv[:-1, -1] = -1 / self.lrq

        # Junction diagonal term
        lr_inv[-1, -1] = 1 / lq + sum(1 / self.lrq)

        return lr_inv

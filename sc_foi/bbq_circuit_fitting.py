"""
Perform transformations between a black box circuit and another circuit model
"""

import matplotlib.pyplot as plt
import numpy as np
from file_io import get_delimiter_from_file
from lmfit.minimizer import Minimizer
from lmfit.parameter import Parameters


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
            plot_scales if plot_scales is not None else {"x": "log", "y": "log"}
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
            zp.append(2.0 / wp_j / np.abs(y_grad.imag[zero_crossing_idxs]))

        self.mode_impedances = np.asarray(zp)
        self.mode_frequencies = np.asarray(wp)

    def plot_open_circuit_admittance(self):

        _, ax = plt.subplots(1, 1, tight_layout=True)

        ax.plot(
            self.hfss_frequencies / self.fscale,
            self.hfss_admittances.imag,
            color="tab:blue",
            marker=".",
            ls="",
        )
        ax.axhline(y=0, color="k", alpha=0.5)
        ax.set_xlabel("Sweep frequencies [GHz]")
        ax.set_ylabel(r"${\rm Im}Y(\omega)$ [S]")
        ax.set_title("Open circuit admittance")

        ax.set_ylim(-1e-1, 1e-1)

    def plot_mode_frequencies_vs_lj(
        self,
    ) -> None:

        _, ax = plt.subplots(1, 1, tight_layout=True)

        for p in range(self.n_modes):
            ax.semilogx(
                self.ljs,
                self.mode_frequencies[:, p] / (2 * np.pi * self.fscale),
                marker="o",
                ls="",
                label=rf"$\omega_{p} / 2\pi$",
            )

        ax.set_xlabel("Junction inductances [H]")
        ax.set_ylabel(r"Mode frequencies [GHz]")
        ax.legend()
        ax.set_title("Mode frequencies from black box")

    def plot_mode_impedances_vs_lj(
        self,
    ) -> None:

        _, ax = plt.subplots(1, 1, tight_layout=True)

        for p in range(self.n_modes):
            ax.plot(
                self.ljs,
                self.mode_impedances[:, p],
                marker="o",
                ls="",
                label=rf"$Z_{p}$",
            )

        ax.set_xlabel("Junction inductances [H]")
        ax.set_ylabel(r"Mode impedance [$\Omega$]")
        ax.set_xscale(self.plot_scales["x"])
        ax.set_yscale(self.plot_scales["y"])
        ax.legend()
        ax.set_title("Mode impedances from black box")

    def show_plots(self) -> None:
        self.plot_mode_impedances_vs_lj()
        self.plot_mode_frequencies_vs_lj()
        self.plot_open_circuit_admittance()

    def run(self) -> None:

        self.fit_admittances()
        self.show_plots()


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

    def _get_a_matrix(self):
        """Compute the system matrix A = U Omega^2 U^t."""


class TwoCavityOneCouplerModel(Minimizer):
    # Inductance and capacitance bounds
    l_min = 0.1
    l_max = 2000
    l_med = 1
    c_min = 0.1e-3
    c_max = 2000e-3
    c_med = 1e-3

    default_unit_scale = 1e9

    plot_scales = {"x": "log", "y": "log"}

    def __init__(
        self,
        data: np.ndarray,
        x: np.ndarray,
        unit_scale: float | None = None,
    ):
        params = self._get_parameters()
        super().__init__(
            self._two_cavity_one_coupler_impedances_cost,
            params=params,
            fcn_args=(x,),
            fcn_kws={
                "data": data,
            },
        )
        self.unit_scale = (
            unit_scale if unit_scale is not None else self.default_unit_scale
        )
        self.data = data
        self.x = x

    def _get_parameters(self):
        params = Parameters()
        params.add("l1", value=0.1, min=self.l_min, max=self.l_max)
        params.add("l2", value=0.3, min=self.l_min, max=self.l_max)
        params.add("l1c", value=0.01, min=self.l_min, max=self.l_max)
        params.add("l2c", value=0.04, min=self.l_min, max=self.l_max)
        params.add("c1", value=300e-3, min=self.c_min, max=self.c_max)
        params.add("c2", value=100e-3, min=self.c_min, max=self.c_max)
        params.add("cj", value=2e-3, min=self.c_min, max=self.c_max)

        return params

    def _two_cavity_one_coupler_impedances_cost(
        self,
        parameters: Parameters,
        x: np.ndarray,
        data: np.ndarray,
    ) -> np.ndarray:

        pars_values = parameters.valuesdict()
        l1 = pars_values["l1"]
        l2 = pars_values["l2"]
        l1c = pars_values["l1c"]
        l2c = pars_values["l2c"]
        c1 = pars_values["c1"]
        c2 = pars_values["c2"]
        cj = pars_values["cj"]

        zp = []
        for lj in x:
            a_matrix = np.array(
                [
                    [(1 / l1 + 1 / l1c) / c1, 0, -1 / l1c / np.sqrt(cj * c1)],
                    [0, (1 / l2 + 1 / l2c) / c2, -1 / l2c / np.sqrt(cj * c2)],
                    [
                        -1 / l1c / np.sqrt(cj * c1),
                        -1 / l2c / np.sqrt(cj * c2),
                        (1 / l1 + 1 / l2 + 1 / lj) / cj,
                    ],
                ]
            )

            if not np.allclose(a_matrix, a_matrix.T):
                raise ValueError(f"a_matrix: {a_matrix} is not symmetric.")

            wsq, p_matrix = np.linalg.eigh(a_matrix)

            if any(wsq < 0):
                print(f"Negative eigenvalue detected: {wsq[wsq < 0] = }")
                print(f"{a_matrix}")

            zp.append(p_matrix[:, -1] ** 2 / cj / np.sqrt(np.abs(wsq)))

        return np.sum((np.asarray(zp) - data) ** 2, axis=1)

    def _get_zp_from_parameters(self, x: np.ndarray, l1: float, l2: float, l1c: float, l2c: float, c1: float, c2: float, cj: float, ) -> np.ndarray:

        zp = []
        for lj in self.x:
            a_matrix = np.array(
                [
                    [(1 / l1 + 1 / l1c) / c1, 0, -1 / l1c / np.sqrt(cj * c1)],
                    [0, (1 / l2 + 1 / l2c) / c2, -1 / l2c / np.sqrt(cj * c2)],
                    [
                        -1 / l1c / np.sqrt(cj * c1),
                        -1 / l2c / np.sqrt(cj * c2),
                        (1 / l1 + 1 / l2 + 1 / lj) / cj,
                    ],
                ]
            )

            if not np.allclose(a_matrix, a_matrix.T):
                raise ValueError(f"a_matrix: {a_matrix} is not symmetric.")

            wsq, p_matrix = np.linalg.eigh(a_matrix)

            if any(wsq < 0):
                print(f"Negative eigenvalue detected: {wsq[wsq < 0] = }")
                print(f"{a_matrix}")

            zp.append(p_matrix[:, -1] ** 2 / cj / np.sqrt(np.abs(wsq)))

        return np.asarray(zp)

    def _parameters_to_floats(self, parameters: Parameters) -> tuple[float, float, float, float, float, float]:
        pars_values = parameters.valuesdict()
        l1 = pars_values["l1"]
        l2 = pars_values["l2"]
        l1c = pars_values["l1c"]
        l2c = pars_values["l2c"]
        c1 = pars_values["c1"]
        c2 = pars_values["c2"]
        cj = pars_values["cj"]

        return l1, l2, l1c, l2c, c1, c2, cj

    def plot_fit(
        self,
        params: Parameters,
    ) -> None:

        _, ax = plt.subplots(1, 1, tight_layout=True)

        print(f"{params = }")

        _, n_modes = self.data.shape
        l1, l2, l1c, l2c, c1, c2, cj = self._parameters_to_floats(parameters=params)
        fit_data = self._get_zp_from_parameters(self.x, l1, l2, l1c, l2c, c1, c2, cj)

        for p in range(n_modes):
            ax.plot(
                self.x,
                self.data[:, p],
                marker="o",
                ls="",
                label=rf"$Z_{{p, {{\rm data}}}}$",
                color=f"C{p}",
            )
            ax.plot(
                self.x,
                fit_data[:, p],
                label=rf"$Z_{{p, {{\rm fit}}}}$",
                color=f"C{p}",
            )

        ax.set_xlabel("Junction inductances [H]")
        ax.set_ylabel(r"Mode impedance [$\Omega$]")
        ax.set_xscale(self.plot_scales["x"])
        ax.set_yscale(self.plot_scales["y"])
        ax.legend()
        ax.set_title("Mode impedance circuit fit")
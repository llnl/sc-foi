"""
Perform transformations between a black box circuit and another circuit model
"""

import numpy as np


class BlackBoxCircuit:
    pass


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

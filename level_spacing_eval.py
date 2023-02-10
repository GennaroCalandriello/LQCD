import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy import sparse
import matplotlib.pyplot as plt
import time
import math


def spacing_predictions(eigen, spacing_kind, data="float"):

    """Calculate the spacing distributions:

    spacing_kind=1 return FN (Further Neighbour) distribution (return s/ mean, s);

    spacing_kind=2 return rude spacing e[i+1]-e[i]

    spacing_kind=3 return the Level Spacing Ratio (LSR)

    spacing_kind=4 return CN (Closest Neighbour) distribution (return s/ mean, s)"""

    n_evl = len(eigen)
    eigen = eigen.real
    eigen = np.sort(eigen)

    if spacing_kind == 1:

        spacing = []

        for e in range(1, n_evl - 1):
            spacing.append(max(eigen[e + 1] - eigen[e], eigen[e] - eigen[e - 1]))

        mean = np.mean(np.array(spacing))

        return np.array(spacing) / mean, np.array(spacing)

    if spacing_kind == 2:

        s_n = 0
        s_n_minus_1 = 0
        r_n = 0
        r_tilde = []

        for k in range(1, n_evl - 1):
            s_n = eigen[k + 1] - eigen[k]
            s_n_minus_1 = eigen[k] - eigen[k - 1]
            r_n = s_n / s_n_minus_1
            r_tilde.append(min(r_n, 1 / r_n))

        return np.array(r_tilde)

    if spacing_kind == 3:
        # level spacing ration
        s_CN = 0
        s_FN = 0
        ratio = []

        for i in range(1, n_evl - 1):
            s_CN = min(eigen[i + 1] - eigen[i], eigen[i] - eigen[i - 1])
            s_FN = max(eigen[i + 1] - eigen[i], eigen[i] - eigen[i - 1])
            ratio.append(s_CN / s_FN)

        return np.array(ratio)

    if spacing_kind == 4:
        s = np.zeros(n_evl - 2)

        for i in range(1, n_evl - 2):
            s[i] = min(eigen[i + 1] - eigen[i], eigen[i] - eigen[i - 1])
            mean = np.mean(np.array(s))

        return s / mean, s


def distribution(sp, kind):

    """Plot theoretical distributions of GSE, GOE, GUE ensemble distributions picking the min and max values of the spacing array
    calculated in the main program"""
    s = np.linspace(0, max(sp), len(sp))
    p = np.zeros(len(s))

    if kind == "GOE":
        for i in range(len(p)):
            p[i] = np.pi / 2 * s[i] * np.exp(-np.pi / 4 * s[i] ** 2)

    if kind == "GUE":
        for i in range(len(p)):
            p[i] = (32 / np.pi ** 2) * s[i] ** 2 * np.exp(-4 / np.pi * s[i] ** 2)

    if kind == "GSE":
        for i in range(len(p)):
            p[i] = (
                2 ** 18
                / (3 ** 6 * np.pi ** 3)
                * s[i] ** 4
                * np.exp(-(64 / (9 * np.pi)) * s[i] ** 2)
            )
    if kind == "Poisson":
        for i in range(len(p)):
            p[i] = np.exp(-s[i])

    if kind == "GOE FN":  # lasciamo perdere va...

        a = (27 / 8) * np.pi
        for i in range(len(p)):
            p[i] = (
                (a / np.pi)
                * s[i]
                * np.exp(-2 * a * s[i] ** 2)
                * (
                    np.pi
                    * np.exp((3 * a / 2) * s[i] ** 2)
                    * (a * s[i] ** 2 - 3)
                    * (
                        math.erf(np.sqrt(a / 6) * s[i])
                        - math.erf(np.sqrt(3 * a / 2) * s[i])
                        + np.sqrt(6 * np.pi * a)
                        * s[i]
                        * (np.exp((4 * a / 3) * s[i] ** 2) - 3)
                    )
                )
            )

    return p

def unfolding_2_punto_0(eig, save=False, fit=False):

    len_analysis=len(eig)
    y = np.arange(len_analysis)

    
    poly = np.polyfit(eig[:len_analysis], y, 50)
    poly_y = np.poly1d(poly)(eig[:len_analysis])
    
    if fit:
        plt.plot(eig[:len_analysis], poly_y, c="red", linestyle="--", label="Unfolding")
        plt.step(eig[:len_analysis], y)
        plt.xlabel("E", fontsize=15)
        plt.ylabel("N(E)", fontsize=15)
        plt.legend()
        plt.show()

    if save:
        np.savetxt(f"unfolded_spectrum_Dirac.txt", poly_y)
    return poly_y


def density_of_states(eigenvalues):
    E0 = eigenvalues[0]
    dn_dE = []
    rho = []
    N = len(eigenvalues)
    prefactor = 1 / (np.pi * np.sqrt(N) * E0)
    for e in eigenvalues:
        dn_dE.append(1 / (2 * E0) * (np.sqrt(1 / (e / E0))))
        rho.append(prefactor * np.sqrt(1 - (e ** 2 / (4 * E0 ** 2 * N))))

    return np.array(dn_dE)


if __name__ == "__main__":

    eig = np.loadtxt("eigenvalues_Henon.txt", dtype=complex)
    dndE = density_of_states(eig.real)
    plt.plot(eig / eig[0], dndE)
    plt.show()

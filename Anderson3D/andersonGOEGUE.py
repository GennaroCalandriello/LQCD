import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as lnlg
from numba import njit
from module.functions import *

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def Hamiltonian2D(L, W, magneticfield):
    """Generate the Hamiltonian matrix in TWO DIMENSIONS for one realization of the random disorder. The value
    of magneticfield is the flux of B added as a phase factor to the hopping terms.
    The presence of B switches the ULSD from GOE to GUE."""

    H = np.zeros((L * L, L * L), dtype=complex)
    for i in range(L):
        for j in range(L):
            neighbors = [
                (i - 1, j),
                (i + 1, j),
                (i, j - 1),
                (i, j + 1),
            ]
            neighbors = [(n_i % L, n_j % L) for n_i, n_j in neighbors]
            expo = np.exp(
                -2 * 1j * np.pi * magneticfield
            )  # add a phase factor depending on the magnetic flux value
            H[i * L + j, i * L + j] = (
                np.random.uniform(-W / 2, W / 2) * expo
            )  # diagonal elements H[i * L + j, i * L +
            for n_i, n_j in neighbors:
                H[i * L + j, n_i * L + n_j] = -1
                H[n_i * L + n_j, i * L + j] = -1
    return H


@njit()
def Hamiltonian3D(magneticfield=1.4):
    """-----------------Hamiltonian----------------------------------------
    Generates the Hamiltonian of the system
    For the GOE case set the magneticfield to 0.0
    For the GUE case set the magneticfield to another value

    njit() test: passed

    -----------------------------------------------------------------------"""

    H = np.zeros((L * L * L, L * L * L), dtype=np.complex128)  # Initialize Hamiltonian
    # On-site disorder term

    for i in range(L):
        for j in range(L):
            for k in range(L):
                idx = i * L * L + j * L + k  # Calculate the index in the Hamiltonian
                # Random disorder term
                disorder = np.random.uniform(-W / 2, W / 2)
                H[idx, idx] = disorder  # on diagonal correct

    # Nearest neighbors
    for i in range(L):
        for j in range(L):
            for k in range(L):
                exp_phi = np.exp(
                    -2 * 1j * np.pi * magneticfield
                )  # add a phase factor depending on the magnetic flux value
                neighbors = [
                    (i - 1, j, k),
                    (i + 1, j, k),
                    (i, j - 1, k),
                    (i, j + 1, k),
                    (i, j, k - 1),
                    (i, j, k + 1),
                ]
                neighbors = [(n_i % L, n_j % L, n_k % L) for n_i, n_j, n_k in neighbors]

                count = 0
                for nx, ny, nz in neighbors:
                    count += 1
                    # if (
                    #     count % 1 == 0
                    #     or count % 2 == 0
                    #     or count % 3 == 0
                    #     or count % 4 == 0
                    # ):
                    #     exp_phi = 1

                    idx1 = i * L * L + j * L + k
                    idx2 = nx * L * L + ny * L + nz

                    H[idx1, idx2] = -1 * exp_phi
    return H


def main():
    variousB = False

    if not variousB:
        # H = Hamiltonian2D(L, W, magneticfield=0.0)
        H = Hamiltonian3D()
        print("Hamiltonian generated", H.shape)

        # Diagonalize it
        (energy_levels, eigenstates) = linalg.eigh(H)

        unfolded = unfold_spectrum(energy_levels)[1]

        p = distribution(unfolded, "GOE")
        plt.title("ULSD for unitary Anderson Hamiltonian", fontsize=16)
        plt.xlabel("s", fontsize=12)
        plt.ylabel(r"p(s)", fontsize=12)
        plt.legend()
        plt.hist(
            unfolded,
            bins=FreedmanDiaconis(unfolded),
            density=True,
            color="deepskyblue",
            histtype="step",
            label=f"unfolded spectrum L={L}",
        )
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            distribution(unfolded, "GSE"),
            color="lightgray",
            linestyle="--",
            label="GSE",
        )
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            distribution(unfolded, "GUE"),
            color="deepskyblue",
            linestyle="--",
            label="GUE",
        )
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            distribution(unfolded, "GOE"),
            color="grey",
            linestyle="--",
            label="GOE",
        )
        plt.legend()
        plt.show()

    else:
        magnetic = np.linspace(0.0, 0.1, 2)

        plt.figure()
        for magneticfield in magnetic:
            H = Hamiltonian2D(L, W, magneticfield=magneticfield)
            (energy_levels, eigenstates) = linalg.eigh(H)

            unfolded = unfold_spectrum(energy_levels)[1]

            num_bins = FreedmanDiaconis(unfolded)

            plt.hist(
                unfolded,
                bins=num_bins,
                density=True,
                histtype="step",
                color=np.random.randint(0, 255, 3) / 255,
                label=r"ULSD for $\varphi = $" f"{round(magneticfield, 3)}",
            )
        p = distribution(unfolded, "GUE")
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            p,
            "lightgray--",
            color=np.random.randint(0, 255, 3) / 255,
            label="GUE",
        )
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            distribution(unfolded, "GSE"),
            "lightgray--",
            color=np.random.randint(0, 255, 3) / 255,
            label="GSE",
        )
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            distribution(unfolded, "GOE"),
            "b--",
            label="GOE",
            color=np.random.randint(0, 255, 3) / 255,
        )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()

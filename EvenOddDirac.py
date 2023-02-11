import numpy as np

from utils.gamma import *
from utils.params import *
from test_lavoroqui import *

"""References:
1. Luscher: "Implementation of the lattice Dirac operator" """


def FieldStrength(U, x, y, z, t, indexinverter):

    # Q = np.array(
    #     ((0 + 0j, 0 + 0j, 0 + 0j), (0 + 0j, 0 + 0j, 0 + 0j), (0 + 0j, 0 + 0j, 0 + 0j))
    # )
    Q = np.zeros((4, 4, su3, su3), dtype=complex)

    for mu in range(4):
        a_mu = [0, 0, 0, 0]
        a_mu[mu] = 1
        for nu in range(4):
            a_nu = [0, 0, 0, 0]
            a_nu[nu] = 1

            if (
                indexinverter
            ):  # sta cosa va controllata meglio, sennò conviene riscriverlo
                mu = nu
                a_nu = a_mu

            Q[mu, nu] = (
                U[x, y, z, t, mu]
                @ U[
                    (x + a_mu[0]) % Nx,
                    (y + a_mu[1]) % Ny,
                    (z + a_mu[2]) % Nz,
                    (t + a_mu[3]) % Nt,
                    nu,
                ]
                @ U[
                    (x + a_nu[0]) % Nx,
                    (y + a_nu[1]) % Ny,
                    (z + a_nu[2]) % Nz,
                    (t + a_nu[3]) % Nt,
                    mu,
                ]
                .conj()
                .T
                @ U[x, y, z, t, nu].conj().T
                + U[x, y, z, t, nu]
                @ U[
                    (x - a_mu[0] + a_nu[0]) % Nx,
                    (y - a_mu[1] + a_nu[1]) % Ny,
                    (z - a_mu[2] + a_nu[2]) % Nz,
                    (t - a_mu[3] + a_nu[3]) % Nt,
                    mu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_mu[0]) % Nx,
                    (y - a_mu[1]) % Ny,
                    (z - a_mu[2]) % Nz,
                    (t - a_mu[3]) % Nt,
                    nu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_mu[0]) % Nx,
                    (y - a_mu[1]) % Ny,
                    (z - a_mu[2]) % Nz,
                    (t - a_mu[3]) % Nt,
                    mu,
                ]
                + U[
                    (x - a_mu[0]) % Nx,
                    (y - a_mu[1]) % Ny,
                    (z - a_mu[2]) % Nz,
                    (t - a_mu[3]) % Nt,
                    mu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_mu[0] - a_nu[0]) % Nx,
                    (y - a_mu[1] - a_nu[1]) % Ny,
                    (z - a_mu[2] - a_nu[2]) % Nz,
                    (t - a_mu[3] - a_nu[3]) % Nt,
                    nu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_mu[0] - a_nu[0]) % Nx,
                    (y - a_mu[1] - a_nu[1]) % Ny,
                    (z - a_mu[2] - a_nu[2]) % Nz,
                    (t - a_mu[3] - a_nu[3]) % Nt,
                    mu,
                ]
                @ U[
                    (x - a_nu[0]) % Nx,
                    (y - a_nu[1]) % Ny,
                    (z - a_nu[2]) % Nz,
                    (t - a_nu[3]) % Nt,
                    nu,
                ]
                + U[
                    (x - a_nu[0]) % Nx,
                    (y - a_nu[1]) % Ny,
                    (z - a_nu[2]) % Nz,
                    (t - a_nu[3]) % Nt,
                    nu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_nu[0]) % Nx,
                    (y - a_nu[1]) % Ny,
                    (z - a_nu[2]) % Nz,
                    (t - a_nu[3]) % Nt,
                    mu,
                ]
                @ U[
                    (x + a_mu[0] - a_nu[0]) % Nx,
                    (y + a_mu[1] - a_nu[1]) % Ny,
                    (z + a_mu[2] - a_nu[2]) % Nz,
                    (t + a_mu[3] - a_nu[3]) % Nt,
                    nu,
                ]
                @ U[x, y, z, t, mu].conj().T
            )
    return Q


def Fmunu(U, x, y, z, t):

    Qmunu = FieldStrength(U, x, y, z, t, indexinverter=False)
    Qnumu = FieldStrength(U, x, y, z, t, indexinverter=True)

    Fmunu = 1 / 8 * (Qmunu - Qnumu)

    return Fmunu


def DiracNaive(U, psi):

    """Simile a quella già scritta"""

    return 1


if __name__ == "__main__":

    U = initializefield(np.zeros((Nx, Ny, Nt, Nz, 4, su3, su3), complex))
    # Q = FieldStrength(U, 0, 0, 1, 0, False)
    F = Fmunu(U, 0, 0, 2, 3)
    print(len(F[0, 2].flatten()))


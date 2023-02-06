import numpy as np
from numba import njit

from module.functions import *
from module.functionsu3 import *

"""These are execrises to comprise how to construct the object that one will use in simulations"""

N = 5
Nx, Ny, Nz, Nt = N, N, N, N  # possibility to asymmetric time extensions

sx = np.array(((0, 1), (1, 0)), complex)
sy = np.array(((0, -1j), (1j, 0)), complex)
sz = np.array(((1, 0), (0, -1)), complex)
s0 = np.identity(su2) + 0j

# Define the gamma matrices
gamma = np.zeros((4, 2, 2), dtype=np.complex128)
gamma[0, :, :] = np.array([[0, 1], [1, 0]])
gamma[1, :, :] = np.array([[0, -1j], [1j, 0]])
gamma[2, :, :] = np.array([[1, 0], [0, -1]])
gamma[3, :, :] = np.array([[1j, 0], [0, -1j]])


gauge = np.zeros(((Nx, Ny, Nz, Nt, 4, su2, su2)), dtype=complex)
fermion2flavor = np.zeros(
    ((Nx, Ny, Nz, Nt, 2, 2)), dtype=complex
)  # fermion field with 2 flavor and 4-spin components
D_up = np.zeros((Nx, Ny, Nz, Nt, 2, 2), dtype=complex)
D_down = D_up.copy()


@njit()
def initializeconfiguration(gauge, chi):

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    for mu in range(4):
                        gauge[x, y, z, t, mu] = SU2SingleMatrix()
                    for flavor in range(2):
                        chi[x, y, z, t, flavor, 0] = complex(
                            np.random.rand(), np.random.rand()
                        )
                        chi[x, y, z, t, flavor, 1] = complex(
                            np.random.rand(), np.random.rand()
                        )

    return gauge, chi


@njit()
def FermionMatrix(D_up, D_down, chi, gauge):

    fermion_up = chi[
        :, :, :, :, :, 0
    ]  # the specific relation between the two spins depend on the characteristic of simulation
    fermion_down = chi[
        :, :, :, :, :, 1
    ]  # here the two spin components of flavors are decoupled

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    for mu in range(4):
                        for m in range(2):
                            for n in range(2):
                                # this naive matrix is wrong, should be recontrolled
                                D_up[x, y, z, t, m, n] = np.sum(
                                    gamma[mu, m, :]
                                    * (
                                        fermion_up[x, y, z, t, :]
                                        - gauge[x, y, z, t, mu, m, :]
                                        * fermion_up[x, y, z, t, n]
                                    )
                                )
                                D_down[x, y, z, t, m, n] = np.sum(
                                    gamma[mu, m, :]
                                    * (
                                        fermion_down[x, y, z, t, :]
                                        - gauge[x, y, z, t, mu, m, :]
                                        * fermion_down[x, y, z, t, n]
                                    )
                                )

    return D_up, D_down


###################################################################################################################
"""Example of quark fields with Dirac, flavor and color indices, here for SU(3)"""

"""References:
1. ChatGPT
2. Implementation of the lattice Dirac operator, Luscher"""

color = 3
Dirac = 4
su3 = 3


gamma_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
gamma_1 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]])
gamma_2 = np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]])
gamma_3 = np.array([[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]])

gamma = np.zeros(((4, 4, 4)), dtype=complex)

gamma[0] = gamma_0
gamma[1] = gamma_1
gamma[2] = gamma_2
gamma[3] = gamma_3


@njit()
def quarkfield(quark, forloops):

    """Each flavor of fermion (quark) has 4 space-time indices, 3 color
     indices and 4 spinor (Dirac) indices"""

    if forloops:
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    for t in range(Nt):
                        for spinor in range(Dirac):
                            for Nc in range(color):
                                quark[x, y, z, t, spinor, Nc] = complex(
                                    np.random.rand(), np.random.rand()
                                )
    # oppure più banalmente
    if not forloops:
        quark = np.random.rand(Nx, Ny, Nz, Nt, Dirac, color) + 1j * np.random.rand(
            Nx, Ny, Nz, Nt, color, Dirac
        )

    return quark


@njit()
def initializefield(gaugesu3):

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    for mu in range(4):
                        gaugesu3[x, y, z, t, mu] = SU3SingleMatrix()

    return gaugesu3


# Ogni quark ha in totale 4*3=12 componenti indipendenti
# i quarks si accoppiano al gauge tramite gli indici di colore
# la matrice di Dirac ha dimensione 4x4x3x3 dipendentemente dagli indici di colore (3) e da quelli spinoriali (4)?
# quarks solo 1 indice spinoriale (che va da 1 a 4)
# le matrici gamma accoppiano gli spinori con gli indici spaziotempo e gli indici di colore con il gauge
# gli indici spinoriali sono contratti con quelli spaziotemporali
# The interaction between quarks and gluons is described by the coupling between the Gamma matrices and the gauge field.
# dimension of quark fields: 4 x 3 x Nx x Ny x Nz x Nt
# dimension of Dirac matrix on each lattice site is 4x4
# direction mu and spinor indices are in the same for loops

# In a lattice gauge theory, on each lattice site, the Dirac matrix indices are 4 spinor indices and 4 space-time indices (mu = 1, ..., 4), representing the components of the Dirac spinor and the direction of the space-time derivatives, respectively.
# The spinor indices run from 1 to 4 and represent the 4 components of a Dirac spinor, while the space-time indices indicate the direction of the space-time derivatives in the lattice theory.
# The Dirac matrix couples with the gauge field through the color indices, which are also present in the theory and run from 1 to 3.


def Dirac_matrix(u, quarks, i, j, k, l):
    # esempio approssimativo, errato, da qui costruisco l'espressione giusta
    # Initialize the Dirac matrix for a single lattice site
    dirac_matrix = np.zeros((4, 4), dtype=complex)

    # Loop over the spinor indices
    for mu in range(4):
        for nu in range(4):
            # Loop over the color indices
            for a in range(3):
                for b in range(3):
                    # Coupling between gamma matrices, quarks, and gauge field
                    dirac_matrix[mu, nu] += (
                        quarks[i, j, k, l, mu, a]
                        * u[i, j, k, l, mu, a, b]
                        * gamma[mu, nu]
                        * quarks[i, j, k, l, nu, b].conj()
                    )

    return dirac_matrix


# @njit()
def DiracMatrixLattice(U, q, dirac_matrix):
    # non ne sono per nulla sicuro, ma sono test per ora
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    for mu in range(4):  # directions
                        for i in range(Dirac):  # spinor index
                            for j in range(color):  # color
                                for k in range(color):
                                    for l in range(color):
                                        # reference 1. example
                                        dirac_matrix[x, y, z, t, mu, i, j, k] += 0.5 * (
                                            np.identity(4, dtype=complex)[i, l]
                                            * np.identity(3, dtype=complex)[j, k]
                                            + U[x, y, z, t, mu, j, l]
                                            * gamma[mu][i, l]
                                            * np.identity(3, dtype=complex)[j, k]
                                        )
                                #         # dirac_matrix[x, y, z, t, mu, i, j, k] += 0.5 * (
                                #         #     U[x, y, z, t, mu, j, l]
                                #         #     * (1 - gamma[mu][i, l])
                                #         #     * q[x, y, z, t, i, j]
                                #         # )

    print(dirac_matrix)


if __name__ == "__main__":

    gaugesu3 = np.zeros((Nx, Ny, Nz, Nt, 4, su3, su3), dtype=complex)
    quark = np.zeros((Nx, Ny, Nz, Nt, Dirac, color), dtype=complex)
    dirac_matrix = np.zeros((Nx, Ny, Nz, Nt, Dirac, Dirac, color, color), dtype=complex)
    # the Dirac matrix is 12*12 on each lattice point, it is in agreement with ref. 2 -> Dirac M. decomposed in 4 blocks
    # each bloc is 6x6 for a total of 144 independent elements on each lattice site

    U = initializefield(gaugesu3)
    quark = quarkfield(quark, True)

    dirac = DiracMatrixLattice(U, quark, dirac_matrix)


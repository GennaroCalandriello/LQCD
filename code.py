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
colour = 3
Dirac = 4
su3 = 3
quark = np.zeros((Nx, Ny, Nz, Nt, colour, Dirac))

gamma_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
gamma_1 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]])
gamma_2 = np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]])
gamma_3 = np.array([[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]])


@njit()
def quarkfield(quark: np.array, forloops: bool):

    """Each flavor of fermion (quark) has 4 space-time indices, 3 color
     indices and 4 spinor (Dirac) indices"""

    forloops = False
    if forloops:
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    for t in range(Nt):
                        for Nc in range(colour):
                            for spinor in range(Dirac):
                                quark[x, y, z, t, Nc, spinor] = complex(
                                    np.random.rand(), np.random.rand()
                                )
    # oppure più banalmente
    if not forloops:
        quark = np.random.rand(Nx, Ny, Nz, Nt, colour, Dirac) + 1j * np.random.rand(
            Nx, Ny, Nz, Nt, colour, Dirac
        )

    return quark


gaugesu3 = np.zeros((Nx, Ny, Nz, Nt, 4, su3, su3))


@njit()
def initializefield(gaugesu3):

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    for mu in range(4):
                        gaugesu3 = SU3SingleMatrix()

    return gaugesu3


quark = quarkfield(quark, False)

# Ogni quark ha in totale 4*3=12 componenti indipendenti
# i quarks si accoppiano al gauge tramite gli indici di colore
# la matrice di Dirac ha dimensione 4x4x3x3 dipendentemente dagli indici di colore (3) e da quelli spinoriali (4)?
# quarks solo 1 indice spinoriale (che va da 1 a 4)
# le matrici gamma accoppiano gli spinori con gli indici spaziotempo e gli indici di colore con il gauge
# gli indici spinoriali sono contratti con quelli spaziotemporali
# The interaction between quarks and gluons is described by the coupling between the Gamma matrices and the gauge field.


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


def DiracMatrixLattice():
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    for mu in range(4):  # qui ci deve essere la somma sulle direzioni
                        pass


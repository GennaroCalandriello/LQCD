import numpy as np
from numba import njit

from module.functionsu3 import *

# Define the gamma matrices
gamma_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
gamma_1 = np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]])
gamma_2 = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]])
gamma_3 = np.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]])

gamma = np.zeros(((4, 4, 4)), dtype=complex)

gamma[0] = gamma_0
gamma[1] = gamma_1
gamma[2] = gamma_2
gamma[3] = gamma_3

N = 5
Nx, Ny, Nz, Nt = N, N, N, N  # possibility to asymmetric time extensions
Dirac = 4
color = 3

# @njit()
def UgammaProd(gamma, U, res):
    for alpha in range(4):
        for beta in range(4):
            total = 0
            for a in range(3):
                for b in range(3):
                    total += gamma[alpha, beta] * (
                        U[a, b] * quark[alpha, a] - U[a, b] * quark[beta, b]
                    )
            res[alpha, beta] = total

    print(res)


def prova():
    x = np.random.rand(1, 8, 2, 8, 10)
    y = np.random.rand(8, 10, 10)
    z = np.einsum("nkctv,kvw->nctw", x, y)
    print(z)

    R = np.zeros((1, 2, 8, 10))

    for n in range(1):
        for c in range(2):
            for t in range(8):
                for w in range(10):
                    total = 0
                    # These are the variables to sum over
                    for v in range(10):
                        for k in range(8):
                            total += x[n, k, c, t, v] * y[k, v, w]
                    R[n, c, t, w] = total

    print(R)

    if z.any() == R.any():
        print("thats all folks")


res = np.zeros((4, 4), dtype=complex)
quark = np.random.rand(4, 3) + 1j * np.random.rand(4, 3)

############################################################################################
@njit()
def initializefield(U):

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    for mu in range(4):
                        U[x, y, z, t, mu] = SU3SingleMatrix() * complex(
                            np.random.rand(), np.random.rand()
                        )

    return U


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


def DiracMatrix(U, psi, D):

    # The Dirac matrix in lattice QCD is represented as a 4x4 complex matrix for each lattice site.

    m = 0.08
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    # D[x, y, z, t] += m
                    for alpha in range(4):
                        for beta in range(4):
                            Dtemp = 0
                            for a in range(3):
                                for b in range(3):
                                    for mu in range(4):

                                        a_mu = [0, 0, 0, 0]
                                        a_mu[mu] = 1
                                        Dtemp += 0.5 * (
                                            gamma[mu][alpha, beta]
                                            * (
                                                m
                                                + U[x, y, z, t, mu, a, b]
                                                * psi[
                                                    (x + a_mu[0]) % Nx,
                                                    (y + a_mu[1]) % Ny,
                                                    (z + a_mu[2]) % Nz,
                                                    (t + a_mu[3]) % Nt,
                                                    alpha,
                                                    a,
                                                ]
                                                - U[
                                                    (x - a_mu[0]) % Nx,
                                                    (y - a_mu[1]) % Ny,
                                                    (z - a_mu[2]) % Nz,
                                                    (t - a_mu[3]) % Nt,
                                                    mu,
                                                    b,
                                                    a,
                                                ]
                                                .conj()
                                                .T
                                                * psi[
                                                    (x - a_mu[0]) % Nx,
                                                    (y - a_mu[1]) % Ny,
                                                    (z - a_mu[2]) % Nz,
                                                    (t - a_mu[3]) % Nt,
                                                    beta,
                                                    b,
                                                ]
                                                .conj()
                                                .T
                                                * gamma[0][alpha, beta]
                                            )
                                        )

                            D[x, y, z, t, alpha, beta] = Dtemp
                    # D[x, y, z, t] = D[x, y, z, t] / np.linalg.norm(D[x, y, z, t])

    return D


def psibar(psi):
    psibar = psi.conj().T @ gamma[0]
    return psibar


def spacing(eigs):
    spac = []
    eigs = eigs.flatten()

    for i in range(1, len(eigs) - 1):
        spac.append(max(eigs[i + 1] - eigs[i], eigs[i] - eigs[i - 1]))

    return spac


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    psi = quarkfield(np.zeros((Nx, Ny, Nz, Nt, 4, 3), complex), True)
    U = initializefield(np.zeros((Nx, Ny, Nz, Nt, 4, 3, 3), complex))
    for _ in range(100):
        print("heatbath numero", _)
        U = HB_updating_links(5.8, U, N)

    Dirac = np.zeros((Nx, Ny, Nz, Nt, 4, 4), complex)
    D_nm = DiracMatrix(U, psi, Dirac)
    # print(D_nm)

    eigs, _ = np.linalg.eig(D_nm)
    print("number of eigenvalues: ", len(eigs.flatten()))

    spac = spacing(eigs.real)
    spac = np.array((spac))
    plt.figure()
    plt.hist(spac, 50, histtype="step")
    plt.show()

    print("mean spacing", np.mean(spac))

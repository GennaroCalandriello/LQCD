import numpy as np
from numba import njit, jit
import math
import matplotlib.pyplot as plt
import scipy.stats as stats


def get_key(x):
    return x[0]



def numba_sorted(arr):

    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if get_key(arr[j]) < get_key(arr[min_idx]):
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr


def sorting_and_ranking(eigenvalues):

    sorted_eigenvalues = sorted(eigenvalues, key=lambda x: x[0])
    # sorted_eigenvalues = numba_sorted(eigenvalues)
    rank = np.arange(0, len(sorted_eigenvalues[:]), 1)
    sorted_eigenvalues = np.array(sorted_eigenvalues)
    rank_list = []

    for i in range(len(rank)):
        rank_list.append([rank[i], (sorted_eigenvalues[i, 1])])

    return rank_list


def spacing_evaluation(ranked_ev):

    N_conf = len(ranked_ev[:])  # number of configurations

    spacing = []
    ranked_ev = np.array(ranked_ev)

    for j in range(N_conf):
        temp = []
        for i in range(N_conf):
            if ranked_ev[i, 1] == j:
                temp.append(ranked_ev[i])
        temp = np.array(temp)

        for i in range(1, len(temp) - 1):
            spacing.append(temp[i + 1, 0] - temp[i, 0])

    spacing = np.array(spacing) / N_conf
    spacing = spacing / np.mean(spacing)
    return spacing


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


def binning(eigenvalues, maxbins):

    bins = np.linspace(min(eigenvalues), max(eigenvalues), maxbins)
    eigenvalues = np.array(eigenvalues)
    digitized = np.digitize(eigenvalues, bins)
    binned_data = [eigenvalues[digitized == i] for i in range(1, len(bins))]

    return binned_data


def histogramFunction(sp, kind: str, bins, low, high):

    GUE = distribution(sp, "GUE")
    GSE = distribution(sp, "GSE")
    GOE = distribution(sp, "GOE")
    POISSON = distribution(sp, "Poisson")
    x = np.linspace(min(sp), max(sp), len(sp))

    std = np.std(sp)
    plt.figure()
    plt.xlabel("s", fontsize=15)
    plt.ylabel("P(s)", fontsize=15)
    plt.title(f"{kind}", fontsize=18)
    binned = np.array(binning(sp, bins + 1))
    errori = []

    for i in range(len(binned)):
        # length = len(binned)
        # std_err = 1 / (np.sqrt(length))*np.std(binned[i])
        std_err = stats.sem(binned[i])
        errori.append(std_err)

    counts, edges, _ = plt.hist(
        sp,
        bins=bins,
        density=True,
        histtype="step",
        fill=False,
        color=np.random.rand(3),
        label=r"$\lambda \in$" f"{round(low, 4)}; {round(high, 4)}",
    )
    # print("mi printi questo counts?", counts)
    # print("e mo printami i dati binnati", binned)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    # bin_width = edges[1] - edges[0]
    plt.errorbar(bin_centers, counts, yerr=errori, xerr=0, fmt=".", color="blue")

    plt.plot(x, GUE, "g--", label="GUE")
    # plt.plot(x, GSE, "b--", label="GSE")
    plt.plot(x, GOE, "r--", label="GOE")
    plt.plot(x, POISSON, "y--", label="Poisson")
    plt.legend()
    plt.show()

def Sturge(data):
    length = len(data)
    return round(1.5 * (np.log2(length) + 1))

def CreateRanges(data, num=10):

    emin=np.amin(np.abs(data[:, 4:204]))
    emax=np.amax(np.abs(data[:, 4:204]))
    ranger=np.linspace(emin, emax, num)

    rnglist=[]
    for i in range(len(ranger)-1):
        rnglist.append([round(ranger[i], 4), round(ranger[i+1], 4)])
    
    return np.array(rnglist), emin, emax

def IPR(data, kind):

    """Separate IPR from the other data and analyze it"""

    N_conf = len(data[:])
    Nt = 22  # time extension of the system
    ev = 200
    Nt_ev = Nt * ev

    IPR_values = data[:, 204 : Nt_ev + 204]  # select only IPR from data
    IPR_array = np.zeros((N_conf, 22 * 100))

    for i in range(N_conf):

        temp = []

        for j in range(
            0, Nt_ev, Nt * 2
        ):  # here I jump the IPR associated with negative (coupled) eigenvalues
            temp.extend(IPR_values[i, j : j + Nt])

        IPR_array[
            i
        ] = temp  # array of 100 positive eigenvalues*22 time slices (each \lamb for 22 Nt)

    IPR_sum = []

    for i in range(
        N_conf
    ):  # here I sum over all time slices for each \lam for each configuration
        for j in range(0, len(IPR_array[0]), Nt):
            sommo = sum((IPR_array[i, j : j + Nt]))
            IPR_sum.append(sommo)

    IPR_final = np.array(reshape_to_matrix(np.array(IPR_sum)))
    # for i in range(100):
    #       ###equivalent to the list comprehension below
    #     temp = []
    #     for j in range(N_conf):
    #         temp.append(IPR_final[j, i])
    #     IPR_mean.append(np.mean(temp))

    IPR_mean = [
        np.mean([x for x in row]) for row in IPR_final.T
    ]  # IPR mean for each \lambda along all conf

    eigenvalues = np.abs(data[:, 4:204])  # select eigenvalues
    eigenvalues = eigenvalues[:, ::2]  # eliminate chiral paired \lambda
    mean = [
        np.mean([x for x in row]) for row in eigenvalues.T
    ]  # find <\lambda> along all paths
    print("lunghezza ipr", len(IPR_mean))
    plt.figure()
    plt.title(r"PR vs $\lambda$" f" for {kind} phase", fontsize=13)
    plt.xlabel(r"$<\lambda>$", fontsize=15)
    plt.ylabel("<PR>", fontsize=15)
    plt.scatter(mean, IPR_mean, s=8, c="b")
    plt.legend()
    plt.show()

def reshape_to_matrix(array):

    shape = (round(len(array) / 100), 100)
    rows, cols = shape
    return [[array[i * cols + j] for j in range(cols)] for i in range(rows)]
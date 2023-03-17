import numpy as np
import pandas as pd
from numba import njit

from module.level_spacing_eval import *

"""Algorithm to analyze distributions:
1. Binning eigenvalues in a certain range
2. Compute ULSD on each bins
3. Plot"""


def loadtxt(newresults=True, topocool=False):

    """Structure of data:
    1st number: RHMC trajectory
    2nd number: 1 non ti interessa
    3rd number: 200 = eigenvalues calculated
    4th number: 0/1 quark up or down
    4:204 numbers: eigenvalues
    204:204+22*200: IPR"""

    if newresults:
        # new results sent 15/03/2023: this is all the statistics we have
        edeconfined = np.loadtxt("PhDresults\conf\confined_bz0\confined_bz0.txt")
        econfined = np.loadtxt("PhDresults\deconf\deconfined_bz0\deconfined_bz0.txt")

    else:
        edeconfined = np.loadtxt("PhDresults/deconf/deconfined/deconfined_bz0.txt")
        econfined = np.loadtxt("PhDresults/conf/confined/confined_bz0.txt")

    if topocool:

        TOPOLOGICALconf = np.loadtxt("PhDresults\conf\confined_bz0\TopoCool.txt")
        TOPOLOGICALdeconf = np.loadtxt("PhDresults\deconf\deconfined_bz0\TopoCool.txt")

        return edeconfined, econfined, TOPOLOGICALdeconf, TOPOLOGICALconf

    else:

        return edeconfined, econfined


def spacing_banale(eigs, kind="CN"):
    """Control and rewrite this!"""

    spacing = []
    rtilde = []
    eigs = np.sort(eigs)

    if kind == "rtilde":

        for i in range(1, len(eigs) - 1):
            sn = eigs[i + 1] - eigs[i]
            snm1 = eigs[i] - eigs[i - 1]
            rtilde.append(min(sn, snm1) / max(sn, snm1))

        return rtilde

    if kind == "naive":

        for i in range(1, len(eigs) - 1):
            spacing.append(eigs[i + 1] - eigs[i])

        return spacing

    if kind == "CN":

        for i in range(1, len(eigs) - 1):
            spacing.append(min(eigs[i + 1] - eigs[i], eigs[i] - eigs[i - 1]))

        return spacing / np.mean(spacing)


def ULSD(eigs):

    levelspacing = []

    for i in range(len(eigs)):
        unfold = unfolding_2_punto_0(eigs[i])
        spacing = spacing_banale(unfold)
        levelspacing.extend(spacing)

    print("lunghe", len(levelspacing))
    return np.array(levelspacing)


def main(
    low, high, quark: int, confined,
):

    """This function extracts the eigenvalues in a certain range or examine the first or the last
    n_evl; quark should be 0 (down) or 1 (up)"""

    edeconfined, econfined = loadtxt()

    if confined:
        analysis = econfined
        kind = "Confined"
    else:
        analysis = edeconfined
        kind = "Deconfined"

    spacing = []
    eigenvalue_list = []

    for j in range(len(analysis[:, 0])):
        if analysis[j, 3] == quark:
            # eigenvalue_list = []
            for i in range(4, 204):
                if analysis[j, i] >= 0:  # or analysis[j, i] <= 0:
                    if analysis[j, i] <= high and analysis[j, i] >= low:
                        eigenvalue_list.append([analysis[j, i], int(j / 2)])

    ranked_ev = sorting_and_ranking(eigenvalue_list)
    spacing = spacing_evaluation(ranked_ev)
    return spacing


def sorting_and_ranking(eigenvalues):

    sorted_eigenvalues = sorted(eigenvalues, key=lambda x: x[0])
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


def histogramMultipleFunction(sp, kind: str, bins, range_list):
    range_list = np.array(range_list)

    GUE = distribution(sp[2], "GUE")
    # GSE = distribution(sp, "GSE")
    # GOE = distribution(sp, "GOE")
    POISSON = distribution(sp[0], "Poisson")

    x = np.linspace(0, max(sp[0]), len(sp[0]))
    xgue = np.linspace(0, max(sp[2]), len(sp[2]))

    plt.figure()
    plt.title(f"{kind}")
    plt.hist(
        sp[0],
        bins=bins,
        density=True,
        histtype="step",
        fill=False,
        color="orange",
        label=r"$\lambda \in$"
        f"{round(range_list[0,0], 5)}; {round(range_list[0, 1], 3)}",
    )
    plt.hist(
        sp[1],
        bins=bins,
        density=True,
        histtype="step",
        fill=False,
        color="y",
        label=r"$\lambda \in$"
        f"{round(range_list[1, 0], 3)}; {round(range_list[1, 1], 3)}",
    )
    plt.hist(
        sp[2],
        bins=bins,
        density=True,
        histtype="step",
        fill=False,
        color="g",
        label=r"$\lambda \in$"
        f"{round(range_list[2, 0], 3)}; {round(range_list[2, 1], 3)}",
    )
    plt.hist(
        sp[3],
        bins=bins,
        density=True,
        histtype="step",
        fill=False,
        color="b",
        label=r"$\lambda \in$"
        f"{round(range_list[3, 0], 3)}{round(range_list[3, 1], 3)}",
    )
    plt.plot(xgue, GUE, "g--", label="GUE")
    # plt.plot(x, GSE, "b--", label="GSE")
    # plt.plot(x, GOE, "r--", label="GOE")
    plt.plot(x, POISSON, "y--", label="Poisson")
    plt.legend()
    plt.show()


def Sturge(data):
    length = len(data)
    return round(1.5 * (np.log2(length) + 1))


def print_equal_elements(array):
    element_counts = {}

    # Count the occurrences of each element
    for element in array:
        if element in element_counts:
            element_counts[element] += 1
        else:
            element_counts[element] = 1

    # Print elements with count greater than 1
    for element, count in element_counts.items():
        if count > 1:
            print(element)


def reshape_to_matrix(array):

    shape = (round(len(array) / 100), 100)
    rows, cols = shape
    return [[array[i * cols + j] for j in range(cols)] for i in range(rows)]


def IPR(confined: bool, quark: int):

    """Separate IPR from the other data and analyze it"""

    deconfinato, confinato = loadtxt()

    if confined:
        data = confinato
        kind = "Confined"
    else:
        data = deconfinato
        kind = "Deconfined"

    N_conf = len(data[::2])
    Nt = 22  # time extension of the system
    ev = 200
    Nt_ev = Nt * ev

    IPR_values = data[::2, 204 : Nt_ev + 204]  # select only IPR from data
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

    IPR_temp = IPR_array.flatten()
    print_equal_elements(IPR_temp)

    np.savetxt("IPR.txt", IPR_array)

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

    eigenvalues = np.abs(data[::2, 4:204])  # select eigenvalues
    eigenvalues = eigenvalues[:, ::2]  # eliminate chiral paired \lambda
    mean = [
        np.mean([x for x in row]) for row in eigenvalues.T
    ]  # find <\lambda> along all paths

    plt.figure()
    plt.title(r"PR vs $\lambda$" f" for {kind} phase", fontsize=13)
    plt.xlabel(r"$<\lambda>$", fontsize=15)
    plt.ylabel("<PR>", fontsize=15)
    plt.scatter(mean, IPR_mean, s=8, c="b")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    distrib = False
    InvPartRatio = False

    if distrib:
        dec, conf = loadtxt()
        decEig = dec[:, 4:204]
        confEig = conf[:, 4:204]

        conf_min = np.amin(np.abs(confEig))
        conf_max = np.amax(np.abs(confEig))
        deconf_min = np.amin(np.abs(decEig))
        deconf_max = np.amax(np.abs(decEig))

        confined = True

        # res = [ele for ele in test_list if ele > 0]
        print(f"min eigenvalue confined, on all conf.: {conf_min}")
        print(f"max eigenvalue confined, on all conf.: {conf_max}")
        print(f"min eigenvalue deconfined, on all conf.: {deconf_min}")
        print(f"max eigenvalue deconfined, on all conf.: {deconf_max}")

        rng = (conf_max - conf_min) / 4
        rng1 = (deconf_max - deconf_min) / 4

        range_ev_confined = [
            [conf_min, conf_min + rng],
            [conf_min + rng, conf_min + 2 * rng],
            [conf_min + 2 * rng, conf_min + 3 * rng],
            [conf_min + 3 * rng, conf_max],
        ]

        range_ev_deconfined = [
            [deconf_min, deconf_min + rng1 * 0.6],
            [deconf_min + rng1, deconf_min + 2 * rng1],
            [deconf_min + 2 * rng1, deconf_min + 3 * rng1],
            [deconf_min + 3 * rng1, deconf_max],
        ]

        if confined:
            kind = "Confined"
            range_ = np.array(range_ev_confined)
        else:
            kind = "Deconfined"
            range_ = np.array(range_ev_deconfined)

        spacing_for_each_rng = []

        for low, high in range_:
            sp = main(low, high, quark=0, confined=False)
            sp = sp.tolist()
            # print(sp)
            spacing_for_each_rng.append(sp)

        # histogramMultipleFunction(spacing_for_each_rng, kind, 60, range_)
        for anal in range(4):

            bins = Sturge(spacing_for_each_rng[anal])
            histogramFunction(
                spacing_for_each_rng[anal], kind, bins, range_[anal, 0], range_[anal, 1]
            )

    if InvPartRatio:
        IPR(confined=True, quark=1)

    else:
        deconf, conf = loadtxt()

        print("length column", len(deconf[0]))
        print("length row", len(deconf[:, 0]))
        print("part of data", deconf[:, 3])


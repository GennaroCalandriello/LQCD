import numpy as np
import pandas as pd
from numba import njit, jit
import matplotlib.pyplot as plt

from modulo.functions import *
from modulo.stats import *
from modulo.various import *


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
        edeconfined = np.loadtxt("PhDresults\deconf\deconfined_bz0\deconfined_bz0.txt")
        econfined = np.loadtxt("PhDresults\conf\confined_bz0\confined_bz0.txt")

    else:
        edeconfined = np.loadtxt("PhDresults/deconf/deconfined/deconfined_bz0.txt")
        econfined = np.loadtxt("PhDresults/conf/confined/confined_bz0.txt")

    if topocool:

        TOPOLOGICALconf = np.loadtxt("PhDresults\conf\confined_bz0\TopoCool.txt")
        TOPOLOGICALdeconf = np.loadtxt("PhDresults\deconf\deconfined_bz0\TopoCool.txt")

        return edeconfined, econfined, TOPOLOGICALdeconf, TOPOLOGICALconf

    else:

        return edeconfined, econfined


def main(analysis, rng):

    """This function perform the ULS and calculate the ULS distribution for each range of \lambda in the spectrum"""

    print(f"Exe in range: [{rng[0]},{rng[1]}]")
    low = rng[0]
    high = rng[1]

    spacing = []
    eigenvalue_list = []

    for j in range(len(analysis[:, 0])):
        for i in range(4, 204):
            if analysis[j, i] >= 0:  # or analysis[j, i] <= 0:
                if analysis[j, i] <= high and analysis[j, i] >= low:
                    eigenvalue_list.append([analysis[j, i], int(j)])

    ranked_ev = sorting_and_ranking(eigenvalue_list)
    spacing = spacing_evaluation(ranked_ev)

    return spacing


def CreateRanges(data, num, start, stop):

    ranger = np.linspace(start, stop, num)

    rnglist = []
    for i in range(len(ranger) - 1):
        rnglist.append([round(ranger[i], 4), round(ranger[i + 1], 4)])

    return np.array(rnglist)


if __name__ == "__main__":

    import multiprocessing
    from functools import partial

    """Execute in multiprocessing the analysis on all range for the spectra of configurations"""

    spacing_analysis = True
    IPR_analysis = False
    confined = False
    varie = False

    dec, conf = loadtxt(newresults=True)

    if confined:
        data = conf
        kind = "Confined"
    else:
        data = dec
        kind = "Deconfined"

    if spacing_analysis:

        emin = np.amin(np.abs(data[:, 4:204]))
        emax = np.amax(np.abs(data[:, 4:204]))

        range_num = 10  # number of range in spectrum that you want to analyze
        start = emin
        stop = 0.042
        rng = CreateRanges(data, range_num, start, stop)

        print("Range of lambda analysis", rng)

        with multiprocessing.Pool(processes=len(rng[:])) as pool:
            part = partial(main, data)
            results = np.array(pool.map(part, rng))
            pool.close()
            pool.join()

        Is0 = []
        Is01 = []

        for i in range(len(results)):
            temp0 = np.array(Tuning(results[i]))
            num_bins = FreedmanDiaconis(temp0)
            temp = GaussianMixtureModelIntegrator(temp0, num_bins, plot=False)
            temp1 = KernelDensityFunctionIntegrator(temp0, num_bins, plot=False)
            histogramFunction(temp0, "Deconfined", num_bins, rng[i, 0], rng[i, 1])
            Is0.append(temp)
            Is01.append(temp1)

        lambspace = np.linspace(start, stop, len(Is0))
        plt.figure()
        plt.title(r"$I_{s_0}$", fontsize=16)
        plt.xlabel(r"$\lambda$", fontsize=12)
        plt.ylabel(r"$I_{s_0}$", fontsize=12)
        plt.scatter(lambspace, Is0, s=12, label="GMM", color="violet")
        plt.scatter(lambspace, Is01, s=12, label="KDE", color="red")
        plt.legend()
        plt.show()

    if IPR_analysis:
        IPR(data, kind)

    if varie:
        spa = main(data, [0.04, 0.052])
        spa = np.array(Tuning(spa))
        # BayesianBlocks(spa)
        histogramFunction(spa, "dec", 50, 0, 0.017)
        FreedmanDiaconis(spa)


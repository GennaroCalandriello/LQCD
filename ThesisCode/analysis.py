import numpy as np
import pandas as pd
from numba import njit, jit

from modulo.functions import *


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
    """This function extracts the eigenvalues in a certain range or examine the first or the last
    n_evl; quark should be 0 (down) or 1 (up)"""
    print(f"Exe in range: [{rng[0]},{rng[1]}]")
    low = rng[0]
    high = rng[1]

    spacing = []
    eigenvalue_list = []

    for j in range(len(analysis[:, 0])):
        # eigenvalue_list = []
        # if analysis[j, 3] == quark:

        for i in range(4, 204):
            if analysis[j, i] >= 0:  # or analysis[j, i] <= 0:
                if analysis[j, i] <= high and analysis[j, i] >= low:
                    eigenvalue_list.append([analysis[j, i], int(j)])

    ranked_ev = sorting_and_ranking(eigenvalue_list)
    spacing = spacing_evaluation(ranked_ev)

    return spacing


if __name__ == "__main__":

    import multiprocessing
    from functools import partial

    spacing_analysis = False
    IPR_analysis = True

    confined = False
    dec, conf = loadtxt(newresults=True)

    if confined:
        data = conf
        kind = "Confined"
    else:
        data = dec
        kind = "Deconfined"

    if spacing_analysis:

        range_num = 10  # number of range in spectrum that you want to analyze
        rng, _, _ = CreateRanges(data, range_num)
        print("Range of lambda analysis", rng)

        with multiprocessing.Pool(processes=len(rng[:])) as pool:
            part = partial(main, data)
            results = np.array(pool.map(part, rng))
            pool.close()
            pool.join()

        for i in range(len(results)):
            bins = Sturge(results[i])
            histogramFunction(results[i], kind, bins, rng[i, 0], rng[i, 1])

    if IPR_analysis:
        IPR(data, kind)

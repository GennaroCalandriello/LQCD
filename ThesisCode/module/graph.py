import numpy as np
import matplotlib.pyplot as plt

from level_spacing_eval import *

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
        errori.append(np.std(binned[i]))

    counts, edges, _ = plt.hist(
        sp,
        bins=bins,
        density=True,
        histtype="step",
        fill=False,
        color=np.random.rand(3),
        label=r"$\lambda \in$" f"{round(low, 4)}; {round(high, 4)}",
    )
    print("mi printi questo counts?", counts)
    print("e mo printami i dati binnati", binned)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    # bin_width = edges[1] - edges[0]
    plt.errorbar(bin_centers, counts, yerr=errori, xerr=0, fmt=".", color="blue")

    plt.plot(x, GUE, "g--", label="GUE")
    # plt.plot(x, GSE, "b--", label="GSE")
    plt.plot(x, GOE, "r--", label="GOE")
    plt.plot(x, POISSON, "y--", label="Poisson")
    plt.legend()
    plt.show()

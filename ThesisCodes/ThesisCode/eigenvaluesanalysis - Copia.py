import numpy as np
import matplotlib.pyplot as plt
import math


from test_lavoroqui import *
from module.level_spacing_eval import *

"""References:
1. Anderson localization in high temperature QCD:
    background configuration properties and Dirac
    eigenmodes; Guido Cossua and Shoji Hashimoto"""

lowconfig = 50  # analyze the low configuration in the spectrum
highconfig = lowconfig  # analyze the highest eigenvalues of each configurations
n_evl = 50  # first 30 eigenvalues of each configurations
highstarting = 100


edeclistlow = []
edeclisthigh = []
econflisthigh = []
econflistlow = []


def main(
    low, high, quark: int, selectnum, loweigen, higheigen, confined,
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

    # eigenvalue_list = []
    spacing = []

    if not selectnum:
        for j in range(len(analysis[:, 0])):
            if analysis[j, 3] == quark:
                eigenvalue_list = []
                for i in range(4, 204):
                    if analysis[j, i] >= 0:  # or analysis[j, i] <= 0:
                        if analysis[j, i] <= high and analysis[j, i] >= low:
                            eigenvalue_list.append(analysis[j, i])
                unfold = unfolding_2_punto_0(eigenvalue_list, False, False)
                spacing.extend(spacing_banale(unfold))

        spacing = np.array(spacing)
        histogramFunction((spacing), kind, 50)

    if selectnum:
        for j in range(len(analysis[:, 0])):
            if analysis[j, 3] == quark:
                eigenvalue_list = []
                if loweigen:
                    for i in analysis[j, 4:n_evl]:
                        if (
                            i >= 0
                        ):  # select only positive eigenvalues (Dirac spectrum is in pairs due to chiral symmetry)
                            eigenvalue_list.append(i)
                if higheigen:
                    for i in analysis[j, highstarting:n_evl]:
                        if i >= 0:
                            eigenvalue_list.append(i)

                unfold = unfolding_2_punto_0(eigenvalue_list)
                spacing.extend(unfold)
        histogramFunction(spacing, kind, 50)

    print(f"number eigenvalues in range [{low},{high}]: {len(eigenvalue_list)}")


def mainIPR(quark, confinato=True):

    """Here is considered the <IPR>, the mean is on all configuration
    wrt the eigenvalues taken at the Nt-th time slices for each RHMC trajectory"""

    confined, deconfined = loadtxt()

    if confinato:
        data = confined
    else:
        data = deconfined
    Nt = 22

    NtXev = Nt * 200  # there is the value of all IPRs for each RHMC traj.
    first_lamdas = (
        100  # here consider the first first_lambdas eigenvalues for each RHMC traj.
    )

    results = []

    data = data[:, : 204 + NtXev]  # seleziono solo l'IPR dai dati
    length = len(data[:, 0])
    results = np.zeros((int(length / 2), first_lamdas))

    for i in range(length):

        IPR_list = []
        if data[i, 3] == quark:
            for j in range(1, 200):
                IPR_list.append(
                    1 / data[i, 204 + Nt * (j)]
                )  # seleziono IPR(\lam, Nt=22), len(IPR_list)=199 #autovalori
            IPR_list = IPR_list[
                ::2
            ]  # divido l'IPR, gli autovalori sono a coppie: (\lam1, -\lam1), (\lam2, -\lam2) etc...
            IPR_list = IPR_list[
                :first_lamdas
            ]  # prendo i primi 40 IPR riferiti ai primi 40 \lam
            results[int(i / 2)] = IPR_list

    meanIPR = []

    for j in range(len(results[0, :])):
        IPR = []
        for i in range(len(results[:])):
            IPR.append(results[i, j])
        meanIPR.append(np.mean(IPR))

    plt.figure()
    plt.scatter(range(len(meanIPR)), meanIPR, s=5, color="g")
    plt.show()

    plt.hist(meanIPR, 20, histtype="step", color="b")
    plt.show()


##########################################other functions and proofs############################àà
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


def unfoldingMatteo():
    pass


def loadtxt():

    """Structure of data:
    1st number: RHMC trajectory
    2nd number: 1 non ti interessa
    3rd number: 200 = eigenvalues calculated
    4th number: 0/1 quark up or down
    4:204 numbers: eigenvalues
    204:204+22*200: IPR"""

    edeconfined = np.loadtxt("PhDresults/deconfined/deconfined_bz0.txt")
    econfined = np.loadtxt("PhDresults/confined/confined_bz0.txt")

    return edeconfined, econfined


###############################################################################################


def histogramFunction(sp, kind: str, bins):

    GUE = distribution(sp, "GUE")
    GSE = distribution(sp, "GSE")
    GOE = distribution(sp, "GOE")
    POISSON = distribution(sp, "Poisson")

    x = np.linspace(0, max(sp), len(sp))

    plt.figure()
    plt.title(f"{kind}")
    plt.hist(
        sp,
        bins=bins,
        density=True,
        histtype="step",
        fill=False,
        color="b",
        label="Spacing distribution",
    )
    plt.plot(x, GUE, "g--", label="GUE")
    # plt.plot(x, GSE, "b--", label="GSE")
    # plt.plot(x, GOE, "r--", label="GOE")
    plt.plot(x, POISSON, "y--", label="Poisson")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    eigen_analysis = True
    IPR_analysis = False

    if eigen_analysis:
        dec, conf = loadtxt()
        decEig = dec[:, 4:204]
        confEig = conf[:, 4:204]
        # res = [ele for ele in test_list if ele > 0]
        print(f"max eigenvalue confined, on all conf.: {np.amax(np.abs(confEig))}")
        print(f"min eigenvalue confined, on all conf.: {np.amin(np.abs(confEig))}")
        print(f"max eigenvalue deconfined, on all conf.: {np.amax(np.abs(decEig))}")
        print(f"min eigenvalue deconfined, on all conf.: {np.amin(np.abs(decEig))}")

        main(
            low=0.01,
            high=0.02,
            quark=0,
            selectnum=False,
            loweigen=True,
            higheigen=False,
            confined=False,
        )

    if IPR_analysis:
        mainIPR(1)
    """Osservazioni: il mobility edge per fase deconfinata è circa 0.025, sopra 0.025 la distrib.
    è GUE"""


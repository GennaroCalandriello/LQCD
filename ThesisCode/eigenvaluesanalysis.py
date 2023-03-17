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
n_evl = 30  # first 30 eigenvalues of each configurations
highstarting = 100


edeclistlow = []
edeclisthigh = []
econflisthigh = []
econflistlow = []


def main(
    low,
    high,
    quark: int,
    selectnum=False,
    loweigen=False,
    higheigen=False,
    confined=False,
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

    eigenvalue_list = []

    if not selectnum:
        for j in range(len(analysis[:, 0])):
            if analysis[j, 3] == quark:
                for i in range(4, 204):
                    if analysis[j, i] >= 0:  # or analysis[j, i] <= 0:
                        if analysis[j, i] <= high and analysis[j, i] >= low:
                            eigenvalue_list.append(analysis[j, i])

    if selectnum:
        for j in range(len(analysis[:, 0])):
            if analysis[j, 3] == quark:
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

    print(f"number eigenvalues in range [{low},{high}]: {len(eigenvalue_list)}")

    unfold = unfolding_2_punto_0(eigenvalue_list)
    sp = spacing_banale(unfold)
    bins = round(
        2.0 * (np.abs(math.log(len(eigenvalue_list), 2) + 1))
    )  # Sturge x (un tot a piacere fra)
    histogramFunction(sp, kind, bins=bins)


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
def spacing_banale(eigs, kind="naive"):
    """Control and rewrite this!"""

    spacing = []
    rtilde = []
    eigs = np.sort(eigs)

    if kind == "rtilde":

        for i in range(1, len(eigs) - 1):
            sn = eigs[i + 1] - eigs[i]
            snm1 = eigs[i] - eigs[i - 1]
            rtilde.append(min(sn, snm1) / max(sn, snm1))

        print("mean spacing", np.mean(rtilde))

        return rtilde

    if kind == "naive":

        for i in range(1, len(eigs) - 1):
            spacing.append(eigs[i + 1] - eigs[i])

        print("mean spacing: ", np.mean(spacing))

        return spacing

    if kind == "FN":

        for i in range(1, len(eigs) - 1):
            spacing.append(max(eigs[i + 1] - eigs[i], eigs[i] - eigs[i - 1]))

        print("mean n ata vota", np.mean(spacing / np.mean(spacing)))

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


def EingenExtraction():

    edeconf, econf = loadtxt()

    for j in range(lowconfig):

        for i in edeconf[2 * j, 4:n_evl]:
            if i >= 0:
                edeclistlow.append(i)
        for k in econf[2 * j, 4:n_evl]:
            if k >= 0:
                econflistlow.append(k)

    for j in range(highconfig):

        for i in edeconf[2 * j, 80 : 80 + n_evl]:
            if i >= 0:
                edeclisthigh.append(i)
        for k in econf[2 * j, 80 : 80 + n_evl]:
            if k >= 0:
                econflisthigh.append(k)

    return edeclisthigh, edeclistlow, econflisthigh, econflistlow


def spacing_analysis():

    """Perform spacing analysis and plot the histograms"""

    edHigh, edLow, ecHigh, ecLow = EingenExtraction()

    unfoldDH = unfolding_2_punto_0(edHigh, False, False)  # deconfined high
    unfoldDL = unfolding_2_punto_0(edLow, False, False)  # deconfined low
    unfoldCH = unfolding_2_punto_0(ecHigh, False)  # confined High
    unfoldCL = unfolding_2_punto_0(ecLow, False, False)  # confined low

    spDH = spacing_banale(unfoldDH)
    spDL = spacing_banale(unfoldDL)
    spCH = spacing_banale(unfoldCH)
    spCL = spacing_banale(unfoldCL)

    return spDH, spDL, spCH, spCL


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

    exe = False
    eigen_analysis = True
    IPR_analysis = True

    if exe:
        spDH, spDL, spCH, spCL = spacing_analysis()
        histogramFunction(spDH, "Deconfined High")
        histogramFunction(spDL, "Deconfined Low")
        histogramFunction(spCH, "Confined High")
        histogramFunction(spCL, "Confined Low")

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
            low=5e-5,
            high=0.004,
            quark=0,
            selectnum=False,
            loweigen=False,
            higheigen=True,
            confined=True,
        )

    if IPR_analysis:
        mainIPR(1)
    """Osservazioni:
    1. NOOO nella fase deconfinata la distribuzione è Poissoniana
    (delocalizzato) per autovalori alti
    2. NOOO nella fase confinata la distribuzione è GOE (!!da ridefinire!!, quindi può essere
    GUE, devo riscrivermi la funzione per l'ULSD per Farther Neighbor) per autovalori
    piccoli non so quanto"""


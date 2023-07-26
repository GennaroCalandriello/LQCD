#!/usr/bin/python
from __future__ import print_function
import math
import random
import sys
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

__author__ = "Dominique Delande"
__copyright__ = "Copyright (C) 2017 Dominique Delande"
__license__ = "GPL version 2 or later"
__version__ = "1.0"
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
# ____________________________________________________________________
#
# time_propagation_anderson_model_2d.py
# Authors: Dominique Delande
# Release date: April, 2, 2017
# License: GPL2 or later
# Tested with Python v2 and Python v3
# -----------------------------------------------------------------------------------------
# This script models localization in the 2d Anderson model with box disorder,
# i.e. uncorrelated on-site energies w_n uniformly distributed in [-W/2,W/2].
# The script diagonalizes the Hamiltonian for a system of finite size L, and periodic boundary conditions
# The script propagates in time an initially localized wave packet with a Gaussian
# density with width sigma_0 (usually between 1.0 and 3.0).
# An additional modulation of the wavefunction is added so that the average k vector in each direction
# is pi/2, so that the energy is on average zero, at the band center.
# The time propagation is done by expansion on the eigenbasis obtained by numerical diagonalization of the Hamiltonian
# The average squared displacement <r^2(t)> is printed in "squared_displacement.dat"
# The average density probability in configuration space (i.e. |\psi_i|^2) is printed in density.dat
# -----------------------------------------------------------------------------------------
if len(sys.argv) != 6:
    print(
        "Usage (5 parameters):\n time_propagation_anderson_model_2d.py L W nr t_max nsteps"
    )
    sys.exit()
L = int(sys.argv[1])
W = float(sys.argv[2])
nr = int(sys.argv[3])
t_max = float(sys.argv[4])
nsteps = int(sys.argv[5])
t_step = t_max / nsteps
# Size of the initial wavepacket
sigma_0 = 2.0
coef = -0.5 / (sigma_0**2)


# Generate a disordered sequence of on-site energies in the array "disorder"
def generate_disorder(L, W):
    disorder = W * ((np.random.uniform(size=L * L)).reshape((L, L)) - 0.5)
    return disorder


# Generate the Hamiltonian matrix for one realization of the random disorder
# The Hamitonian is in the LxL array "H"
def generate_hamiltonian(L, W):
    H = np.zeros((L * L, L * L))
    disorder = generate_disorder(L, W)
    for i in range(L):
        ip1 = (i + 1) % L
        for j in range(L):
            H[i * L + j, i * L + j] = disorder[i, j]
            jp1 = (j + 1) % L
            H[ip1 * L + j, i * L + j] = 1.0
            H[i * L + j, ip1 * L + j] = 1.0
            H[i * L + jp1, i * L + j] = 1.0
            H[i * L + j, i * L + jp1] = 1.0
    return H


def generate_initial_state(L):
    psi = np.zeros(L * L)
    #  phase1=2.0*math.pi*random.random()
    #  phase2=2.0*math.pi*random.random()
    phase1 = 0.0
    phase2 = 0.0
    # add a combination of exp(+/-ik) along each direction
    # to adjust average energy around 0
    for i in range(L):
        for j in range(L):
            psi[i * L + j] = (
                math.exp(coef * ((i - L / 2) ** 2 + (j - L / 2) ** 2))
                * math.cos(phase1 + 0.5 * math.pi * (i - L / 2))
                * math.cos(phase2 + 0.5 * math.pi * (j - L / 2))
            )
    # Normalize
    psi *= 1.0 / math.sqrt(np.sum(psi**2))
    return psi


def compute_r2(psi):
    L = int(math.sqrt(psi.size))
    r2 = 0.0
    for i in range(L):
        for j in range(L):
            r2 += ((i - L / 2) ** 2 + (j - L / 2) ** 2) * (psi[i * L + j] ** 2)
    return r2


# ____________________________________________________________________
# view_density.py
# Author: Dominique Delande
# Release date: April, 2, 2017
# License: GPL2 or later
# Tested with Python v2 and Python v3
# Reads 2d data in a ASCII file and make a color 2d plot with matplotlib
# In the ASCII file, the data must be on consecutive lines, following the rows
# of the 2d data, without hole
# The first two lines should contain "# n1 delta1" and "# n2 delta2" where
# n1 (n2) is the number of rows (columns)
# delta1 and delta2 are optional arguments given the step in the corresponding dimension
# Each line may contain several data, the last column is plotted,
# unless the there is a second argument, which is the column to be plotted
# ____________________________________________________________________


def view_density(file_name, column=-1, block=True):
    f = open(file_name, "r")
    # f = open('eigenstate.dat','r')
    line = (f.readline().lstrip("#")).split()
    n1 = int(line[0])
    if len(line) > 1:
        delta1 = float(line[-1])
    else:
        delta1 = 1.0
    line = (f.readline().lstrip("#")).split()
    n2 = int(line[0])
    if len(line) > 1:
        delta2 = float(line[-1])
    else:
        delta2 = 1.0
    # print sys.argv,len(sys.argv)
    arr = np.loadtxt(file_name, comments="#").reshape(n1, n2, -1)
    # print arr
    Z = arr[:, :, column]
    print("Maximum value = ", Z.max())
    print("Minimum value = ", Z.min())
    plt.figure()
    plt.imshow(Z, origin="lower", interpolation="nearest")
    plt.show(block)
    return


squared_displacement = np.zeros(nsteps + 1)
density = np.zeros(L * L)

ir = 0
filename = "squared_displacement.dat"
f = open(filename, "w")

while ir < nr:
    H = generate_hamiltonian(L, W)
    psir = generate_initial_state(L)
    Hpsir = np.tensordot(H, psir, axes=(0, 0))
    energy = np.dot(psir, Hpsir)
    #  print(energy)
    #  if energy is not close to 0, reject it and generate a new configuration
    if abs(energy) > 0.25:
        #    print("Energy =",energy," rejected!")
        continue
    ir = ir + 1
    print("Configuration", ir, energy)
    (energy_levels, eigenstates) = linalg.eigh(H)
    # The diagonalization routine does not sort the eigenvalues (which is stupid, by the way)
    # Thus, sort them
    idx = np.argsort(energy_levels)
    energy_levels = energy_levels[idx]
    eigenstates = eigenstates[:, idx]
    #  print(eigenstates)
    tab_coef = np.tensordot(eigenstates, psir, (0, 0))
    for j in range(nsteps + 1):
        time = t_step * j
        tab_cos = np.cos(time * energy_levels)
        tab_sin = np.sin(time * energy_levels)
        psi_t_r = np.tensordot(eigenstates, tab_cos * tab_coef, axes=(1, 0))
        psi_t_i = np.tensordot(eigenstates, tab_sin * tab_coef, axes=(1, 0))
        #    energy=np.dot(psi_t_r,np.tensordot(H,psi_t_r,axes=(0,0)))+np.dot(psi_t_i,np.tensordot(H,psi_t_i,axes=(0,0)))
        #    print(time,energy,compute_r2(psi_t_r)+compute_r2(psi_t_i))
        squared_displacement[j] += compute_r2(psi_t_r) + compute_r2(psi_t_i)
    density[0 : L * L] += psi_t_r[0 : L * L] ** 2 + psi_t_i[0 : L * L] ** 2
#  print(solve.t,y)
filename = "density.dat"
f = open(filename, "w")
f.write("# %d\n" % (L))
f.write("# %d\n" % (L))
for i in range(L):
    for j in range(L):
        f.write("%g\n" % (density[i * L + j] / nr))
f.close()
filename_r2 = "squared_displacement.dat"
f = open(filename_r2, "w")
for j in range(nsteps + 1):
    f.write("%g %g\n" % (j * t_step, squared_displacement[j] / nr))
f.close()
print("Done, results saved in", filename, "and", filename_r2)
view_density("density.dat")

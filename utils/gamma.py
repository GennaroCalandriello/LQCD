import numpy as np

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

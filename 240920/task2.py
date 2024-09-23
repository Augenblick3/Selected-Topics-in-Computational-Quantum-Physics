import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def construct_hamiltonian(N, F):
    H = np.zeros((N, N))
    for i in range(N-1):
        H[i, i+1] = -1
        H[i+1, i] = -1

    for j in range(N):
        H[j, j] = F * (j+1)

    return H

def construct_psi_0(alpha, N0, k0, N):
    psi_0 = np.zeros(N, dtype=complex)
    for i in range(N):
        a = -alpha ** 2 / 2 * (i - N0) ** 2
        psi_0[i] = np.exp(a) * np.exp(1j * k0 * i)
    psi_0 /= np.linalg.norm(psi_0)
    return psi_0

def time_evolution(eigenvectors, eigenvalues, psi_0, t):
    c = np.dot(eigenvectors.T.conj(), psi_0)
    psi_t = np.dot(eigenvectors, c * np.exp(-1j * eigenvalues * t))
    return psi_t

def main():
    N, F, k0, alpha, N0 = 101, 0.1, np.pi / 2, 0.15, 51
    H = construct_hamiltonian(N, F)
    psi_0 = construct_psi_0(alpha, N0, k0, N)

    eigenvalues, eigenvectors = eigh(H)
    psi_42 = time_evolution(eigenvectors, eigenvalues, psi_0, 42)

    prob = []
    for j in [10, 20, 30, 40, 50]:
        prob_j = np.abs(psi_42[j]) ** 2
        prob.append(prob_j)
    prob = np.array(prob)

    t_values = np.linspace(0, 100, 1000)
    position_j = np.linspace(0, N, N)
    J, T = np.meshgrid(position_j, t_values)
    psi_t_values = np.array([time_evolution(eigenvectors, eigenvalues, psi_0, t) for t in t_values])
    prob_values = np.abs(psi_t_values) ** 2

    plt.figure()
    plt.contourf(T, J, prob_values, levels=100, cmap='viridis')
    plt.xlabel('Time t')
    plt.ylabel('Position t')
    plt.savefig("student_2022012301_hm_1.png")
    plt.show()

    print("========Task 2: Output========")
    print("(1)_10_eigenvalues:", eigenvalues[:10])
    print("(2)_Probability:", prob)
    print("(3)_Plot: figure saved as student_2022012301_hm_1.png")

if __name__ == '__main__':
    main()

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

def cnt_particles_before(state, site, spin):
    pos = 2 * site + spin
    mask = (1 << pos) - 1
    return bin(state & mask).count('1')

def cnt_particles(state, site, spin):
    pos = 2 * site + spin
    return (state >> pos) & 1

def create_annihilation_op(site, spin, sites):
    dim = 4 ** sites
    operator = lil_matrix((dim, dim))
    for i in range(dim):
        if (i >> (2 * site + spin)) & 1:
            j = i ^ (1 << (2 * site + spin))
            sign = (-1) ** cnt_particles_before(i, site, spin)
            operator[j, i] = sign
    return operator.tocsr()

def create_creation_op(site, spin, sites):
    return create_annihilation_op(site, spin, sites).transpose()

def create_number_op(site, spin, sites):
    return create_creation_op(site, spin, sites) @ create_annihilation_op(site, spin, sites)

def get_basis(N_up, N_down, sites):
    dim = 4 ** sites
    basis = []
    for state in range(dim):
        n_up = 0
        n_down = 0
        for i in range(sites):
            n_up += cnt_particles(state, i, 0)
            n_down += cnt_particles(state, i, 1)
        if n_up == N_up and n_down == N_down:
            basis.append(state)
    return basis

def construct_hamiltonian(sites, edges, extended_edges, t, U, V, mu):
    dim = 4 ** sites
    H = lil_matrix((dim, dim), dtype=np.float64)
    for (i, j) in edges:
        for spin in [0, 1]:
            ci = create_creation_op(i, spin, sites)
            cj = create_creation_op(j, spin, sites)
            ci_dagger = ci.transpose()
            cj_dagger = cj.transpose()
            H -= t * (ci_dagger @ cj + cj_dagger @ ci)

    for (i, j) in extended_edges:
        ni_up = create_number_op(i, 0, sites)
        nj_up = create_number_op(j, 0, sites)
        ni_down = create_number_op(i, 1, sites)
        nj_down = create_number_op(j, 1, sites)
        H -= V * (ni_up + ni_down) @ (nj_up + nj_down)

    for i in range(sites):
        ni_up = create_number_op(i, 0, sites)
        ni_down = create_number_op(i, 1, sites)
        H += U * (ni_up @ ni_down)
        H -= mu * (ni_up + ni_down)

    return H.tocsr()

def calculate_expectation_value(ground_state, sites):
    exp_values = np.zeros((sites, 2))
    for i in range(sites):
        for spin in [0, 1]:
            ni = create_number_op(i, spin, sites)
            exp_values[i, spin] = ground_state @ ni @ ground_state
    return exp_values

def calculate_sub_hamiltonian(H, basis):
    dim = len(basis)
    sub_H = lil_matrix((dim, dim), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            sub_H[i, j] = H[basis[i], basis[j]]
    eigenvalues_sub = np.zeros(6)
    eigenvalues_sub, _ = eigsh(sub_H, k=6, which='SA')
    return eigenvalues_sub

def main():
    t = 1.0
    U = 6.0
    V = 0.3
    mu = 3.0

    N_up = 2
    N_down = 4

    sites = 6
    edges = [(0, 2), (1, 3), (2, 4), (3, 5), (1, 2), (3, 4)]
    extended_edges = [(0 ,1), (2, 3), (1, 4), (4, 5)]

    H = construct_hamiltonian(sites, edges, extended_edges, t, U, V, mu)

    basis = get_basis(N_up, N_down, sites)
    eigenvalues_sub = calculate_sub_hamiltonian(H, basis)

    eigenvalues, eigenvectors = eigsh(H, k=20, which='SA')
    ground_states = eigenvectors[:, 0]

    exp_values = calculate_expectation_value(ground_states, sites)

    print("========Task 1: Output========")
    print("(1)_6_eigenvalues_nup_2_ndown_4:", eigenvalues_sub)
    print()
    print("(2)_20_eigenvalues:", eigenvalues)
    print()
    print("(3)_Density_Up_Spin:", exp_values[:, 0])
    print("(3)_Density_Down_Spin:", exp_values[:, 1])

if __name__ == '__main__':
    main()

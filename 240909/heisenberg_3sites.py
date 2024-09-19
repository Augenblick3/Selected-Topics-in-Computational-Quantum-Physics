import numpy as np  

sigma_x = np.array([[0, 1/2],
                    [1/2, 0]], dtype=complex)
sigma_y = np.array([[0, -1j/2],
                    [1j/2, 0]], dtype=complex)
sigma_z = np.array([[1/2, 0],
                    [0, -1/2]], dtype=complex)
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

# 定义自旋算符 S_i^alpha = (1/2) * sigma^alpha  
def spin_operator(sigma, site, total_sites):
    """构造第 site 个自旋的算符，总共有 total_sites 个自旋"""
    op_list = []
    for i in range(total_sites):
        if i == site:
            op_list.append(sigma)
        else:
            op_list.append(I)
    operator = op_list[0]
    for op in op_list[1:]:
        operator = np.kron(operator, op)
    return operator

# 构造哈密顿量矩阵  
def construct_hamiltonian():
    total_sites = 3 
    dim = 2 ** total_sites
    H = np.zeros((dim, dim), dtype=complex)

    # 定义相互作用项的自旋对  
    interactions = [(0, 1),  # S1 · S2  
                    (1, 2)]  # S2 · S3  

    for (i, j) in interactions:
        # 计算 S_i^alpha · S_j^alpha 的和
        for sigma in [sigma_x, sigma_y, sigma_z]:
            # 构建 S_i^alpha  
            S_i = spin_operator(sigma, i, total_sites)
            # 构建 S_j^alpha  
            S_j = spin_operator(sigma, j, total_sites)
            H += S_i @ S_j  # 使用 @ 符号表示矩阵乘法
    return H.real

def construct_basis_labels():
    basis_labels = []
    for i in range(8):
        bits = format(i, '03b')
        state = ''.join(['↑' if b == '0' else '↓' for b in bits])
        basis_labels.append('|' + state + '⟩')
    return basis_labels

def main():
    H = construct_hamiltonian()
    print("Hamiltonian matrix H in the z-basis:")
    np.set_printoptions(precision=2, suppress=True)
    print(H)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    print("\nEigenvalues:")
    print(eigenvalues)

    # 输出特征向量（归一化）
    print("\nEigenvectors (columns correspond to eigenvectors):")
    for i, eigenvalue in enumerate(eigenvalues):
        if np.abs(eigenvalue) < 1e-3:
            eigenvalue = 0
        if np.abs(eigenvalue - 0.5) < 1e-3:
            eigenvalue = 0.5
        print(f"\nEigenvalue {eigenvalue}:")
        eigenvector = eigenvectors[:, i]
        basis_labels = construct_basis_labels()
        for coeff, label in zip(eigenvector, basis_labels):
            if np.abs(coeff) > 1e-3:
                coeff_str = f"{coeff.real:.3g}"
                if coeff.imag != 0:
                    coeff_str += f"+{coeff.imag:.3g}j"
                print(f"{coeff_str} {label}")

if __name__ == "__main__":
    main()

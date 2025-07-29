import torch

def get_independent_rows(B, tol=1e-10):
    _, S, V = torch.linalg.svd(B.double())
    rank = (S > tol).sum().item()
    independent_indices = []
    for i in range(B.shape[0]):
        B_temp = B[independent_indices + [i]]
        if torch.linalg.matrix_rank(B_temp.double()) > len(independent_indices):
            independent_indices.append(i)
        if len(independent_indices) == rank:
            break
    return independent_indices

# Your matrix B
B = torch.tensor([
    [-1, 0, 0, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],  # Carbon
    [0, -2, 0, 0, -2, 0, -1, 0, -1, -1, 0, -1, 0],  # Hydrogen
    [-2, -1, 0, -2, -1, 0, -3, -3, -1, 0, 0, -1, 0],  # Oxygen
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0],  # Sodium
    [0, 0, -2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0],  # Nitrogen
    [0, 0, 0, 0, 0, 0, -1, -2, -1, 1, 1, 0, 0],  # Charge
], dtype=torch.float64)

indep_rows = get_independent_rows(B)
print(indep_rows)
B_dep = B[indep_rows]

# Check invertibility
try:
    torch.linalg.inv(B_dep @ B_dep.T)
    print("B_dep is invertible.")
except RuntimeError as e:
    print("B_dep is not invertible:", e)

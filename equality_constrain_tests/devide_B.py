import torch
import itertools

def get_valid_B_partitions(B: torch.Tensor, tol=1e-12, max_condition_number=1e4):
    """
    Given a constraint matrix B (m x n), find all partitions into B_dep and B_indep
    such that B_dep is square, invertible, and well-conditioned.
    Returns list of (B_dep, B_indep, dep_indices, indep_indices)
    """
    m, n = B.shape
    all_indices = list(range(n))
    valid_partitions = []

    for dep_indices in itertools.combinations(all_indices, m):
        dep_indices = list(dep_indices)
        indep_indices = [i for i in all_indices if i not in dep_indices]

        B_dep = B[:, dep_indices]

        # Check: square, invertible, and well-conditioned
        if B_dep.shape[0] == B_dep.shape[1]:
            try:
                cond_number = torch.linalg.cond(B_dep)
                if (
                    torch.linalg.matrix_rank(B_dep) == m and
                    torch.abs(torch.det(B_dep)) > tol and
                    cond_number < max_condition_number
                ):
                    B_indep = B[:, indep_indices]
                    valid_partitions.append((B_dep, B_indep, dep_indices, indep_indices))
                    if dep_indices == [1, 5, 7, 8, 11]:
                        print('success')
                    # print(f"Valid B_dep indices: {dep_indices}, condition number: {cond_number:.2e}")
            except RuntimeError:
                continue  # Singular or ill-conditioned, skip

    return valid_partitions

# Example B matrix
B = torch.tensor([
    [-1, 0, 0, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],  # Carbon
    [0, -2, 0, 0, -2, 0, -1, 0, -1, -1, 0, -1, 0],  # Hydrogen
    [-2, -1, 0, -2, -1, 0, -3, -3, -1, 0, 0, -1, 0],  # Oxygen
    [0, 0, -2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0],  # Nitrogen
    [0, 0, 0, 0, 0, 0, -1, -2, -1, 1, 1, 0, 0],  # Charge
], dtype=torch.float64)

partitions = get_valid_B_partitions(B)

print(f"\nFound {len(partitions)} valid and well-conditioned (B_dep, B_indep) combinations.")

# Show one example
if partitions:
    B_dep, B_indep, dep_idx, indep_idx = partitions[0]
    print("Example B_dep indices:", dep_idx)
    print("Example B_indep indices:", indep_idx)
    print("B_dep shape:", B_dep.shape)
    print("B_indep shape:", B_indep.shape)
else:
    print("No suitable partitions found.")



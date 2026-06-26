import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Compute initial distances and radii to determine constraint tightness
    x_centers = v0[0::3]
    y_centers = v0[1::3]
    r_centers = v0[2::3]

    # Precompute all pairwise distances and constraint tightness
    x_c = x_centers.reshape(n, 1)
    y_c = y_centers.reshape(n, 1)
    r_c = r_centers.reshape(n, 1)

    dx = x_c - x_c.T
    dy = y_c - y_c.T
    dist_sq = dx**2 + dy**2
    r_sum = r_c + r_c.T
    overlap_constraints = dist_sq - r_sum**2

    # Compute constraint tightness as absolute value of overlap constraints
    tightness = np.abs(overlap_constraints)
    tightness = tightness[np.triu_indices(n, 1)]  # Only upper triangle

    # Get indices of all constraints
    constraint_indices = np.arange(n * (n - 1) // 2)

    # Sort constraints by tightness and create a permutation of circle indices
    sorted_indices = np.argsort(tightness)
    permuted_indices = np.zeros(n * (n - 1) // 2, dtype=int)
    for i, idx in enumerate(sorted_indices):
        i1, i2 = np.unravel_index(idx, (n, n-1))
        permuted_indices[i] = (i1, i2 + i1 + 1)  # (i1, i2) where i2 > i1

    # Apply the permutation to the constraint functions
    new_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            # Get the corresponding permutation index
            idx = np.where((permuted_indices[:, 0] == i) & (permuted_indices[:, 1] == j))[0]
            if idx.size == 0:
                idx = np.where((permuted_indices[:, 0] == j) & (permuted_indices[:, 1] == i))[0][0]
            # Add constraints for the permuted indices
            new_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Replace the original constraints with the new ones
    cons = new_cons

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
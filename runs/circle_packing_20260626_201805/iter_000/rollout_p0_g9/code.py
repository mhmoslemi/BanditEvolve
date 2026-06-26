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
    
    # Vectorize overlap constraints
    x_coords = v0[0::3]
    y_coords = v0[1::3]
    r_coords = v0[2::3]
    
    # Precompute indices for efficient constraint evaluation
    i_indices = np.arange(n)
    j_indices = np.arange(n)
    i_j_pairs = np.array([[i, j] for i in range(n) for j in range(i+1, n)])
    
    def vectorized_overlap_constraint(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        dx = x[i_j_pairs[:, 0]] - x[i_j_pairs[:, 1]]
        dy = y[i_j_pairs[:, 0]] - y[i_j_pairs[:, 1]]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r[i_j_pairs[:, 0]] + r[i_j_pairs[:, 1]])**2
        return dist_sq - min_dist_sq

    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: vectorized_overlap_constraint(v)[np.where((i_j_pairs[:, 0] == i) & (i_j_pairs[:, 1] == j))[0][0]]})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
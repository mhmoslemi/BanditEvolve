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
    
    # Vectorized overlap constraint for performance
    def vectorized_overlap_constraint(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dx = x_centers[:, np.newaxis] - x_centers[np.newaxis, :]
        dy = y_centers[:, np.newaxis] - y_centers[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r_radii[:, np.newaxis] + r_radii[np.newaxis, :])**2
        return dist_sq - min_dist_sq

    # Convert vectorized constraint to list of scalar constraints
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_cons.append({"type": "ineq", "fun": constraint_func})
    cons.extend(overlap_cons)

    # Add mutation-like diversification by swapping sub-structures
    # This is a probabilistic swap of clusters to avoid local optima
    def swap_clusters(v, swap_prob=0.2):
        if np.random.rand() < swap_prob:
            # Randomly select two clusters
            cluster1 = np.random.randint(0, n // 2)
            cluster2 = np.random.randint(0, n // 2)
            # Get indices of circles in clusters (for simplicity, assume clusters are even)
            indices1 = np.arange(cluster1*2, (cluster1+1)*2)
            indices2 = np.arange(cluster2*2, (cluster2+1)*2)
            # Swap centers and radii
            v[indices1*3], v[indices2*3] = v[indices2*3], v[indices1*3]
            v[indices1*3+1], v[indices2*3+1] = v[indices2*3+1], v[indices1*3+1]
            v[indices1*3+2], v[indices2*3+2] = v[indices2*3+2], v[indices1*3+2]
        return v

    # Inject mutation-like diversification into the optimization
    # Create a modified objective with mutation applied to the initial guess
    def mutated_neg_sum_radii(v):
        v_mutated = swap_clusters(np.copy(v))
        return -np.sum(v_mutated[2::3])

    res = minimize(mutated_neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
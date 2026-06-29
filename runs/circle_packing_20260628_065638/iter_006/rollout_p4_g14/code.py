import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = np.random.uniform(0.1, 0.9, n)
    ys = np.random.uniform(0.1, 0.9, n)
    # Cluster the points into 5 columns to promote compact spatial arrangement
    cluster_size = n // cols
    for i in range(cols):
        # Ensure points in the same column are vertically aligned
        cluster_indices = np.arange(i * cluster_size, (i + 1) * cluster_size)
        if i < n % cols:
            cluster_indices = np.append(cluster_indices, n - (cols - i - 1))
        # Adjust vertical positions for compactness
        ys[cluster_indices] = np.linspace(0.2, 0.8, len(cluster_indices))
        # Slight randomization to break symmetry
        ys[cluster_indices] += np.random.uniform(-0.03, 0.03, len(cluster_indices))
        # Shift alternate columns to avoid vertical alignment
        if i % 2 == 1:
            ys[cluster_indices] += 0.15
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

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
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Non-local reconfiguration: identify and expand the most isolated cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Compute pairwise distances to identify isolation
        dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dists[i] += np.sqrt(dx*dx + dy*dy)
        # Find the cluster with the most isolation
        cluster_indices = np.argsort(dists)[::-1][:5]  # Select top 5 most isolated circles
        # Compute centroid of the cluster
        cluster_centers = np.column_stack([centers[0][cluster_indices], centers[1][cluster_indices]])
        cluster_centroid = np.mean(cluster_centers, axis=0)
        # Adjust positions of the cluster to form a more compact configuration
        for i in cluster_indices:
            v[3*i] = cluster_centroid[0]
            v[3*i+1] = cluster_centroid[1]
            # Increase radius of the cluster by a controlled amount
            v[3*i+2] += 0.003
        # Clip radii to ensure they stay within bounds
        v[2::3] = np.clip(v[2::3], 1e-4, 0.5)
        # Re-evaluate with the modified parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
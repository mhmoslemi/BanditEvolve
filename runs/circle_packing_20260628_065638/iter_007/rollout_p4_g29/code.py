import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Generate initial positions with a randomized geometric clustering algorithm
    np.random.seed(42)
    cluster_centers = np.random.rand(3, 3)  # 3 clusters in 2D space
    cluster_sizes = np.random.randint(5, 10, size=3)
    cluster_weights = cluster_sizes / cluster_sizes.sum()
    
    # Assign points to clusters
    cluster_assignments = np.random.choice(3, size=n, p=cluster_weights)
    xs = []
    ys = []
    for c in range(3):
        mask = (cluster_assignments == c)
        cluster_points = np.random.rand(np.sum(mask), 2)
        cluster_points[:, 0] = cluster_centers[c, 0] + (cluster_points[:, 0] - 0.5) * 0.5
        cluster_points[:, 1] = cluster_centers[c, 1] + (cluster_points[:, 1] - 0.5) * 0.5
        xs.extend(cluster_points[:, 0])
        ys.extend(cluster_points[:, 1])
    
    # Introduce controlled perturbation to cluster positions
    for i in range(n):
        xs[i] += np.random.uniform(-0.05, 0.05)
        ys[i] += np.random.uniform(-0.05, 0.05)
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized overlap constraints
    def vectorized_overlap_constraints(v):
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        dx = centers[0][:, None] - centers[0][:]
        dy = centers[1][:, None] - centers[1][:]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (radii[:, None] + radii[None, :])**2
        return dist_sq - min_dist_sq

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Perturb the cluster centers to enable expansion
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        
        # Identify the cluster with the smallest average radius
        cluster_r = np.array([np.mean(radii[cluster_assignments == c]) for c in range(3)])
        cluster_idx = np.argmin(cluster_r)
        
        # Perturb the positions of the cluster
        perturbation = 0.03 * np.random.rand(3)
        centers[0][cluster_assignments == cluster_idx] += perturbation[0]
        centers[1][cluster_assignments == cluster_idx] += perturbation[1]
        
        # Re-evaluate with new positions
        new_v = np.empty(3 * n)
        new_v[0::3] = centers[0]
        new_v[1::3] = centers[1]
        new_v[2::3] = radii
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
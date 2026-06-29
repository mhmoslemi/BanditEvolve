import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using randomized geometric clustering
    np.random.seed(42)
    xs = np.random.uniform(0.1, 0.9, n)
    ys = np.random.uniform(0.1, 0.9, n)
    # Cluster the points to form a tighter group in the center
    cluster_center = np.array([0.5, 0.5])
    cluster_radius = 0.2
    cluster_indices = np.random.choice(n, size=10, replace=False)
    xs[cluster_indices] = cluster_center[0] + np.random.uniform(-cluster_radius, cluster_radius, 10)
    ys[cluster_indices] = cluster_center[1] + np.random.uniform(-cluster_radius, cluster_radius, 10)
    # Spread out the remaining points to avoid too much overlap
    for i in range(n):
        if i not in cluster_indices:
            xs[i] += np.random.uniform(-0.1, 0.1)
            ys[i] += np.random.uniform(-0.1, 0.1)
    
    r0 = 0.25 / np.sqrt(n) - 1e-3
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
    
    # Vectorized constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10})
    
    # Targeted expansion of cluster: identify the cluster and expand its radii
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Identify cluster based on proximity to cluster center
        cluster_mask = np.sqrt((centers[0] - 0.5)**2 + (centers[1] - 0.5)**2) < 0.25
        cluster_indices = np.where(cluster_mask)[0]
        # Increase radii of cluster elements by a small amount
        v[3*cluster_indices + 2] += 0.005
        # Clip radii to ensure they stay within bounds
        v[2::3] = np.clip(v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
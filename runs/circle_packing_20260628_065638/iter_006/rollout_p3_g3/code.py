import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    np.random.seed(42)
    cluster_centers = np.random.rand(n, 2)
    cluster_centers = np.apply_along_axis(lambda x: x / np.max(x) * 0.8 + 0.1, axis=1, arr=cluster_centers)
    cluster_centers = np.vstack([cluster_centers, np.random.rand(1, 2)])
    cluster_centers = np.vstack([cluster_centers, np.random.rand(1, 2)])
    cluster_centers = cluster_centers[:n]
    cluster_radii = np.random.rand(n) * 0.1 + 0.05
    
    # Assign each circle to a cluster
    cluster_assignments = np.random.randint(0, n, size=n)
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    
    for i in range(n):
        cluster = cluster_assignments[i]
        x[i] = cluster_centers[cluster, 0] + np.random.uniform(-0.05, 0.05)
        y[i] = cluster_centers[cluster, 1] + np.random.uniform(-0.05, 0.05)
        r[i] = cluster_radii[cluster]
    
    # Expand radii of the most isolated cluster
    distances = np.zeros(n)
    for i in range(n):
        dist = np.sum((np.array([x[i], y[i]]) - np.array([x, y])).T ** 2, axis=1)
        distances[i] = np.sum(np.sqrt(dist))
    
    # Find the most isolated cluster
    isolated_cluster = np.argmin(distances)
    r[isolated_cluster] *= 1.5
    r = np.clip(r, 1e-4, 0.5)
    
    v0 = np.empty(3 * n)
    v0[0::3] = x
    v0[1::3] = y
    v0[2::3] = r

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Create constraints for all pairs
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
    
    # Non-local perturbation strategy: randomize positions of a subset of circles
    if res.success:
        v = res.x
        # Select a random subset for perturbation
        perturb_indices = np.random.choice(n, size=int(n * 0.3), replace=False)
        # Apply small random perturbation to their positions
        perturbation = 0.05 * np.random.rand(len(perturb_indices) * 3)
        perturbed_v = v.copy()
        idx = 0
        for i in perturb_indices:
            perturbed_v[3*i] += perturbation[idx]
            perturbed_v[3*i+1] += perturbation[idx+1]
            perturbed_v[3*i+2] += perturbation[idx+2]
            idx += 3
        # Clip radii to ensure they stay within bounds
        perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
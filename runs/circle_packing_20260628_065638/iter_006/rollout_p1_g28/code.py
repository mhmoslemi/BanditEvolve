import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    np.random.seed(42)
    cluster_centers = np.random.rand(4, 2)
    cluster_centers = (cluster_centers - 0.5) * 0.4 + 0.5  # Center clusters in the square
    cluster_radii = np.full(4, 0.15)
    cluster_radii[0] = 0.25
    cluster_radii[1] = 0.2
    cluster_radii[2] = 0.18
    cluster_radii[3] = 0.17
    
    # Distribute circles within each cluster
    xs = []
    ys = []
    for i in range(n):
        cluster_idx = i % 4
        angle = 2 * np.pi * np.random.rand()
        radius = np.random.rand() * cluster_radii[cluster_idx] + 0.02
        x = cluster_centers[cluster_idx, 0] + radius * np.cos(angle)
        y = cluster_centers[cluster_idx, 1] + radius * np.sin(angle)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
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
    
    # Targeted expansion of the most isolated cluster
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Identify the cluster with the largest distance to other clusters
        cluster_indices = np.floor(n / 4) * np.arange(4)
        cluster_distances = np.zeros(4)
        for i in range(4):
            cluster_points = np.where(np.arange(n) % 4 == i)[0]
            cluster_center = np.mean(centers[cluster_points], axis=0)
            distances = np.linalg.norm(centers[cluster_points] - cluster_center, axis=1)
            cluster_distances[i] = np.max(distances)
        max_cluster_idx = np.argmax(cluster_distances)
        # Increase radii of circles in the most isolated cluster
        perturb_indices = np.where(np.arange(n) % 4 == max_cluster_idx)[0]
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
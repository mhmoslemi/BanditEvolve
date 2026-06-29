import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    def random_cluster_centers(n, cols, rows):
        # Create a grid of cluster centers
        cluster_centers = []
        for r in range(rows):
            for c in range(cols):
                x = (c + 0.5) / cols
                y = (r + 0.5) / rows
                cluster_centers.append([x, y])
        # Randomly select n centers from the grid
        selected_indices = np.random.choice(len(cluster_centers), n, replace=False)
        selected_centers = [cluster_centers[i] for i in selected_indices]
        # Add random noise to break symmetry
        for i in range(n):
            selected_centers[i][0] += np.random.uniform(-0.05, 0.05)
            selected_centers[i][1] += np.random.uniform(-0.05, 0.05)
        return selected_centers
    
    # Generate initial cluster positions
    xs = [c[0] for c in random_cluster_centers(n, cols, rows)]
    ys = [c[1] for c in random_cluster_centers(n, cols, rows)]
    
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
    def vectorized_overlap(v):
        radii = v[2::3]
        x = v[0::3]
        y = v[1::3]
        dist_sq = np.zeros((n, n))
        for i in range(n):
            dx = x - x[i]
            dy = y - y[i]
            dist_sq[:, i] = dx*dx + dy*dy
        return dist_sq - (radii[:, np.newaxis] + radii[np.newaxis, :]) ** 2

    # Create constraints for all pairs using vectorized calculation
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Create constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: vectorized_overlap(v)[i, j]})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Non-local reconfiguration: identify the most isolated cluster
    if res.success:
        v = res.x
        radii = v[2::3]
        # Compute isolation metric (distance to nearest neighbor)
        dists = np.zeros(n)
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if i != j:
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        min_dist = dist
            dists[i] = min_dist
        # Select the most isolated cluster
        isolated_index = np.argmax(dists)
        # Apply controlled expansion to the isolated cluster
        perturbation = np.random.uniform(-0.05, 0.05, size=3)
        perturbed_v = v.copy()
        perturbed_v[3*isolated_index] += perturbation[0]
        perturbed_v[3*isolated_index+1] += perturbation[1]
        perturbed_v[3*isolated_index+2] += perturbation[2]
        # Clip radii to ensure they stay within bounds
        perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
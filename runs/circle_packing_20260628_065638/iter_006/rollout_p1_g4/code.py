import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    np.random.seed(42)
    clusters = np.random.rand(3, 3)  # 3 clusters to start with
    cluster_centers = np.random.rand(3, 2)
    cluster_radii = np.random.rand(3) * 0.1 + 0.05
    
    # Generate initial points using Voronoi-like clustering
    xs = []
    ys = []
    for _ in range(n):
        # Choose a cluster to assign the point to
        cluster_idx = np.random.choice(3)
        # Generate a point within the cluster's bounds
        x = cluster_centers[cluster_idx, 0] + (np.random.rand() - 0.5) * 2 * cluster_radii[cluster_idx]
        y = cluster_centers[cluster_idx, 1] + (np.random.rand() - 0.5) * 2 * cluster_radii[cluster_idx]
        # Ensure the point stays within the unit square
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
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
    
    # After initial optimization, identify and expand the most isolated cluster
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate distances between all pairs of centers
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_matrix[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify the cluster with the largest average distance to others
        cluster_indices = np.arange(n)
        cluster_assignments = np.random.randint(0, 3, n)
        for i in range(n):
            cluster = cluster_assignments[i]
            cluster_indices[cluster] = np.append(cluster_indices[cluster], i)
        
        # Compute average distance for each cluster
        cluster_distances = []
        for cluster in range(3):
            if len(cluster_indices[cluster]) == 0:
                continue
            cluster_avg_dist = 0
            for i in cluster_indices[cluster]:
                for j in cluster_indices[cluster]:
                    if i != j:
                        cluster_avg_dist += dist_matrix[i, j]
            cluster_avg_dist /= (len(cluster_indices[cluster]) * (len(cluster_indices[cluster]) - 1))
            cluster_distances.append(cluster_avg_dist)
        
        # Select the cluster with the largest average distance
        max_cluster_idx = np.argmax(cluster_distances)
        perturb_indices = cluster_indices[max_cluster_idx]
        
        # Perturb the positions of the selected cluster
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
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    np.random.seed(42)  # For reproducibility
    centers = np.random.rand(n, 2)
    radii = np.full(n, 0.05)
    
    # Cluster centers using k-means to form tight clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=cols, random_state=42).fit(centers)
    cluster_assignments = kmeans.labels_
    
    # Adjust positions to form a grid-like structure within the unit square
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        cluster = cluster_assignments[i]
        col = cluster % cols
        row = cluster // cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Introduce small random variation to break symmetry
        xs[i] = x + np.random.uniform(-0.02, 0.02)
        ys[i] = y + np.random.uniform(-0.02, 0.02)
    
    # Adjust radii based on cluster density
    cluster_radii = np.zeros(cols)
    for i in range(n):
        cluster = cluster_assignments[i]
        cluster_radii[cluster] += 1.0 / (np.sum(1.0 / (np.sum((centers - centers[i])**2, axis=1) + 1e-8)))
    cluster_radii /= np.sum(cluster_radii)
    radii = cluster_radii[cluster_assignments] * 0.3 - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = radii

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

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Perturb the most tightly packed cluster to enable radius expansion
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate cluster assignments based on current positions
        kmeans = KMeans(n_clusters=cols, random_state=42).fit(np.column_stack([centers[0], centers[1]]))
        cluster_assignments = kmeans.labels_
        # Find the cluster with the smallest average distance between circles
        cluster_dists = np.zeros(cols)
        for c in range(cols):
            cluster_indices = np.where(cluster_assignments == c)[0]
            for i in cluster_indices:
                for j in cluster_indices:
                    if i != j:
                        dx = centers[0][i] - centers[0][j]
                        dy = centers[1][i] - centers[1][j]
                        cluster_dists[c] += np.sqrt(dx*dx + dy*dy)
        cluster_dists /= len(cluster_indices)
        tightest_cluster = np.argmin(cluster_dists)
        # Perturb the most tightly packed cluster
        cluster_indices = np.where(cluster_assignments == tightest_cluster)[0]
        perturbation = 0.05 * np.random.rand(len(cluster_indices) * 3)
        perturbed_v = v.copy()
        idx = 0
        for i in cluster_indices:
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
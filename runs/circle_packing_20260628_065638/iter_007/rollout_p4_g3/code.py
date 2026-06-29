import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    np.random.seed(42)
    cluster_centers = np.random.rand(n, 2)
    cluster_centers = (cluster_centers - 0.5) * 2.0  # Center in [-1, 1]
    cluster_radii = np.random.rand(n) * 0.05 + 0.01
    
    # Assign circles to clusters
    clusters = [[] for _ in range(cols)]
    for i in range(n):
        dists = np.linalg.norm(cluster_centers - cluster_centers[i], axis=1)
        nearest = np.argmin(dists)
        clusters[nearest].append(i)
    
    # Initialize positions based on cluster centers
    xs = []
    ys = []
    for i in range(cols):
        for j in clusters[i]:
            x = cluster_centers[j][0]
            y = cluster_centers[j][1]
            # Introduce small random jitter
            x += np.random.uniform(-0.01, 0.01)
            y += np.random.uniform(-0.01, 0.01)
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
    
    # Focus on expanding the most tightly packed cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        
        # Calculate pairwise distances for all circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[0][i] - centers[0][j]
                dy = centers[1][i] - centers[1][j]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find the cluster with the smallest average inter-circle distance
        cluster_distances = np.zeros(cols)
        for i in range(cols):
            cluster = np.array(clusters[i])
            sub_dists = dists[cluster[:, None], cluster]
            cluster_distances[i] = np.mean(sub_dists[np.triu_indices_from(sub_dists, 1)])
        
        # Select the cluster with the smallest average distance to expand
        expand_cluster_index = np.argmin(cluster_distances)
        expand_indices = np.array(clusters[expand_cluster_index])
        
        # Increase radius of circles in the selected cluster
        expand_v = v.copy()
        for i in expand_indices:
            expand_v[3*i + 2] = np.clip(v[3*i + 2] + 0.005, 1e-4, 0.5)
        
        # Perturb positions of circles in the selected cluster
        perturbation = 0.02 * np.random.rand(len(expand_indices) * 3)
        for i in expand_indices:
            expand_v[3*i] += perturbation[0]
            expand_v[3*i+1] += perturbation[1]
            expand_v[3*i+2] += perturbation[2]
            perturbation = perturbation[3:]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, expand_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    cluster_centers = np.random.rand(5, 2)
    cluster_centers = (cluster_centers - 0.5) * 0.6 + 0.5  # Center and scale clusters
    cluster_radii = np.random.uniform(0.05, 0.15, 5)
    cluster_offset = np.random.rand(5, 2) * 0.15 - 0.075  # Randomly offset clusters
    
    # Initialize positions using randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        cluster_idx = i % 5
        x = cluster_centers[cluster_idx, 0] + cluster_offset[cluster_idx, 0]
        y = cluster_centers[cluster_idx, 1] + cluster_offset[cluster_idx, 1]
        # Introduce small random offset to break symmetry
        x += np.random.uniform(-0.03, 0.03)
        y += np.random.uniform(-0.03, 0.03)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.25 / cols - 1e-3
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
    
    # Non-local reconfiguration: identify isolated cluster and expand its radii
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute cluster assignments based on nearest cluster center
        cluster_centers = np.random.rand(5, 2)
        cluster_centers = (cluster_centers - 0.5) * 0.6 + 0.5
        cluster_distances = np.array([np.min(np.sum((centers - cluster_centers[i])**2, axis=1)) for i in range(5)])
        cluster_indices = np.argmin(cluster_distances)
        
        # Select the cluster with the most isolated circles for expansion
        cluster_radii = np.array([np.sum(radii[np.argsort(cluster_distances) == i]) for i in range(5)])
        cluster_radii = cluster_radii / np.sum(cluster_radii)
        cluster_idx = np.argmax(cluster_radii)
        cluster_circles = np.where(cluster_distances == cluster_idx)[0]
        
        # Increase radii of isolated cluster circles
        perturbed_v = v.copy()
        perturbed_v[3*cluster_circles + 2] = np.clip(radii[cluster_circles] * 1.3, 1e-4, 0.5)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
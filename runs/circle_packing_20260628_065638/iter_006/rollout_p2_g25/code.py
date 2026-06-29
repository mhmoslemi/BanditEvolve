import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    # Generate random cluster centers in the square
    cluster_centers = np.random.rand(4, 2)
    cluster_centers = (cluster_centers - 0.5) * 0.8 + 0.5  # Center and scale within [0.1, 0.9]
    cluster_radii = np.full(4, 0.15)  # Initial cluster radii
    
    # Generate circle positions within each cluster
    xs = []
    ys = []
    for cluster_idx in range(4):
        cluster_x, cluster_y = cluster_centers[cluster_idx]
        cluster_r = cluster_radii[cluster_idx]
        # Generate 6-7 circles within the cluster
        for i in range(6 + np.random.choice(1, p=[0.8, 0.2])):
            angle = 2 * np.pi * np.random.rand()
            radius = cluster_r * np.random.rand() * 0.7 + 0.15
            x = cluster_x + radius * np.cos(angle)
            y = cluster_y + radius * np.sin(angle)
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
            xs.append(x)
            ys.append(y)
    
    # Ensure the total number of circles is 26
    while len(xs) < n:
        xs.append(np.random.uniform(0.1, 0.9))
        ys.append(np.random.uniform(0.1, 0.9))
    
    # Initial radius guess - small but positive
    r0 = np.full(n, 0.1)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized overlap constraints
    def vectorized_overlap_constraints(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx ** 2 + dy ** 2
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        return dist_sq - r_sum ** 2

    # Convert to list of constraint functions for scipy.optimize
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Convert vectorized constraints to individual constraint functions
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
    
    # Non-local reconfiguration: identify the most isolated cluster and expand its radii
    if res.success:
        v = res.x
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Compute cluster distances and find the most isolated cluster
        cluster_indices = np.arange(n)
        cluster_distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                cluster_distances[i, j] = dx*dx + dy*dy
                cluster_distances[j, i] = cluster_distances[i, j]
        
        avg_dist = np.mean(cluster_distances[cluster_distances > 0])
        threshold = avg_dist * 2  # Identify isolated clusters
        
        # Find the cluster with the largest average distance to others
        cluster_avgs = np.zeros(n)
        for i in range(n):
            cluster_avgs[i] = np.mean(cluster_distances[i, cluster_distances[i] > 0])
        
        isolated_cluster = np.argmax(cluster_avgs)
        
        # Expand radii of the isolated cluster
        expansion_factor = 1.2
        new_r = r.copy()
        new_r[isolated_cluster] *= expansion_factor
        new_r[isolated_cluster] = np.clip(new_r[isolated_cluster], 1e-4, 0.5)
        
        # Create perturbed v with expanded radii
        perturbed_v = v.copy()
        perturbed_v[2::3] = new_r
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
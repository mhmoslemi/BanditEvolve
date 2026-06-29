import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    def generate_initial_positions(n, cols, rows):
        # Create a grid of cluster centers
        cluster_x = np.linspace(0.1, 0.9, cols)
        cluster_y = np.linspace(0.1, 0.9, rows)
        cluster_centers = np.array([[x, y] for y in cluster_y for x in cluster_x])
        
        # Randomly assign circles to clusters
        cluster_indices = np.random.choice(range(len(cluster_centers)), size=n, replace=True)
        cluster_positions = np.array([cluster_centers[i] for i in cluster_indices])
        
        # Add small random perturbation to break symmetry
        cluster_positions += np.random.uniform(-0.03, 0.03, size=cluster_positions.shape)
        
        # Ensure positions are within the unit square
        cluster_positions = np.clip(cluster_positions, 0.0, 1.0)
        
        return cluster_positions
    
    initial_positions = generate_initial_positions(n, cols, rows)
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = initial_positions[:, 0]
    v0[1::3] = initial_positions[:, 1]
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
    
    # Post-optimization: identify the most isolated cluster and expand its radii
    if res.success:
        v = res.x
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Calculate distances from each circle to all other circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dists[i, j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
        
        # Identify the cluster with the largest minimum distance to other circles
        min_distances = np.min(dists, axis=1)
        isolated_indices = np.argsort(min_distances)[-3:]  # Select top 3 most isolated clusters
        
        # Increase radii of the most isolated clusters while respecting the constraints
        for i in isolated_indices:
            current_r = r[i]
            # Try to expand radius while maintaining constraints
            new_r = np.min([0.5, current_r + 0.01])  # Conservative expansion
            # Apply the expansion in a controlled manner
            perturbed_v = v.copy()
            perturbed_v[3*i+2] = new_r
            # Re-optimize with this perturbation
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
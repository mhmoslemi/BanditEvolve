import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols

    # Initialize positions with geometric hashing for randomized but structured distribution
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    # Ensure that positions are spread over the unit square
    xs *= 0.95
    ys *= 0.95
    # Convert to normalized coordinates [0, 1]
    xs = (xs - 0.5) * 0.9 + 0.5
    ys = (ys - 0.5) * 0.9 + 0.5
    # Create a grid of base positions
    grid_xs = np.linspace(0.1, 0.9, cols)
    grid_ys = np.linspace(0.1, 0.9, rows)
    grid_centers = np.array([grid_xs, grid_ys]).T
    # Use k-means to cluster the randomized points around the grid cells
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n, random_state=42).fit(np.column_stack([xs, ys]))
    grid_indices = kmeans.labels_
    # Assign positions to nearest grid cell centers
    final_xs = np.zeros(n)
    final_ys = np.zeros(n)
    for i in range(n):
        center = grid_centers[grid_indices[i]]
        final_xs[i] = center[0] + np.random.uniform(-0.05, 0.05)
        final_ys[i] = center[1] + np.random.uniform(-0.05, 0.05)
    # Ensure points are not too close to the boundaries
    final_xs = np.clip(final_xs, 0.01, 0.99)
    final_ys = np.clip(final_ys, 0.01, 0.99)

    # Initialize radii with a reasonable starting value
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = final_xs
    v0[1::3] = final_ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    cons = []
    for i in range(n):
        # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            # Create a constraint function that takes the full decision vector and returns the constraint
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, constraints=cons, 
                   options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-10})
    
    # Radically reconfigure through geometric hashing
    if res.success:
        v = res.x
        # Apply spatial hashing and perturbation
        hash_offset = np.random.rand(n, 2) * 0.08
        new_v = v.copy()
        new_v[0::3] += hash_offset[:, 0]
        new_v[1::3] += hash_offset[:, 1]
        
        # Re-optimize with new configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds, constraints=cons, 
                       options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-10})
    
    # Targeted expansion on the smallest circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances and identify least constrained circle
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find the circle with the smallest non-zero radius
        smallest_radii = np.where(radii > 0, radii, np.inf)
        smallest_radius_idx = np.argmin(smallest_radii)
        
        # Calculate expansion factor to control how much to increase the smallest circle's radius
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008  # Aim for a modest increase
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create adjusted radius vector with adjacency-based expansion
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * 1.5  # Slight over-expansion to trigger reconfiguration
        for i in range(n):
            if i != smallest_radius_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-optimize with new radius configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds, constraints=cons, 
                       options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-10})
    
    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
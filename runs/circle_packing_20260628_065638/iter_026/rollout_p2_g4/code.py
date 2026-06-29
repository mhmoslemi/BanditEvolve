import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with enhanced staggered grid and dynamic offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small random offset with dynamic variation
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
        # Alternate row staggering for better spacing
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius based on dynamic spacing and padding
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds consistent with 3*n parameters
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Boundary constraints
    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints (vectorized)
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # First optimization with dense constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-10})
    
    # Asymmetric spatial reconfiguration with stochastic spatial hash
    if res.success:
        v = res.x
        # Create stochastic spatial perturbation with increased variation
        spatial_hash = np.random.rand(n, 2) * 0.07
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        # Re-evaluate
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})
    
    # Targeted radius expansion with isolation assessment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute vectorized distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        isolation_scores = np.min(dists, axis=1)  # Minimum distance to other circles
        least_constrained_idx = np.argmax(isolation_scores)
        
        # Calculate expansion factor
        current_sum = np.sum(radii)
        target_total_sum = current_sum + 0.009  # Conservative expansion to avoid overlap
        expansion_factor = (target_total_sum - current_sum) / (n - 1)
        
        # Apply expansion with careful radial enforcement
        new_radii = radii.copy()
        expanded_v = v.copy()
        
        # Expand least constrained circle first
        new_radii[least_constrained_idx] = np.clip(radii[least_constrained_idx] + expansion_factor * 1.2, 1e-4, 0.5)
        
        # Apply expansion to others with slight variation
        for i in range(n):
            if i != least_constrained_idx:
                # Add slight stochastic variation to expansion
                new_radii[i] = np.clip(radii[i] + expansion_factor * (1 + np.random.rand() * 0.3), 1e-4, 0.5)
        
        # Apply expansion and re-evaluate
        expanded_v[2::3] = new_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})
    
    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
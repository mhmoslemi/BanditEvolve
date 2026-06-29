import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with adaptive geometric tiling and asymmetric distribution
    xs = []
    ys = []
    
    # Create staggered grid with randomized jitter
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Adaptive jitter: scale with row position and column proximity
        row_jitter_factor = 1.0 / (rows + 1)
        col_jitter_factor = 1.0 / (cols + 1)
        x_jitter = np.random.uniform(-row_jitter_factor * 0.25, row_jitter_factor * 0.25)
        y_jitter = np.random.uniform(-col_jitter_factor * 0.25, col_jitter_factor * 0.25)
        
        # Randomized offset for symmetry breaking
        x = x_center + x_jitter
        y = y_center + y_jitter
        
        # Stagger alternate rows to avoid alignment
        if row % 2 == 1:
            x += 0.5 / cols
        
        # Check for edge condition if needed for boundary optimization
        xs.append(x)
        ys.append(y)
    
    # Base radius estimation with adaptive scaling and edge case handling
    edge_radius_scaling = 1.0 / (cols + rows + 2)
    r0 = 0.4 * edge_radius_scaling - 1e-3
    # Additional safety: ensure minimal radius doesn't fall below 1e-4
    r0 = max(r0, 1e-4)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for the decision vector (3*n entries)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint functions
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with optimized numerical evaluation
    for i in range(n):
        for j in range(i + 1, n):
            # Use direct evaluation to minimize lambda capture issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11, "eps": 1e-12})
    
    # Geometric reconfiguration with adaptive directional hashing and dynamic spatial perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate spatial hashing with row/column dependency for targeted perturbation
        # Create spatial hash based on relative position and radius
        spatial_hash = np.random.rand(n, 2) * 0.005
        # Additional directional bias based on row parity
        directional_bias = np.zeros(n)
        for i in range(n):
            row = i // cols
            if row % 2 == 1:
                directional_bias[i] = 0.01
        perturbed_v = v.copy()
        
        for i in range(n):
            # Apply directional perturbation based on row parity and spatial hash
            if i % 2 == 0:
                perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * (1.0 + directional_bias[i] * 0.1)
                perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * (1.0 + directional_bias[i] * 0.15)
            else:
                perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * (1.0 + directional_bias[i] * 0.15)
                perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * (1.0 + directional_bias[i] * 0.1)
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})

    # Targeted radius expansion on least constrained circle with dynamic growth budget
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Adaptive growth budget - use 0.0065 for total sum increase with directional boost
        current_total = np.sum(radii)
        target_growth = 0.0065
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii)) * 1.2
        
        # Create expansion vector with directional perturbation
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1  # Over-expansion to force rebalancing
        for i in range(n):
            if i != least_constrained_idx:
                # Directional expansion with row-specific bias
                row = i // cols
                if row < least_constrained_idx // cols:
                    expansion_i = expansion_factor * 0.8
                elif row > least_constrained_idx // cols:
                    expansion_i = expansion_factor * 1.0
                else:
                    expansion_i = expansion_factor * 1.2  # Row level boost
                # Apply spatial hashing-aware expansion
                if i % 2 == 0:
                    expansion_i *= 1.05
                else:
                    expansion_i *= 0.95
                # Add stochastic perturbation for fine tuning
                expansion_i += np.random.uniform(-expansion_i * 0.05, expansion_i * 0.05)
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation using optimized validation
        def validate_expanded_config(new_radii, centers):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        return False
            return True
        
        # Iteratively apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check if configuration is valid
            valid = validate_expanded_config(new_radii, expanded_centers)
            if valid:
                break
            else:
                # Reduce expansion by 2% per invalid iteration
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and reconfigured spatial setup
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 100, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
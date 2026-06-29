import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initial geometric setup with randomized staggered grid and adaptive bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Adaptive spatial bias: clusters are denser at edges to leverage symmetry breaking
        # Horizontal shift for staggered grid with row-specific scaling
        row_scale_factor = 0.4 + 0.05 * row
        x_center += np.random.uniform(-0.1 * row_scale_factor, 0.1 * row_scale_factor)
        y_center += np.random.uniform(-0.1, 0.1)
        
        if row % 2 == 1:  # Alternate rows are offset to break alignment
            x_center += 0.45 / cols
        
        xs.append(x_center)
        ys.append(y_center)
    
    r0 = 0.3 / cols + 1e-3  # Initial radius increased to explore feasibility space
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints using lambda with explicit i capture
    cons = []
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with dynamic constraint weighting and vectorized calculation
    for i in range(n):
        for j in range(i + 1, n):
            # Dynamic constraint weight to prioritize tightest packing first
            # Using vectorized calculation for speed and memory
            dx = v0[3*i] - v0[3*j]
            dy = v0[3*i+1] - v0[3*j+1]
            r_sum = v0[3*i+2] + v0[3*j+2]
            distance_squared = dx*dx + dy*dy
            # Overlap constraint: distance^2 >= (r1 + r2)^2
            # We'll use this expression with a small epsilon to avoid numerical issues
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: distance_squared - (v[3*i+2] + v[3*j+2])**2 })

    # Initial optimization with adaptive iteration and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-9})
    
    # Post-optimization reconfiguration - asymmetric spatial reshaping
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dist_matrix = np.zeros((n, n))
        
        # Vectorized distance matrix calculation for efficient constraint re-evaluation
        dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
        dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # Identify circles most constrained by overlapping neighbors
        avg_dists = np.mean(dist_matrix, axis=1)
        avg_dists[avg_dists < 1e-10] = 1  # Avoid division by zero
        constraint_weights = 1 / avg_dists  # Higher constraint weight on denser circles

        # Randomized geometric tiling for spatial disruption
        tile_pattern = np.random.rand(n, n)
        tile_pattern[0] = 0  # Fixed first circle to preserve initial layout integrity
        tile_pattern = tile_pattern / (1 + np.sum(tile_pattern, axis=1, keepdims=True))  # Normalize
        
        # Reconfigure constrained circles
        for i in range(n):
            if i == 0: continue
            # Apply geometric tile displacement with adaptive scale based on constraint weight
            scale = 0.03 * constraint_weights[i]
            v[3*i] += tile_pattern[i, 0] * scale * radii[i]
            v[3*i+1] += tile_pattern[i, 1] * scale * radii[i]
        
        # Re-evaluate with spatial reconfiguration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})
    
    # Advanced expansion phase with multi-objective radius adjustment
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Compute global expansion potential with spatial and radial balance
        # Compute distances to all others with vectorization
        dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
        dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # Compute expansion feasibility matrix: potential growth for each circle
        # Based on min distance to any other, scaled by local density
        min_dist = np.min(dist_matrix[np.triu_indices(n, 1)], axis=1)
        min_dist[min_dist < 1e-10] = 1  # Avoid division by zero
        expansion_potential = (min_dist - radii) / (np.std(radii) + 1e-6)  # Normalized by standard deviation
        
        # Apply targeted radius expansion with constraints
        expansion_budget = 0.006
        expansion_vector = np.zeros(n)
        for i in range(n):
            if expansion_potential[i] > 1e-4:  # Only expand if there's room for growth
                expansion_vector[i] = np.clip(expansion_budget * (expansion_potential[i] / np.sum(expansion_potential)), 0, expansion_budget)
        
        # Construct new_v and enforce bounds
        new_radii = radii + expansion_vector
        
        # Apply soft constraint expansion with validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, moderate expansion by a decay factor
                new_radii = radii + (new_radii - radii) * 0.95
        
        v = expanded_v.copy()
        
        # Re-evaluate with updated radii configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-8})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
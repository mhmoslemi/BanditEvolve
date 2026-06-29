import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a refined geometric hashing approach
    xs = []
    ys = []

    # Adaptive geometric hashing with row-dependent scaling
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = col / cols + 0.5 / cols
        y_center = row / rows + 0.5 / rows
        
        # Row-dependent offset scaling to improve space utilization
        row_factor = np.cos(np.pi * row / rows)  # sinusoidal offset for better density
        col_factor = np.sin(np.pi * col / cols)  # sinusoidal offset for better density
        
        # Add geometric hash for fine-grained position perturbation
        x_offset = np.random.uniform(-0.02 * row_factor, 0.02 * row_factor)
        y_offset = np.random.uniform(-0.02 * col_factor, 0.02 * col_factor)
        
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Alternate row staggering with spatial-aware offset
        if row % 2 == 1:
            # Add row-dependent stagger for non-uniform grid
            x += (0.45 / cols) * (row_factor / (row + 1)) if rows > row + 1 else 0
            
        xs.append(x)
        ys.append(y)

    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i and explicit closures
    cons = []
    for i in range(n):
        def make_boundary_constraint(direction: str, idx: int):
            def constraint(v):
                if direction == 'left':
                    return 1.0 - v[3*idx] - v[3*idx+2]
                elif direction == 'right':
                    return v[3*idx] - v[3*idx+2]
                elif direction == 'bottom':
                    return 1.0 - v[3*idx+1] - v[3*idx+2]
                elif direction == 'top':
                    return v[3*idx+1] - v[3*idx+2]
                else:
                    raise ValueError("Invalid direction")
            return constraint
        cons.append({"type": "ineq", "fun": make_boundary_constraint('left', i)})
        cons.append({"type": "ineq", "fun": make_boundary_constraint('right', i)})
        cons.append({"type": "ineq", "fun": make_boundary_constraint('bottom', i)})
        cons.append({"type": "ineq", "fun": make_boundary_constraint('top', i)})
    
    # Overlap constraints with geometric hashing and spatial regularization
    for i in range(n):
        for j in range(i+1, n):
            def make_overlap_constraint(i, j):
                def constraint(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    # Use vectorized distance squared minus (r1 + r2)^2 with regularization
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2 + 1e-12 * (v[3*i+2] + v[3*j+2])  # regularization
                return constraint
            cons.append({"type": "ineq", "fun": make_overlap_constraint(i, j)})

    # Primary optimization: high precision and adaptive constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11})
    
    # Post-optimization spatial reconstruction with advanced hashing and dynamic expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Reconstruct with spatial hash + density-aware perturbation
        perturb_matrix = np.random.rand(n, 2) * 0.02 * np.sqrt(np.sum(radii)) * (1.0 + 0.1 * np.random.rand())
        perturbed_v = v.copy()
        
        for i in range(n):
            # Spatial-aware perturbation with directional bias
            direction = 1 if (i // cols) % 2 == 1 else -1
            perturbed_v[3*i] += perturb_matrix[i, 0] * (1.0 + 0.2 * (np.random.rand() * direction))
            perturbed_v[3*i+1] += perturb_matrix[i, 1] * (1.0 + 0.2 * (np.random.rand() * direction))
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion with geometric-aware gradient scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Precompute pairwise distances for efficient access
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Efficiently compute minimal distance to others (avoiding loops)
        min_dists = np.min(dists, axis=1)
        # Score based on density: circles with higher min_distances are more constrained
        constraint_scores = np.reciprocal(min_dists + 1e-10) * np.sqrt(np.mean(radii))
        
        # Choose circle to expand - prioritize those with low constraint score
        least_constrained_idx = np.argmin(constraint_scores) if np.any(constraint_scores) else 0
        
        # Calculate current total sum
        current_total = np.sum(radii)
        # Determine possible expansion based on remaining space
        max_possible_growth = 0.0065  # Based on historical SOTA + margin
        expansion_target = current_total + max_possible_growth
        
        # Estimate available capacity by finding maximal radius for least constrained
        # This is an approximation but ensures a safer expansion
        max_possible_radius = min(0.5, 1.0 - np.max(centers[:, 0] + radii) if centers[:, 0] + radii <= 1 else 0.5)
        max_grow = -radii[least_constrained_idx] if radii[least_constrained_idx] < max_possible_radius else 0
        total_possible_growth = max_grow * n

        # Calculate growth multiplier
        growth_multiplier = np.clip((expansion_target - current_total) / (total_possible_growth + 1e-10), 0, 1)
        
        # Build expansion vector with soft growth and regularization
        expansion = np.zeros(n)
        expansion[least_constrained_idx] = growth_multiplier * max_grow
        # Distribute residual expansion with row-based regularization
        for i in range(n):
            if i == least_constrained_idx:
                continue
            # Add slightly more growth to lower density regions based on row index
            row = i // cols
            expansion[i] += growth_multiplier * np.sqrt(1.0 - row / rows) * np.random.uniform(0.5, 1.3)
        
        new_radii = radii + expansion
        new_radii = np.clip(new_radii, 1e-6, 0.5)
        
        # Build new decision vector with soft constraint relaxation
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Re-evaluate with soft constraint relaxation and higher precision
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
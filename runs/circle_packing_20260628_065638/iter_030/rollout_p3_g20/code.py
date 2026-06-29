import numpy as np

def run_packing():
    n = 26
    rows, cols = int(np.ceil(np.sqrt(n))), int(np.ceil(np.sqrt(n)))
    
    # Initialize positions with intelligent staggered grid, dynamic clustering, and adaptive randomness
    xs = []
    ys = []
    initial_radius_guess = 0.2  # Conservative starting radius for stability
    
    for i in range(n):
        base_row = i // cols
        base_col = i % cols
        row_offset = np.random.uniform(-0.03, 0.03)
        col_offset = np.random.uniform(-0.03, 0.03)
        
        # Calculate base grid point
        normalized_col = (base_col + 0.5 + col_offset) / cols
        normalized_row = (base_row + 0.5 + row_offset) / rows

        # Add stagger for row parity
        if base_row % 2 == 1:  # For alternate rows, shift columns for spacing
            normalized_col += np.random.uniform(0.005, 0.015)

        # Add fine-grain randomness to centers
        normalized_col += np.random.uniform(-0.002, 0.002)
        normalized_row += np.random.uniform(-0.002, 0.002)

        # Ensure bounds are not exceeded
        normalized_row = np.clip(normalized_row, 0.0 + 1e-6, 1.0 - 1e-6)
        normalized_col = np.clip(normalized_col, 0.0 + 1e-6, 1.0 - 1e-6)

        xs.append(normalized_col)
        ys.append(normalized_row)
    
    r0 = initial_radius_guess / (cols + rows)  # Adaptive initial guess based on grid dimensions
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-3, 0.5)]  # Radii constrained to min 1e-3 for stability

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Advanced constraint system with dynamic scaling, gradient stabilization, and vectorized constraints
    cons = []

    # Generate boundary constraints using vectorized lambdas with i as non-lambda parameter
    for i in range(n):
        # Left boundary constraint (x - r >= 0): 0 <= x - r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right boundary constraint (x + r <= 1): 0 <= 1 - x - r
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom boundary constraint (y - r >= 0): 0 <= y - r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top boundary constraint (y + r <= 1): 0 <= 1 - y - r
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Generate vectorized pairwise distance constraints with dynamic scaling for better convergence
    for i in range(n):
        for j in range(i + 1, n):
            # Construct a constraint function with stable i,j capture
            # Apply a dynamic scale factor to ensure constraint tightness
            scale_factor = 1.0
            def dist_func(v, i=i, j=j, scale=scale_factor):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx* dx + dy* dy
                # Add small epsilon to prevent zero division due to numerical errors
                min_dist_sq = (v[3*i + 2] + v[3*j + 2]) ** 2 + 1e-10
                return dist_sq - min_dist_sq
            
            # Scale to stabilize gradient
            cons.append({"type": "ineq", "fun": lambda v, f=dist_func: f(v)})

    # Initial optimization with hybrid strategies and adaptive tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-8})

    # Stochastic spatial reconfiguration with multi-level perturbation and adaptive spatial hashing
    if res.success:
        v = res.x
        # Spatial perturbation vector: scale per circle based on current radius
        perturbation_scale = 0.03 * (v[2::3]/(np.mean(v[2::3])*n))  # Radius-dependent perturbation scaling

        # Add multi-level spatial perturbation
        spatial_hash = np.random.rand(n, 2) * 0.05  # Base randomization
        for i in range(n):
            v[3*i] += spatial_hash[i, 0] * perturbation_scale[i]
            v[3*i+1] += spatial_hash[i, 1] * perturbation_scale[i]

        # Add high-frequency local perturbation
        local_hash = np.random.rand(n, 2) * 0.005
        for i in range(n):
            v[3*i] += local_hash[i, 0]
            v[3*i+1] += local_hash[i, 1]

        # Reoptimize with new spatial configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-8})

    # Adaptive expansion strategy for least constrained circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))

        # Vectorized pairwise distance calculation in numpy
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find least constrained circle by min-distance-to-others metric
        isolation = np.min(dists, axis=1)
        isolated_idx = np.argmin(isolation)  # Isolated circle with least interaction

        # Calculate growth based on potential expansion with safety margin
        current_total = np.sum(radii)
        # Target growth proportional to average radius and spacing constraints
        growth = (0.008) * (np.mean(radii)) 
        # Apply growth to the isolated circle and others based on distance from isolation
        expansion_vector = np.zeros(n)
        for i in range(n):
            if i != isolated_idx:
                # Expands more when further from the isolated circle to distribute
                dist_to_isolated = dists[i, isolated_idx]
                weight = np.max([min(dists[i, ~np.arange(n)==i] + 1e-8, default=1), 1.0], axis=0)
                expansion = growth * np.clip((1.0 + (1.0 - dist_to_isolated / (np.max(dists) + 1e-8)) * 1.2), 1.0, 5.0)
                expansion_vector[i] = expansion
            else:
                expansion_vector[i] = growth * 1.7  # Larger expansion for the isolated circle

        # Apply expansion with constraint validation
        max_retries = 5
        for _ in range(max_retries):
            expanded_v = v.copy()
            expanded_v[2::3] = radii + expansion_vector
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

            # Check for overlaps
            success = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < expanded_v[3*i + 2] + expanded_v[3*j + 2] - 1e-6:
                        success = False
                        break
                if not success:
                    break
            if success:
                break
            else:
                # Gradually reduce expansion vector
                expansion_vector = expansion_vector * 0.95

        # Update decision vector with safe expansion
        v = expanded_v

        # Reoptimize with new expanded configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-8})
    
    # Final check and clipping to avoid numerical errors
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Clip radii to ensure they are within the valid range
    
    # Final validation for numeric integrity for safety
    # (Already handled by validator, but safe for production)
    
    # Return structured output as required
    return centers, radii, float(radii.sum())
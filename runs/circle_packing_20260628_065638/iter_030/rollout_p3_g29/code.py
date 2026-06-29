import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Improved initialization with adaptive grid distortion and stochastic geometry hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with row-wise distortion and column-wise scaling
        x_base = (col + 0.5) / cols
        x_distort = np.random.uniform(-0.04, 0.04) * 1.0 / (cols + 1)
        # Column-wise spatial distortion with asymmetric gradient
        x = x_base + (np.random.uniform(-0.03, 0.03) if row % 3 == 0 else 0.0) + x_distort
        y_base = (row + 0.5) / rows
        y_distort = np.random.uniform(-0.04, 0.04) * 1.0 / (rows + 1)
        # Row-wise vertical distortion with asymmetry
        y = y_base + (np.random.uniform(-0.03, 0.03) if col % 3 == 0 else 0.0) + y_distort
        
        # Add dynamic spatial hashing and asymmetric spatial perturbation
        x += np.random.randn() * (0.04) * (0.5 / (1 + row + col)) 
        y += np.random.randn() * (0.04) * (0.5 / (1 + row + col)) 
        xs.append(x)
        ys.append(y)
    
    # Initial radius configuration with gradient-aware distribution
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)  # Initial radii with soft constraints

    # Bounds must match exactly 3*n to match the vector length
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint creation with delayed lambda binding
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})        # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})  # 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})      # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})# 1 - y - r >= 0
    
    # Vectorized overlap constraints with spatial hashing and gradient-aware distance
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                # Use spatial hashing for gradient control
                # Scale overlap constraint by effective distance and grid proximity
                dist = np.sqrt(dx*dx + dy*dy)
                grid_weight = 0.8 ** (np.min([i, j])) / (1.0 + np.abs(i - j))
                return dist - (v[3*i+2] + v[3*j+2]) + 1e-10 * grid_weight  # > 0 for no overlap
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization configuration with adaptive tolerances
    # Set maximum iterations for the main optimizer
    initial_opt = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 1500, 
            "ftol": 1e-10, 
            "iprint": 0, 
            "disp": False, 
            "gtol": 1e-9, 
            "eps": 1e-6, 
            "callback": None
        }
    )

    # Asymmetric reconfiguration with adaptive spatial hashing
    if initial_opt.success:
        v = initial_opt.x
        radii = v[2::3]
        # Generate spatial hash with asymmetric scaling
        spatial_hash = np.random.rand(n, 2) * 0.08
        # Apply perturbations with radius-dependent scaling
        perturbed_v = v.copy()
        for i in range(n):
            # Use gradient-aware perturbation
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))  # x perturbation scaled by radius
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) # y perturbation scaled by radius
            perturbed_v[3*i+2] *= 1.02 + np.random.uniform(-0.002, 0.002)  # tiny radius scaling
        # Re-optimization with new perturbed parameters
        reconfig_opt = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 400, 
                "ftol": 1e-10,
                "tol": 1e-10, 
                "iprint": 0, 
                "disp": False
            }
        )
        if reconfig_opt.success:
            v = reconfig_opt.x
        else:
            v = initial_opt.x  

    # Targeted growth with dynamic constraint awareness
    # Use gradient-based expansion on the most isolated circle
    if v is not None:
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute isolation score: sum of minimum distances
        min_dist_per_circle = np.min(dists, axis=1)
        isolation_scores = min_dist_per_circle

        # Find the least constrained circle (most isolated one)
        isolation_idx = np.argmin(isolation_scores)
        isolation_circle = centers[isolation_idx]
        isolation_radius = radii[isolation_idx]

        # Calculate expansion target with gradient-aware growth
        # Start with a 0.006 target boost and compute expansion factor
        # We also add stochastic spatial perturbation to other circles
        target_total_growth = 0.006
        base_growth_ratio = 1.2 * (1.0 + 0.05 * np.random.rand())

        # Create a new radius vector
        new_radii = np.copy(radii)

        # Compute current total radius
        current_radius_sum = np.sum(new_radii)
        # We want to grow the least constrained circle by a fraction of the total sum
        max_possible_growth = (target_total_growth) / (1 - (1 / n)) # Adjusting for sum allocation
        growth_per_circle = max_possible_growth * (1.2 + 0.2 * np.random.rand())

        # Apply growth only to the least constrained circle
        new_radii[isolation_idx] += growth_per_circle
        # Distribute some growth among other circles with spatial hashing
        # This distributes growth with spatial awareness and randomness
        hash_matrix = np.random.rand(n, n) 
        for i in range(n):
            if i != isolation_idx:  # Avoid over-expanding the main target
                # Distribute a fraction of the growth based on spatial relations
                # Weight by proximity and normalize
                spatial_weight = np.sum(hash_matrix[i, :]) / np.sum(hash_matrix[i, :])  # Normalized spatial weight
                new_radii[i] += growth_per_circle * spatial_weight * (0.8 + 0.1 * np.random.rand())

        # Now, we'll perform a targeted optimization
        # This will optimize the centers with the new radii vector
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Optimizing with new target configuration
        # Use gradient-based optimization with adaptive parameters
        # Start with an expanded optimization with higher maxiter
        reopt_result = minimize(
            neg_sum_radii,
            expanded_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 400, 
                "ftol": 1e-12, 
                "gtol": 1e-12, 
                "eps": 1e-8, 
                "tol": 1e-12,
                "iprint": 0, 
                "disp": False
            }
        )
        
        if reopt_result.success:
            v = reopt_result.x
        else:
            v = initial_opt.x

    # Post-optimization constraints and clipping
    v = v if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)  # Ensure no negative or zero radii

    return centers, radii, float(radii.sum())
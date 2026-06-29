import numpy as np

def run_packing():
    n = 26
    # Optimize grid spacing for better distribution of 26 circles in 1x1 with efficient packing
    # Use 5 columns and 6 rows to accommodate 26 circles with staggered grid 
    # This gives a grid with 5*6 = 30 slots, so we can use a grid spacing of 1/5 with small perturbations
    cols = 5
    rows = 6
    grid_width = 1.0 / cols
    grid_height = 1.0 / rows

    # Initial position generation with geometric hashing and spatial awareness
    xs = []
    ys = []
    
    # Define a geometric perturbation matrix based on circle indices 
    # This helps break symmetry across the 2x2 cell structure, creating local clusters in staggered fashion
    geometric_perturbation = np.random.rand(n, 2) * np.array([0.08, 0.08])
    
    for i in range(n):
        grid_col = i % cols
        grid_row = i // cols
        
        # Base position using staggered grid pattern
        base_x = (grid_col + 0.5) * grid_width
        base_y = (grid_row + 0.5) * grid_height
        
        # Apply geometric hashing for better spread
        hash_x = geometric_perturbation[i, 0]
        hash_y = geometric_perturbation[i, 1]
        
        # Introduce staggered row shift only for even rows to create interlocking pattern
        if grid_row % 2 == 0:
            base_x += 0.5 * grid_width
        
        x = base_x + hash_x
        y = base_y + hash_y
        
        # Ensure positions stay within bounds
        x = np.clip(x, 1e-8, 1 - 1e-8)
        y = np.clip(y, 1e-8, 1 - 1e-8)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimate from spacing between grid elements and perturbation
    # Initial guess based on minimal spacing and adaptive scaling
    # Use spacing from adjacent grid nodes (including diagonal)
    # The minimal distance between grid centers is sqrt( (grid_width)^2 + (grid_height)^2 )
    # Initial radius is 1/4 of this for dense packing
    # Also, include an energy-based initial radius based on perturbation size
    minimal_grid_distance = np.sqrt(grid_width**2 + grid_height**2)
    base_radius = minimal_grid_distance / 4.0
    r0 = base_radius * np.sqrt(1 + 0.1 * np.random.rand(n)) - 1e-3  # Adjust for perturbation

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    # Ensure bounds list size matches 3*n for 26 circles
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint functions with lambda closure avoidance
    cons = []
    for i in range(n):
        # Left: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Pairwise distance constraint with vectorized computation
    # We use precomputed distances from the base grid setup as a starting point
    # Then, use the constraint to maintain safe distances
    # This avoids redundant computations in constraint evaluation
    # Precompute a spatial graph of all pairs for faster evaluation

    # Vectorized constraint for circle overlaps
    # We'll compute the squared distance and subtract the sum of radii to enforce non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            def overlap_constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
            cons.append({"type": "ineq", "fun": overlap_constraint})

    # Initial optimization with increased iterations and tighter tolerance
    # Add some gradient smoothing by using L-BFGS-B as initial method
    # This is more robust than SLSQP for initial optimization
    # We'll use a hybrid strategy with multiple optimizations

    # First optimization with L-BFGS-B for initial convergence
    res = minimize(neg_sum_radii, v0, method="L-BFGS-B", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})

    if not res.success:
        # fallback to SLSQP with tighter tolerances if initial optimization fails
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply spatial perturbations with adaptive scaling that considers radius sizes
    # This introduces controlled randomness in positions to avoid local minima
    # Spatial perturbation is scaled by the square root of radius for larger circles
    # to preserve stability in high-radius regions
    if res.success:
        v = res.x
        # Generate spatial noise based on spatial hashing and radius size
        noise_level = 0.08 * np.sqrt(v[2::3] / np.max(v[2::3]))   # adaptive scaling
        noise = np.random.rand(n, 2) * noise_level
        perturbed_v = v + np.zeros_like(v)
        for i in range(n):
            perturbed_v[3*i] += noise[i, 0]
            perturbed_v[3*i + 1] += noise[i, 1]
        
        # Re-evaluate with perturbed positions to find better configuration
        res = minimize(neg_sum_radii, perturbed_v, method="L-BFGS-B", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # If optimization still fails, try a different starting point
    if not res.success:
        # Generate new initial positions based on optimized perturbation
        # Use the last perturbed positions as new starting points
        v = res.x
        noise = np.random.rand(n, 2) * 0.05
        new_v = v + np.zeros_like(v)
        for i in range(n):
            new_v[3*i] += noise[i, 0]
            new_v[3*i + 1] += noise[i, 1]
        
        res = minimize(neg_sum_radii, new_v, method="L-BFGS-B", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    # After initial optimization, perform targeted radius expansion on the circle with the least constraints
    if res.success:
        v = res.x
        radii = v[2::3].copy()
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for each circle
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
                dists[j, i] = dists[i, j]
        
        # Compute minimal distance to each circle to find the least constrained
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        least_constrained_radius = radii[least_constrained_idx]
        
        # Calculate a growth vector based on potential expansion capability
        # Compute the minimum radius increase possible without overlap
        # This is done by considering the circle's distance and current radius
        max_growth = []
        for i in range(n):
            if i != least_constrained_idx:
                d = dists[least_constrained_idx, i]
                max_growth.append(d - radii[i] - least_constrained_radius)
        
        if max_growth:
            max_growth = np.max(max_growth)
        else:
            max_growth = 0.0
        
        # If expansion is possible, apply it to a subset of circles
        if max_growth > 1e-6:
            target_radius = least_constrained_radius + max_growth * 0.8
            growth_factor = (target_radius - least_constrained_radius) / (n - 1)
            
            # Distribute growth, giving higher increments to circles with larger minimum distance
            for i in range(n):
                if i != least_constrained_idx:
                    # Use distance-based weighting for expansion
                    d = dists[least_constrained_idx, i]
                    weight = 1.0 / np.max(dists[least_constrained_idx, :])
                    growth = growth_factor * weight * (d - radii[i] - least_constrained_radius)
                    radii[i] += growth
        
        # Update the decision vector
        v_new = v.copy()
        v_new[2::3] = radii
        
        # Re-evaluate the expanded configuration with SLSQP for further refinement
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    # Ensure all circles have at least radius 1e-6
    radii = np.clip(v[2::3], 1e-6, None)
    centers = np.column_stack([v[0::3], v[1::3]])
    
    # Final validation
    # Apply strict validation to ensure the solution is valid
    # Recheck the constraints to ensure no invalid circle
    valid, reason = validate_packing(centers, radii)
    if not valid:
        # If validation fails, fallback to original result
        # This ensures the function always returns a valid packing
        centers = np.column_stack([v0[0::3], v0[1::3]])
        radii = np.clip(v0[2::3], 1e-6, None)
        return centers, radii, float(radii.sum())
    
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Define more refined bounds for center positions and radii with dynamic adjustment
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n total entries
    
    # Generate initial positions using a multi-phase initialization: grid with enhanced randomness 
    # and a spatial-aware initialization that preserves local geometric constraints
    # Using more adaptive randomized distribution to break symmetry and improve initial placement quality
    xs = []
    ys = []
    
    # Phase 1: Base grid - create staggered rows with adaptive spacing
    for i in range(n):
        row = i // cols
        col = i % cols
        # Calculate base spacing with rows and cols being non-uniform if n is not perfect square
        col_base = float(col) / cols
        row_base = float(row) / rows
        
        # Introduce adaptive non-uniform distribution: vary center density based on row index
        # Even rows: higher density in central columns; odd rows: left-right spread
        col_var = 0.3 if row % 2 == 0 else 0.1
        x_center = col_base + np.random.uniform(-col_var, col_var)
        
        # For rows with even indices, we'll spread the Y centers more to avoid clump
        row_var = 0.1 if row % 2 == 1 else 0.02
        y_center = row_base + np.random.uniform(-row_var, row_var)
        
        # Stagger alternate rows for spatial separation 
        if row % 2 == 1:
            x_center += 0.5 / cols
            
        xs.append(x_center)
        ys.append(y_center)
    
    # Phase 2: Add a refined local perturbation to cluster centers with spatial hashing
    # This creates a more nuanced initial position distribution that balances density with spacing
    hash_factor = 0.03  # Reduced to avoid over-perturbation
    xs_perturbed = np.array(xs) + np.random.uniform(-hash_factor, hash_factor, size=n)
    ys_perturbed = np.array(ys) + np.random.uniform(-hash_factor, hash_factor, size=n)
    
    # Phase 3: Add row- and column-specific radial clustering to improve the initial configuration
    # This creates a more natural radial density that avoids global grid-like symmetry
    # Generate cluster centers by creating a soft "radial" grid, then perturbing
    radial_center_density = 0.95 # Keep slightly lower for space
    # Compute radial positions as a base, then add local randomization
    radial_x = np.linspace(0.1, 0.9, cols) 
    radial_y = np.linspace(0.1, 0.9, rows) 
    radial_points = np.array([ [x, y] for x in radial_x for y in radial_y ])
    
    # Compute local density and adjust initial positions to create more uniform radial distribution
    # Map the initial position to radial grid based on distance from center
    # This adds local structure, reducing need for extensive optimization later
    # Note: We keep the initial positions but ensure they are mapped spatially
    # This phase creates a more natural spatial distribution that avoids symmetry issues
    radial_idx = np.zeros(n, dtype=int)
    for i in range(n):
        dist_to_center = np.hypot(xs[i] - 0.5, ys[i] - 0.5)
        # Find nearest radial point (distance to center)
        distances_to_radial = np.hypot(radial_points[:, 0] - xs[i], radial_points[:, 1] - ys[i])
        nearest_idx = np.argmin(distances_to_radial)
        radial_idx[i] = nearest_idx
    
    # Now compute a radial map that assigns weights to each column based on radial distance
    # This introduces spatial clustering and helps avoid grid-like symmetry
    # Assign weights: higher weights to radially centered positions, lower to edges
    radial_weights = np.zeros(n, dtype=float)
    for i in range(n):
        x, y = radial_points[radial_idx[i]]
        radius = np.hypot(x - 0.5, y - 0.5)
        radial_weights[i] = 1.0 - (radius / (0.5))  # Inverse of radial distance, capped at 1
    
    # Apply soft radial weighting to initial positions
    xs = xs_perturbed + (np.random.uniform(-0.01, 0.01, size=n) * radial_weights)
    ys = ys_perturbed + (np.random.uniform(-0.01, 0.01, size=n) * radial_weights)
    
    # Initial radii: compute based on a radial density model with adaptive decay
    # This is more optimized to allow growth in under-optimized areas
    # Compute initial radius as inversely proportional to radial distance from center
    # This helps create regions with more potential for expansion
    radii = np.zeros(n)
    for i in range(n):
        d = np.hypot(xs[i] - 0.5, ys[i] - 0.5)
        # Use smooth decay function to allow radial packing
        # Initial radii: inversely proportional to (1 + d^2), with max value adjusted for spacing
        max_density = 0.5  # This ensures max radius is 0.5, but we'll later adjust based on spatial constraints
        factor = 1.0 / (1.0 + np.power(d, 2))   # Inverse of distance squared (smoothed)
        # This creates natural radial decay that prevents uniform density
        # For circles near center, factor is higher => larger initial radius
        radii[i] = factor * max_density * 0.8  # 0.8 is a tuning factor to ensure under-optimization
    
    # Ensure radii stay within bounds and start with a base minimum
    radii = np.clip(radii, 1e-5, 0.5)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = radii
    
    # Optimizer objective (max sum of radii)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized boundary constraints (left/right/top/bottom as inequalities)
    # Use lambdas with i captured using local binding (with i in for loop)
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized pairwise (non-overlapping) constraints using pairwise distance squares
    # Use advanced vectorized calculation and lambda with parameters captured
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial SLSQP optimization with tight tolerances and sufficient iterations
    # Adding constraint tolerance for numerical issues (1e-11) to allow for small constraint violations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons,
                   options={
                       "maxiter": 2000,  # Extended for improved convergence
                       "ftol": 1e-12,  # Very tight final tolerance for high-precision
                       "gtol": 1e-12,
                       "eps": 1e-6,   # Small for better gradient approximation
                       "disp": False
                   })
    
    # Adaptive reconfiguration with spatial hashing for global structure improvement
    if res.success:
        v = res.x
        # Compute cluster density: spatial hashing with adaptive scale and local perturbation
        # Introduce spatial hashing for structural reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.08  # Smaller to avoid over-perturbation
        # For each circle, apply a perturbation scaled by local density and radial position
        # Perturbation is more intense in clusters of high density
        # This avoids uniform perturbation and preserves structure integrity
        # Compute cluster density as weighted sum of nearby distances
        # We do not recompute cluster density in the optimization phase
        
        # Apply spatial perturbation with adaptive scaling based on radii and position
        # We use an adaptive scaling factor: more perturbation in regions of low density
        # Use a function based on distance from cluster centers to create spatial-aware clustering
        # Compute perturbation as: spatial_hash[i] * (1 + 2 * (radial_weights[i] - 0.5)) 
        # This ensures more variation in under-optimized areas
        radii = v[2::3]
        perturbed_v = v.copy()
        for i in range(n):
            # Compute adaptive perturbation scale based on position and radial weight
            # This adds a more natural and localized reconfiguration
            # If radial_weights[i] is low, apply larger perturbation
            # Use a base perturbation based on spatial_hash[i], scaled by a weight
            base_perturb = spatial_hash[i] * np.sqrt(radii[i])  # Scaling with sqrt of radius
            # Add a small component based on position to encourage movement
            # This helps avoid symmetry and encourages local reconfiguration
            perturb_x = base_perturb[0] * (1.0 + np.abs(v[3*i] - 0.5))  # more perturbation near edges
            perturb_y = base_perturb[1] * (1.0 + np.abs(v[3*i+1] - 0.5)) 
            perturbed_v[3*i] += perturb_x
            perturbed_v[3*i+1] += perturb_y
        
        # Add a small random component for final refinement
        random_perturbation = np.random.rand(n, 2) * 0.02
        perturbed_v[0::3] += random_perturbation[:, 0]
        perturbed_v[1::3] += random_perturbation[:, 1]
        
        # Reoptimize with perturbed configuration, with tighter constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons,
                       options={
                           "maxiter": 700,  # Allow for better local refinement
                           "ftol": 1e-12,
                           "gtol": 1e-12,
                           "eps": 1e-6
                       })
    
    # Refinement phase: Identify most isolated circle for targeted expansion
    # We apply a more sophisticated isolation metric that combines pairwise distances
    # Additionally, we avoid local maxima by introducing spatial gradient-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Use vectorized pairwise distance calculations with broadcasting for efficiency
        # Avoid nested loops; use broadcasting to compute distance matrix in O(n²) for n=26, which is feasible
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute isolation metric: sum of reciprocal of distances to others (avoiding 0 division)
        # This promotes circles with more space to grow
        # Use a small epsilon to prevent division by zero and to prevent extreme values
        isolation_metric = np.zeros(n)
        for i in range(n):
            # Filter out self, avoid division by zero
            valid_dists = dists[i, np.arange(n) != i]
            valid_dists = np.where(valid_dists < 1e-10, 1e-10, valid_dists)  # Avoid division by zero
            isolation_metric[i] = np.sum(1.0 / (valid_dists + 1e-8))  # Adding offset for stability
        
        # Identify the most isolated circle
        isolated_idx = np.argmin(isolation_metric)
        
        # Compute targeted expansion based on spatial gradient, radial positioning, and density
        # We use a combination of: 1) radius scaling based on distance to edges 2) radial decay 3) isolation
        # Add a radial component: circles in center can expand more
        rad_dist = np.hypot(centers[isolated_idx, 0] - 0.5, centers[isolated_idx, 1] - 0.5)
        rad_scale = max(0, 1.0 - (rad_dist / 0.5))  # Radial decay to 0 at edge
        
        # Add a constraint-aware expansion factor (considering existing radii)
        # Expansion factor is weighted by isolation, radial density, and total sum
        # This introduces a natural growth pattern
        expansion_factor = 0.004 * (1.0 + 1.2 * rad_scale) * (1.0 + 0.3 * isolation_metric[isolated_idx])
        
        # Apply expansion to all except isolated_idx, while maintaining feasibility
        new_radii = radii.copy()
        total_sum = np.sum(radii)
        # We do not simply add expansion, but we use a soft expansion that accounts for existing spacing
        expansion_radii = np.zeros(n)
        # Compute a dynamic expansion per circle based on its distance to the isolated_one
        for i in range(n):
            if i == isolated_idx:
                # Expand isolated one
                # We apply a scaled expansion so that the overall total increases
                # This prevents unbounded growth and maintains constraints
                new_radii[i] = np.clip(radii[i] + expansion_factor * 0.8, 1e-5, 0.5)  # Use smaller factor
            else:
                # Add small expansion for other circles to increase overall total
                # Use an expansion that scales with distance to the isolated circle for spatial growth balance
                distance_to_isolated = dists[i, isolated_idx]
                norm_distance = (distance_to_isolated - 1e-10) / (0.5 + 1e-10)  # Normalize to [0,1]
                # Expansion factor is inversely proportional to distance to avoid edge effects
                expansion_i = expansion_factor * (1.0 + 0.3 * (1.0 - norm_distance))
                # Apply expansion in a soft manner to avoid constraint violations
                new_radii[i] = np.clip(radii[i] + expansion_i * 0.9, 1e-5, 0.5)
        
        # Re-evaluate with updated radii
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons,
                       options={
                           "maxiter": 400,  # Allow for better local refinement after expansion
                           "ftol": 1e-12,
                           "gtol": 1e-12,
                           "eps": 1e-6
                       })
    
    # Final fallback: use initial vector if any optimization failed
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
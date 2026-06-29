import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Generate initial positions using a randomized grid with dynamic spacing
    # First create a grid of coordinates based on rows and cols with spatial jitter
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Random perturbation to avoid clustered initial configurations
        x_jitter = np.random.uniform(-0.04, 0.04)
        y_jitter = np.random.uniform(-0.04, 0.04)
        
        # Staggered layout to prevent alignment
        if row % 2 == 1:
            base_x += 0.5 / cols
        
        x = base_x + x_jitter
        y = base_y + y_jitter
        
        # Add with bounds checking for safety
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radial estimate based on grid spacing
    # Calculate spacing between nearby grid points as a baseline
    # We'll allow for some initial variation to improve convergence
    min_grid_spacing = 0.35 / cols
    r0 = min_grid_spacing - 2e-4
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # Left margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints (simplified to avoid numerical issues)
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda to capture i and j
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2)
            })

    # Phase 1: Initial optimization with high tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Phase 2: Apply spatial reconfiguration using randomized geometric tiling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Use randomized tiling to reconfigure spatial relationships
        # Generate seed positions using grid-based tiling pattern
        tile_cols = 4
        tile_rows = 4
        tiling_radius = 0.5 / (tile_cols * tile_rows)
        tiling_offset = np.random.uniform(-0.08, 0.08, (n, 2))
        
        # Reassign circle positions with spatial hashing
        # This triggers a global shift that allows for non-local optimization
        new_v = v.copy()
        for i in range(n):
            new_x = (np.floor(centers[i, 0] / tiling_radius) % tile_cols) / tile_cols
            new_y = (np.floor(centers[i, 1] / tiling_radius) % tile_rows) / tile_rows
            new_x += tiling_offset[i, 0]
            new_y += tiling_offset[i, 1]
            new_v[3*i] = np.clip(new_x, 0.0, 1.0)
            new_v[3*i+1] = np.clip(new_y, 0.0, 1.0)
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    # Phase 3: Targeted radius expansion on globally constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pair-wise distances with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle via minimum-minimum distance metric
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        
        # Calculate dynamic growth target based on spatial occupancy and convergence potential
        current_sum = np.sum(radii)
        target_sum = 2.675  # Slightly aggressive target to explore unexplored density regions
        growth_factor = (target_sum - current_sum) / (n - 1)  # Spread growth over circles
        
        # Create expansion vector with enhanced focus on least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += growth_factor * 1.4  # Aggressive boost
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = growth_factor * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        while True:
            # Validate new radius configuration for overlaps
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
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
                # If overlap, reduce expansion slightly with exponential decay
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    rows = 5
    cols = (n + rows - 1) // rows
    
    # Adaptive seeding with multi-layer geometric hashing and symmetry-busting initialization
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Use adaptive hashing to break regular grid symmetry
        x_offset = (np.random.rand() - 0.5) * 0.08 * (1 + 0.5 * (row % 3 == 0))  # Extra bias for even rows
        y_offset = (np.random.rand() - 0.5) * 0.08 * (1 + 0.5 * (col % 3 == 0))  # Extra bias for even cols
        x = x_center + x_offset
        y = y_center + y_offset
        # Staggered alternating rows with dynamic scaling
        if row % 2 == 1:
            x += (0.5 / cols) * (0.8 if col == 0 else 0.3)  # Less aggressive shift towards edges
        xs.append(x)
        ys.append(y)
    
    # Start with larger initial radii with localized decay
    r0 = 0.38 / cols - 1e-3  # Slightly larger than parent
    # Apply a local radius variation to avoid uniformity
    local_r = r0 * np.ones(n)
    row_idx = np.arange(n) // cols
    # Reduce radii on rows with more circles per row to balance workload
    for r in range(rows):
        row_mask = (row_idx == r)
        local_r[row_mask] = r0 * (1 - 0.2 * (np.count_nonzero(row_mask) / cols))
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = local_r

    # Maintain precise bounds with 3*n entries
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Radii must stay positive, bounded between 0.01 and 0.5

    # Use vectorized objective with gradient approximation
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with capture safety
    cons = []
    for i in range(n):
        def bound_fun(v, i=i):
            # Avoid lambda issues by using nested functions
            x = v[3*i]
            y = v[3*i+1]
            r = v[3*i+2]
            return [x - r, 1.0 - x - r, y - r, 1.0 - y - r]
        # Split into four inequality constraints
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with advanced geometric hashing and spatial partitioning
    # We will use a precomputed spatial grid to reduce constraint evaluation complexity
    max_dist_sq = (1.0 + 1e-6)**2  # Safe upper bound for square distance
    # Create a spatial grid with adaptive resolution based on initial radius estimate
    # We use a hash grid for constraint grouping
    grid_size = 0.15  # Smaller than parent to increase spatial specificity
    spatial_hash = np.floor(np.array(xs) / grid_size).astype(int)
    spatial_hash_y = np.floor(np.array(ys) / grid_size).astype(int)
    
    # Create a sparse adjacency graph to reduce constraint count without losing validity
    # Instead of checking all pairs (n^2), we will check neighboring grid cells only
    grid = {}
    neighbors = []
    for i in range(n):
        x = xs[i]
        y = ys[i]
        hx, hy = spatial_hash[i], spatial_hash_y[i]
        grid_key = (hx, hy)
        if grid_key not in grid:
            grid[grid_key] = []
        grid[grid_key].append(i)
    
    # Precompute neighbor indices for each circle using spatial hashing
    neighbor_indices = [[] for _ in range(n)]
    for key in grid:
        for i in grid[key]:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_key = (key[0] + dx, key[1] + dy)
                    if neighbor_key in grid:
                        for j in grid[neighbor_key]:
                            if j != i:
                                if j not in neighbor_indices[i]:
                                    neighbor_indices[i].append(j)
    # Ensure unique pairs by checking once per pair
    # This will drastically reduce the number of constraints while still validating critical overlaps

    # Add constraints for overlapping circles in spatially grouped regions
    for i in range(n):
        for j in neighbor_indices[i]:
            if j > i:  # Avoid doubling
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist_sq = dx*dx + dy*dy
                    r1 = v[3*i+2]
                    r2 = v[3*j+2]
                    return dist_sq - (r1 + r2)**2
                cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with aggressive settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "gtol": 1e-9})
    
    # Adaptive spatial perturbation for escaping local optima
    if res.success:
        v = res.x
        # Calculate spatial hash for perturbation
        current_hash = np.floor(np.array([v[0::3], v[1::3]]) / grid_size).T.astype(int)
        # For circles in the same grid, apply spatial hashing perturbation
        perturbation_grid = np.random.rand(n, 2) * 0.0
        perturbation_grid[:, 0] = np.random.rand(n) * (0.01 + 0.1 * (np.random.rand(n) > 0.9)) # Small random perturbation
        perturbation_grid[:, 1] = np.random.rand(n) * (0.01 + 0.1 * (np.random.rand(n) > 0.9)) 
        # Ensure perturbation doesn't break initial grid spacing
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation_grid[i, 0]
            perturbed_v[3*i+1] += perturbation_grid[i, 1]
        
        # Re-evaluate with perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-9})
    
    # Targeted growth on minimal constrained circle with soft constraint expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Precompute all pair distances for constraint validation
        # Use numpy broadcasting for vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Calculate min distances for each circle to determine constraint tightness
        min_distances = np.min(distances, axis=1)
        # Compute radius expansion potential based on spacing
        expansion_potential = min_distances - (radii + 1e-6)  # To avoid negative expansion
        expansion_potential[expansion_potential < 0] = 0  # Clamp to non-negative
        expansion_potential /= (1 + np.mean(radii))  # Normalize by a spatial scaling factor
        
        # Identify the circle with the most expansion potential
        largest_growth_idx = np.argmax(expansion_potential)
        max_growth = expansion_potential[largest_growth_idx]
        
        # Apply controlled growth with gradient approximation and constraint validation
        target_growth = 0.008
        expansion_factor = target_growth / max_growth if max_growth > 0 else 0.005
        
        # Create growth vector with localized expansion
        new_radii = radii.copy()
        new_radii[largest_growth_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != largest_growth_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
        
        # Re-validate new configuration with strict constraint check
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Re-check constraint validity using the same distance approach
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
                # If invalid, slightly reduce expansion
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Apply the validated expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-optimization with spatial configuration and radius expansion
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-11, "gtol": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
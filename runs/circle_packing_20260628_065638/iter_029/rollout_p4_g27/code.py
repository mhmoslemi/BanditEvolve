import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Use hexagonal grid with adaptive row staggering for better spacing
    def hexagonal_layout():
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            # Base grid with centering and hexagonal spacing
            base_x = (col + 0.5) / cols
            base_y = (row + 0.5) / rows
            # Apply hexagonal staggering
            if row % 2 == 1:
                # Shift alternate rows to simulate hexagonal packing
                base_x += 0.5 / cols
            # Add a geometric noise distribution to avoid symmetrical clustering
            x_noise = np.random.uniform(-0.03, 0.03) * (1.0 / (cols ** 1.5))
            y_noise = np.random.uniform(-0.03, 0.03) * (1.0 / (rows ** 1.5))
            # Add noise with inverse scaling to larger grids
            noise_scale = 1.0 / (max(cols, rows) * 1.1)
            x_noise *= noise_scale
            y_noise *= noise_scale
            xs.append(base_x + x_noise)
            ys.append(base_y + y_noise)
        return np.array(xs), np.array(ys)

    # Initialize with hexagonal layout
    xs, ys = hexagonal_layout()
    r0 = 0.34 / (cols + 0.5) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n-length bounds to match vector length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Use vectorized and cacheable constraints with lambda with i-j closures for SLSQP
    cons = []
    
    # Boundary constraints
    for i in range(n):
        def boundary_constraint(v, dim, i=i):
            if dim == 0:
                return v[3*i] - v[3*i + 2]
            elif dim == 1:
                return 1.0 - v[3*i] - v[3*i + 2]
            elif dim == 2:
                return v[3*i + 1] - v[3*i + 2]
            elif dim == 3:
                return 1.0 - v[3*i + 1] - v[3*i + 2]
            else:
                raise ValueError(f"Invalid dimension {dim}")
        # Use separate lambdas to avoid closure capture conflicts
        cons.append({"type": "ineq", "fun": lambda v, i=i: boundary_constraint(v, 0, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: boundary_constraint(v, 1, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: boundary_constraint(v, 2, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: boundary_constraint(v, 3, i)})
    
    # Pairwise overlap constraints with optimized computation
    def overlap_constraint(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
    # Use precomputed pairs to minimize lambda closure overhead
    for i in range(n):
        for j in range(i+1, n):
            # Use lambda that captures i and j
            cons.append({"type": "ineq", "fun": (lambda v, i=i, j=j: overlap_constraint(v, i, j))})

    # Use a hybrid optimization strategy with multiple phase refines
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Phase 1: Spatial perturbations with geometric hashing for escape from local minima
    if res.success:
        v = res.x
        
        # Spatial hashing to create diversity in perturbations
        # Use hexagonal grid hashing with adaptive radius scaling
        hash_grid = np.random.rand(n, 2) * 0.1
        hash_grid *= (1.0 + 0.5 * np.abs(radii / np.mean(radii)))
        
        # Apply a directional perturbation for more effective escape
        # Perturb towards hexagonal neighbors with weighted noise
        for i in range(n):
            x_noise = hash_grid[i, 0] * (1.0 / (cols * 2.0))
            y_noise = hash_grid[i, 1] * (1.0 / (rows * 2.0))
            v[3*i] += x_noise
            v[3*i+1] += y_noise
            
            # Apply radius-specific noise to avoid overexpansion
            r_noise = hash_grid[i, 0] * 0.01 * (radii[i] / 0.5)
            v[3*i+2] += r_noise
            
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12})

    # Phase 2: Global expansion with dynamic non-overlap checking
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate pairwise distances matrix efficiently
        c = centers
        d = np.sqrt((c[:, np.newaxis, 0] - c[np.newaxis, :, 0])**2 + (c[:, np.newaxis, 1] - c[np.newaxis, :, 1])**2)
        
        # Identify the circle with the largest available expansion space
        min_dist_to_neighbors = np.min(d, axis=1)
        expansion_idx = np.argmax(min_dist_to_neighbors)
        
        # Calculate the theoretical max expansion
        max_growth = (min_dist_to_neighbors[expansion_idx] - radii[expansion_idx]) / 2
        target_sum_growth = max(0.001, np.min([0.006, max_growth * 3.0]))
        
        # Calculate optimal radius expansion to prevent overexpansion
        total_radii = np.sum(radii)
        target_new_sum = total_radii + target_sum_growth
        expansion_per_circle = (target_new_sum - total_radii) / n
        
        # Add expansion to all circles, except the expansion_idx
        expansion_radii = radii.copy()
        expansion_radii[expansion_idx] += max_growth * 1.1  # slight overexpansion to find optimal
        for i in range(n):
            if i != expansion_idx:
                expansion_radii[i] += expansion_per_circle * 0.9
        
        # Create new decision vector
        new_v = v.copy()
        new_v[2::3] = expansion_radii
        new_centers = np.column_stack([new_v[0::3], new_v[1::3]])
        
        # Check if this new configuration is valid
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((new_centers[i, 0] - new_centers[j, 0])**2 + (new_centers[i, 1] - new_centers[j, 1])**2)
                if dist < expansion_radii[i] + expansion_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
        else:
            # If invalid, perform a gradual expansion to find the maximal valid configuration
            current_radii = radii.copy()
            while True:
                expansion = (target_sum_growth / 3.0) * 0.85
                new_radii = current_radii.copy()
                new_radii[expansion_idx] += expansion * 1.2
                for i in range(n):
                    if i != expansion_idx:
                        new_radii[i] += expansion * 0.9
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        dist = np.sqrt((new_centers[i, 0] - new_centers[j, 0])**2 + (new_centers[i, 1] - new_centers[j, 1])**2)
                        if dist < new_radii[i] + new_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    current_radii = new_radii
                else:
                    break
            new_v = v.copy()
            new_v[2::3] = current_radii
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
    
    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
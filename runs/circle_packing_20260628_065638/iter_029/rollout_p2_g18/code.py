import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Base grid with advanced geometric hashing and spatial constraint balancing
    grid_centers_x = (np.arange(cols) + 0.5) / cols
    grid_centers_y = (np.arange(rows) + 0.5) / rows
    xs_initial = []
    ys_initial = []
    radius_base = 0.34 / cols + 1e-4  # Slight increase for expansion readiness
    
    # Initial placement with enhanced spatial hashing and row staggering + density-aware offsets
    for i in range(n):
        col = i % cols
        row = i // cols
        # Base position with grid offset for uniform spreading
        base_x = grid_centers_x[col]
        base_y = grid_centers_y[row]
        
        # Spatial hashing for perturbation: random vector scaled by radius-aware scaling
        hash_offset = np.random.rand(2, 1) * 0.06
        x_perturb = hash_offset[0, 0] * (radius_base / cols) * 1.1
        y_perturb = hash_offset[1, 0] * (radius_base / cols) * 1.1
        
        x = base_x + x_perturb
        y = base_y + y_perturb
        
        # Implement dynamic row staggering
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 - (row % cols) / rows)
        else:
            x -= 0.5 / cols * (1.0 - (row % cols) / rows)
        
        # Clip coordinates to unit box with tolerance for floating point drift
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs_initial.append(x)
        ys_initial.append(y)
    
    # Initial radius configuration with density-aware distribution and growth-ready baseline
    r0 = radius_base * np.ones(n) - 1e-4  # Avoid zero-radius edge cases
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs_initial)
    v0[1::3] = np.array(ys_initial)
    v0[2::3] = r0
    
    # Ensure bounds are precisely 3*n entries; this is critical for SLSQP
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-4, 0.5))  # Safety boundary for radius
    
    # Define cost function: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint definitions using function closures with captured i
    cons = []
    
    # Boundary constraints: 4 per circle (left, right, bottom, top)
    for i in range(n):
        # Left: x >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom: y >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints: pairwise separation between circles
    for i in range(n):
        for j in range(i + 1, n):
            # Constraint: distance^2 - (r_i + r_j)^2 >= 0
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                              (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                              - (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization with tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "gtol": 1e-10})
    
    # First-level reconfiguration with spatial reordering for dynamic constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial perturbations with adaptive density-aware offsets
        # Introduce asymmetric spatial hashing with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        
        # Apply asymmetric displacement based on local density + density-aware scaling
        for i in range(n):
            dx_perturb = spatial_hash[i, 0] * (radii[i] / np.mean(radii) * 1.2)
            dy_perturb = spatial_hash[i, 1] * (radii[i] / np.mean(radii) * 1.2)
            perturbed_v[3*i] += dx_perturb
            perturbed_v[3*i+1] += dy_perturb
        
        # Refine with adjusted perturbation
        perturbed_v = np.clip(perturbed_v, 0.0, 1.0)
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # If success, proceed with dynamic reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distance matrix with vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle: max min distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor based on current total and adaptive density
        current_total = np.sum(radii)
        target_growth = 0.0085  # Slightly higher for aggressive expansion
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii)) * 1.05
        
        # Generate expansion vector with targeted growth
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        
        # Apply stochastic expansion with adaptive factor
        for i in range(n):
            if i != least_constrained_idx:
                # Adaptive stochastic expansion scaling
                # Introduce randomness with density-aware amplification
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())
                expansion_i *= np.log1p(radii[i] / np.mean(radii))  # Density-aware scaling
                new_radii[i] += expansion_i
        
        # Apply expansion with local constraint validation, max 5 iteration retries
        for _ in range(5):
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_ = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_ = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_**2 + dy_**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # Reduce expansion factor by 5% if invalid
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final refine with expanded configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Final validation and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Perform final validation check to prevent invalid solutions
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < radii[i] + radii[j] - 1e-12:
                # Adjust radii if needed
                radii[i] = np.clip(radii[i] - (radii[i] + radii[j] - dist), 1e-6, None)
                radii[j] = np.clip(radii[j] - (radii[i] + radii[j] - dist), 1e-6, None)
    
    return centers, radii, float(radii.sum())
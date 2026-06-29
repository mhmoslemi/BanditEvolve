import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Enhanced initialization with dynamic spacing, adaptive offset, and asymmetric distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Adaptive offset based on row/column spacing
        offset_ratio = np.sqrt(row**2 + col**2) / np.sqrt(n)
        x = x_center + np.random.uniform(-0.1 * offset_ratio, 0.1 * offset_ratio)
        y = y_center + np.random.uniform(-0.1 * offset_ratio, 0.1 * offset_ratio)
        # Staggered grid with asymmetric row displacement
        if row % 2 == 1:
            x += 0.3 * offset_ratio / (cols * np.sqrt(2))
            y += 0.3 * offset_ratio / (rows * np.sqrt(2))
        # Apply nonlinear jitter to enhance spreading
        x += np.random.uniform(-0.02, 0.02) * (1 + (row % 3 == 0))
        y += np.random.uniform(-0.02, 0.02) * (1 + (col % 3 == 0))
        xs.append(x)
        ys.append(y)
    
    # Calculate initial radius based on grid spacing and adaptive scaling
    r0 = 0.35 / cols - 1e-3
    r0 *= 1.3 * (1 + np.sqrt((n - cols*rows)/cols)) 
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure the bounds list has 3*n entries to match the vector length
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with lambda binding
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with dynamic scaling
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2 * (1 + 0.01*(i+j)) })

    # Initial optimization with strict tolerances and hybrid strategies
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-12})

    # Spatial constraint reconfiguration with adaptive perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create spatial hash with dynamic weight based on radius variation
        spatial_hash = np.random.rand(n, 2) * 0.07
        perturbed_v = v.copy()
        for i in range(n):
            # Adaptive perturbation based on radius size and local spacing
            scale = np.sqrt((radii[i]/np.mean(radii))**2 + (1/(np.sqrt((centers[i,0]+centers[i,1])**2 + 1))))
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale
            
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12})

    # Targeted radius expansion with advanced prioritization and geometric reorganization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting with adaptive scaling
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find spatially least constrained circle in non-Euclidean space
        min_dists = np.min(dists, axis=1)
        weighted_dists = min_dists * (1 / (1 + np.sqrt(n)) + radii / np.mean(radii))
        least_constrained_idx = np.argmax(weighted_dists)  # Weighted selection
        
        # Calculate optimized expansion based on spatial and numerical dynamics
        current_total = np.sum(radii)
        target_growth = 0.009 # Increase from parent's 0.007
        expansion_factor_base = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Expand with directional bias and spatial awareness
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor_base * 1.2 # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Use angular distance to determine expansion priority
                angle_diff = np.arctan2(centers[i, 1] - centers[least_constrained_idx, 1], 
                                       centers[i, 0] - centers[least_constrained_idx, 0])
                direction = np.random.rand() * (1 + 0.2 * np.exp(-0.5 * (angle_diff)**2))
                expansion_i = expansion_factor_base * (1.0 + 0.08 * np.cos(angle_diff) * direction)
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation and gradient-aware adjustments
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration with optimized checking
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
                # If invalid, decrease expansion slightly with gradient-aware decay
                new_radii = radii + (new_radii - radii) * 0.95

        # Update decision vector and re-evaluate with refined configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
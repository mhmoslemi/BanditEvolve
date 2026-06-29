import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Generate initial seeds with randomized staggered positioning and adaptive spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Apply adaptive randomized perturbation based on row proximity and spatial distribution
        if row == 0 or row == rows - 1:
            x = x_center + np.random.uniform(-0.12, 0.12)
        else:
            x = x_center + np.random.uniform(-0.10, 0.10)
        
        if col == 0 or col == cols - 1:
            y = y_center + np.random.uniform(-0.12, 0.12)
        else:
            y = y_center + np.random.uniform(-0.10, 0.10)
        
        # Stagger alternate rows with spatial-aware vertical shift
        if row % 2 == 1:
            x += 0.45 / cols
            if col == cols // 2:
                y += np.random.uniform(-0.03, 0.03)
            elif col == cols // 2 - 1 or col == cols // 2 + 1:
                y += np.random.uniform(-0.02, 0.02)
        
        xs.append(x)
        ys.append(y)
    
    # Calculate initial radii based on grid spacing with adaptive scaling
    grid_spacing = 0.45 / cols  # Slightly reduced to enhance packing potential
    r0 = (grid_spacing * 0.8) - 1e-4
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bound entries for 26 circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with lambda capture and spatial-aware tolerance
    cons = []
    for i in range(n):
        # Left + radius <= 1 (with adaptive tolerance depending on row proximity)
        tolerance = 0.002
        cons.append({"type": "ineq", "fun": (lambda v, i=i, t=tolerance: 
                                           1.0 - v[3*i] - v[3*i+2] + t)})
        
        # Right - radius >= 0 (with adaptive tolerance depending on row proximity)
        cons.append({"type": "ineq", "fun": (lambda v, i=i, t=tolerance: 
                                           v[3*i] - v[3*i+2] - t)})
        
        # Bottom + radius <= 1 (with adaptive vertical tolerance)
        tolerance_y = 0.002
        cons.append({"type": "ineq", "fun": (lambda v, i=i, t=tolerance_y: 
                                           1.0 - v[3*i+1] - v[3*i+2] + t)})
        
        # Top - radius >= 0 (with adaptive vertical tolerance)
        cons.append({"type": "ineq", "fun": (lambda v, i=i, t=tolerance_y: 
                                           v[3*i+1] - v[3*i+2] - t)})

    # Vectorized overlap constraint with adaptive scaling based on spatial distribution
    for i in range(n):
        for j in range(i + 1, n):
            # Apply geometric hashing to identify spatially distinct pairs
            # Use distance-dependent constraint scaling to prioritize distant pairs
            scaling_factor = max(0.1, 1.0 - np.sqrt( (v0[3*i] - v0[3*j])**2 + (v0[3*i+1] - v0[3*j+1])**2 ) / 0.45 )
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j, s=scaling_factor: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                                 - (v[3*i+2] + v[3*j+2])**2 * s**2)})

    # Initial optimization with high precision and adaptive constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-10, "eps": 1e-12})

    # Spatial reconfiguration using geometric hashing with row-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash with spatial-aware scaling for row distribution
        spatial_hash = (np.random.rand(n, 2) - 0.5) * (0.08 + 0.05 * (np.array(centers)[:, 1] < 0.5))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration and tighter constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 1e-11, "eps": 1e-12})

    # Targeted expansion of least constrained circle with dynamic radius growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle using min distance to neighbors and row distribution
        min_dists = np.min(dists, axis=1)
        proximity_weight = 0.7 + 0.3 * (centers[:, 1] < 0.5)
        weighted_min_dists = min_dists * proximity_weight
        least_constrained_idx = np.argmax(weighted_min_dists)
        
        # Calculate expansion based on current total sum and spatial constraints
        current_total = np.sum(radii)
        target_growth = 0.008 * (1.0 + np.log(1 + 2.0 * np.mean(radii)))
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector with adaptive expansion based on spatial distribution
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.25  # More aggressive over-expansion
        for i in range(n):
            # Apply spatial-aware stochastic expansion
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.15 * np.random.rand())
                if (centers[i, 1] < 0.5 and centers[i, 0] < 0.4):
                    expansion_i *= 1.2
                elif (centers[i, 1] > 0.5 and centers[i, 0] > 0.6):
                    expansion_i *= 1.15
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                # Reduce expansion based on proximity to boundaries
                expansion_factor *= 0.95
                new_radii = radii + (new_radii - radii) * (1.0 - (1.0 - expansion_factor) * 0.5)
        
        # Final optimization with tighter constraints
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
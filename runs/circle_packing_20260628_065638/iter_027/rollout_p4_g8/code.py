import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Step 1: Initialize with randomized geometric tiling and perturbed grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Apply perturbation with dynamic scale and random spatial bias
        x = base_x + np.random.uniform(-0.04 * (row + 1), 0.04 * (row + 1))
        y = base_y + np.random.uniform(-0.04 * (row + 1), 0.04 * (row + 1))
        # Alternate row shift with adaptive magnitude to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 / (row + 1)) + np.random.uniform(-0.003, 0.003)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Same length as decision variable

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective is to maximize sum of radii

    # Step 2: Vectorized boundary constraints with lambda closures and proper closures
    cons = []
    for i in range(n):
        # Left boundary: x - r <= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r <= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Step 3: Vectorized circle-to-circle constraints using lambda with captured i and j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })

    # Step 4: Initial optimization with tighter tolerances and increased iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})

    # Step 5: Spatial reconfiguration with advanced geometric hashing algorithm
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate advanced geometric hash with spatial awareness and adaptive perturbation
        # Using randomized perturbation based on distance to nearest neighbors and current radius
        # This creates a more organic layout that better accommodates larger radii
        
        # Compute nearest neighbors for each point
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        
        # Create random hash with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Scale perturbation by radius to allow for more expansion for smaller circles
            scale = np.clip((radii[i] / np.mean(radii)) ** 0.5, 0.5, 1.5)
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-11})

    # Step 6: Targeted radius expansion using dynamic constraint-based expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized pairwise distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (least minimum distance to any other circle)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        least_constrained_radius = radii[least_constrained_idx]
        
        # Calculate growth based on current total sum and spatial constraints
        current_total = np.sum(radii)
        target_growth = 0.012
        target_ratio = (target_growth / n) / (1 - current_total / 5.0)  # Growth based on remaining space
        
        # Calculate base radius expansion based on current configuration
        base_radius = 0.05  # Safe baseline for expansion
        scale_factor = (least_constrained_radius / np.min(radii)) ** 0.7  # Adaptive scaling for small radii
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += target_ratio * scale_factor * 2.0  # Stronger expansion
        
        # Apply stochastic expansion to nearby circles with diminishing return
        # This creates a cascading effect where more circles can expand
        for i in range(n):
            if i != least_constrained_idx:
                # Compute proximity to the least constrained circle
                proximity = 1.0 / (1.0 + np.sqrt((centers[i, 0] - centers[least_constrained_idx, 0])**2 + 
                                                  (centers[i, 1] - centers[least_constrained_idx, 1])**2)) ** 0.5
                
                # Compute adaptive expansion amount based on proximity and current state
                expansion = target_ratio * (1.0 + 0.3 * np.random.rand()) * proximity * (1.0 / (1.0 + min_dists[i]))
                new_radii[i] += expansion
        
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
                # If invalid, decrease expansion slightly with adaptive rate
                decay_rate = 0.99 if np.min(new_radii) < 0.001 else 0.98  # More severe decay for near-zero radii
                new_radii = radii + (new_radii - radii) * decay_rate
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-11})

    # Step 7: Final validation and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
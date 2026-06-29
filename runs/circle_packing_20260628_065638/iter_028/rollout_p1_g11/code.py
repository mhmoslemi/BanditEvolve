import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Initial grid points with spacing for better initial packing
        x_center = (col + 0.1) / cols
        y_center = (row + 0.1) / rows
        # Add small randomized offset for diversity and spread
        offset_x = np.random.uniform(-0.02, 0.02) if row % 2 == 0 else np.random.uniform(-0.02, 0.02)
        offset_y = np.random.uniform(-0.02, 0.02) if row % 2 == 0 else np.random.uniform(-0.02, 0.02)
        x = x_center + offset_x
        y = y_center + offset_y
        # Apply staggered grid shift for alternate rows (improves spacing)
        if row % 2 == 1:
            x += 0.25 / cols  # increased shift for better vertical spacing
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with a more aggressive initial value based on grid spacing
    initial_grid_gap = 0.5 / cols
    # We expect an average radius around 0.08 for 25 circles in a 5x5 grid
    # Initial larger radius as we are going to do more aggressive expansion
    r0 = 0.085  # increased from the parent's 0.375 / cols - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure the bounds list has 3*n entries for the vector of length 3n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11})

    # First reconfiguration - spatial grid hashing with dynamic expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate advanced spatial hashing with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Use a scaled factor based on the radius distribution
            scale = 0.5 + 0.5 * (radii[i] / np.mean(radii))
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    # Second reconfiguration - targeted radial expansion using global configuration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix in vectorized form
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the circle with the smallest minimum distance to others (most crowded)
        min_dists = np.min(dists, axis=1)
        # Use the one with the smallest minimum distance for expansion
        # (the most constrained circle) to enable radical growth
        least_constrained_idx = np.argmin(min_dists)
        
        # We implement directed radial expansion based on spatial hashing and adjacency vectors
        # Create directional spatial hashing for enhanced radial expansion
        directional_hash = np.random.rand(n, 2) * 0.03
        # Create adjacency weights based on proximity to least constrained
        adj_weights = np.linalg.norm(centers - centers[least_constrained_idx], axis=1)
        
        # Calculate total current sum and propose a target growth strategy
        current_total = np.sum(radii)
        # We attempt to increase total by at least 0.015 (over 2.634) to reach 2.65
        target_growth = 0.015  # increased from parent's 0.007
        expansion_factor = target_growth / (n)  # distribute growth across all with bias
        # Add a small directional bump to the least constrained circle
        expansion_factor += 0.008  # add direct boost
        
        # Create expanded radii with spatial and adjacency weighting
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.35  # significant over-expansion
        for i in range(n):
            # Adjust expansion based on adjacency distance and directional hashing
            if i != least_constrained_idx:
                # Use inverse of distance as a weight to boost nearby
                adj_factor = 1 / (adj_weights[i] / 0.1 + 1.0)
                expansion_i = expansion_factor * (1.0 + 0.5 * directional_hash[i, 0])
                # Apply adjacency weight and directional influence
                new_radii[i] += expansion_i * adj_factor * (1.0 / (1 + adj_weights[i]))  # inverse scaling

        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    # Final iteration with fine adjustment
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
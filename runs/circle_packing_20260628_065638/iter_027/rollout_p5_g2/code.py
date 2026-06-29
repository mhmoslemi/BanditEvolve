import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with geometric clustering and directional bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Introduce adaptive spatial distortion with directional bias
        x = x_center + (np.random.rand() - 0.5) * 0.1
        y = y_center + (np.random.rand() - 0.5) * 0.1
        
        # Add directional bias for alternate rows (staggered grid)
        if row % 2 == 1:
            x += 0.5 / cols * (1 + np.random.rand() * 0.2)
        xs.append(x)
        ys.append(y)
    
    # Initial radius distribution with improved spacing
    r0 = 0.35 / cols - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with capture
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

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Apply geometric perturbation using adaptive spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Adaptive spatial hash: weighted by radial spacing and position
        spatial_hash = np.random.rand(n, 2) * 0.06 * np.sqrt(radii)
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * max(radii[i] * 0.7, 0.0001)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * max(radii[i] * 0.7, 0.0001)
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion using adaptive reordering algorithm
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify minimum spacing circles for topological reordering
        total_dists = np.sum(dists, axis=1)
        least_constrained_idx = np.argsort(total_dists)
        
        # Generate optimal reordering to maximize expansion potential
        # Use bubble sort-inspired reordering to maximize spacing
        ordered_indices = np.argsort(total_dists)
        reordered_centers = centers[ordered_indices]
        reordered_radii = radii[ordered_indices]
        
        # Create expansion vector based on reordering and spatial proximity
        new_radii = np.zeros(n)
        new_radii[:] = reordered_radii[:]
        
        # Calculate expansion based on spatial spacing and radial constraints
        expansion = 0.006 / (np.sum(reordered_radii))
        for i in range(n):
            if i == 0:
                new_radii[i] += expansion * 1.2 * np.sqrt(reordered_radii[i])
            else:
                # Spatial expansion based on distance from previous circle
                dist_from_prev = np.sqrt((reordered_centers[i, 0] - reordered_centers[i-1, 0])**2 +
                                         (reordered_centers[i, 1] - reordered_centers[i-1, 1])**2)
                if dist_from_prev < 2 * (reordered_radii[i-1] + reordered_radii[i]):
                    expansion_rate = expansion * 1.5 * (1 + np.random.rand() * 0.2)
                else:
                    expansion_rate = expansion * 0.8 * (1 + np.random.rand() * 0.2)
                new_radii[i] += expansion_rate * np.sqrt(reordered_radii[i])
        
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
                # If invalid, decrease expansion slightly with adaptive decay
                new_radii = radii + (new_radii - radii) * 0.95

        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with updated constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
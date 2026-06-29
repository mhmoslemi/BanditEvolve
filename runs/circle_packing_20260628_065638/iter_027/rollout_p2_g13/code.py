import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with highly randomized hexagonal grid with adaptive spacing
    xs = []
    ys = []
    # First pass: cluster in hexagonal grid with perturbed lattice
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce spatial distortion to break symmetry
        # Apply hexagonal coordinate perturbation
        offset_x = np.random.uniform(-0.04, 0.04)
        offset_y = np.random.uniform(-0.04, 0.04)
        # Alternate row shift for staggered grid
        if row % 2 == 1:
            offset_x += 0.5 / cols  # shift right in odd rows
        x = x_center + offset_x
        y = y_center + offset_y
        xs.append(x)
        ys.append(y)
    
    # Refine initial positions using Voronoi tessellation for optimal dispersion
    initial_centers = np.column_stack([xs, ys])
    # Compute Voronoi regions but only use the center of each region
    # Approximate by shifting points outwards while maintaining grid structure
    vor_x = initial_centers[:, 0] + 0.05 * (np.random.rand(n) - 0.5)
    vor_y = initial_centers[:, 1] + 0.05 * (np.random.rand(n) - 0.5)
    # Enforce grid parity constraints
    for i in range(n):
        vor_x[i] = np.clip(vor_x[i], 0.0, 1.0)
        vor_y[i] = np.clip(vor_y[i], 0.0, 1.0)
    
    # Finalize initial configuration
    xs = vor_x.tolist()
    ys = vor_y.tolist()
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

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
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Dynamic spatial reconfiguration with adaptive hashing and perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix for all pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Define spatial hashing with adaptive weights
        spatial_hash = np.random.rand(n, 2) * 0.06
        # Compute spatial energy for reconfiguration
        spatial_energy = np.sum((dists - radii[:, None] - radii[None, :]) ** 2, axis=1)
        least_constrained_idx = np.argmin(spatial_energy)
        
        # Generate perturbation with adaptive scaling to least constrained circle
        perturbation = spatial_hash * radii[least_constrained_idx] * 0.9
        perturbed_v = v.copy()
        
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        
        # Re-evaluate with reconfigured positions
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion with global spatial awareness
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix (vectorized with broadcasting)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute constraint-based spatial metrics
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)  # Least constrained circle
        
        # Calculate global expansion potential
        current_total = np.sum(radii)
        # Target a dynamic expansion based on spatial energy and current capacity
        expansion_factor = 0.0125 * (current_total / (1.0 - current_total))
        # Asymmetric radius expansion to least constrained circle
        new_radii = radii.copy()
        # Allow for localized expansion of the least constrained circle
        new_radii[least_constrained_idx] += expansion_factor * 1.4  # Slight over-expansion
        # Distribute expansion proportionally across others
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * 0.8  # Conservative expansion
                
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
                # If invalid, decrease expansion slightly with exponential decay
                new_radii = radii + (new_radii - radii) * 0.95

        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final optimization pass with dynamic reconfiguration and tighter constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix once again
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Generate secondary spatial constraint metric for reconfiguration
        spatial_energy_new = np.sum((dists - radii[:, None] - radii[None, :]) ** 2, axis=1)
        least_constrained_idx_new = np.argmin(spatial_energy_new)

        # Reapply spatial perturbation based on new spatial layout
        spatial_hash_new = np.random.rand(n, 2) * 0.04
        perturbation_new = spatial_hash_new * radii[least_constrained_idx_new] * 0.7
        perturbed_v_new = v.copy()
        
        for i in range(n):
            perturbed_v_new[3*i] += perturbation_new[i, 0]
            perturbed_v_new[3*i+1] += perturbation_new[i, 1]
        
        # Final evaluation step
        res = minimize(neg_sum_radii, perturbed_v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
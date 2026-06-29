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
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.32 / cols - 1e-3
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
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10})
    
    # Radical geometric reconfiguration via randomized geometric hashing with enhanced spatial mixing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Enhance spatial mixing via geometric hashing with adaptive randomization
        spatial_hash = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        for i in range(n):
            # Scale perturbation based on both radius and position spacing
            radius_factor = radii[i] / np.mean(radii)
            spacing_factor = np.std(centers) * 0.5
            perturb_x = spatial_hash[i, 0] * radius_factor * spacing_factor
            perturb_y = spatial_hash[i, 1] * radius_factor * spacing_factor
            perturbed_v[3*i] += perturb_x
            perturbed_v[3*i+1] += perturb_y
        
        # Re-evaluate with new spatial configuration with higher solver robustness
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    # Topological reordering with advanced radius expansion via multi-level constraint refinement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the circle with smallest non-zero radius for expansion
        smallest_radius_idx = np.argmin(np.abs(radii))
        smallest_radius = radii[smallest_radius_idx]
        min_dists = np.min(dists, axis=1)
        most_isolated_idx = np.argmax(min_dists)
        
        # Calculate expansion factor using adaptive heuristic with geometric constraints
        current_total = np.sum(radii)
        target_growth = 0.007
        expansion_factor_base = target_growth / (n - 1) * (current_total / np.sum(radii))
        expansion_factor = expansion_factor_base * np.sqrt(np.sum((centers - np.mean(centers, axis=0))**2)/np.std(centers)**2)
        
        # Create expansion vector with targeted expansion on most isolated and smallest radius circle
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * 1.15
        new_radii[most_isolated_idx] += expansion_factor * 1.2
        
        # Apply stochastic expansion to other circles with adaptive scaling
        for i in range(n):
            if i not in [smallest_radius_idx, most_isolated_idx]:
                expansion_i = expansion_factor * (1.15 + 0.05 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Validate and refine expanded radii with local refinement
        iterations = 0
        max_iterations = 3
        while iterations < max_iterations:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between circles
            valid = True
            min_overlap_dist = 1e-9
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Apply further refinement using gradient-based spatial scaling
                dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
                dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
                dists = np.sqrt(dx**2 + dy**2)
                
                # Spatial clustering refinement
                cluster_indices = np.argmin(np.min(dists, axis=1))
                cluster_radius = new_radii[cluster_indices]
                cluster_center = centers[cluster_indices]
                
                # Adjust radii using inverse radius spacing and local density
                density = np.mean(1 / (np.sqrt((centers - cluster_center)**2).sum(axis=1)))
                adjustment_factor = float(np.clip(np.sqrt(density) * 0.002, 0, 0.03))
                new_radii[cluster_indices] += adjustment_factor
                break  # After successful refinement without overlap
            else:
                # If overlap detected, reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
        
        # Update decision vector with refined expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with final configuration using enhanced convergence settings
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
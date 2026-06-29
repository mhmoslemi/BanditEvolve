import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    initial_cell_size = 1.0 / cols
    
    # Initialize with advanced geometric hashing + adaptive perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x_center = (col + 0.5) * initial_cell_size
        base_y_center = (row + 0.5) * initial_cell_size
        # Introduce hierarchical geometric hashing for better configuration diversity
        # First layer: base grid shift
        base_offset_x = (row * col) % cols / (cols + 1)
        base_offset_y = (row * (cols - col)) % rows / (rows + 1)
        
        # Second layer: spatial symmetry-breaking via random seed-based rotation
        base_phi = np.random.uniform(0, 2 * np.pi)
        x_offset = np.sin(base_phi) * (1 / cols)
        y_offset = np.cos(base_phi) * (1 / cols)
        
        x = base_x_center + x_offset + np.random.uniform(-0.02, 0.02)
        y = base_y_center + y_offset + np.random.uniform(-0.02, 0.02)
        
        # Alternate row shift for staggered grid
        if row % 2 == 1:
            x += initial_cell_size / (cols + 1)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with spatial density compensation
    r0 = (1.0 / cols / np.sqrt(2)) * (n / (cols * rows))**(-0.35) - 1e-3
    r0 = np.clip(r0, 1e-4, 0.5)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Enforce consistent bounds structure
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Define objective with careful gradient handling
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Build constraints with closure capturing
    cons = []
    
    # Bound constraints for all circles
    for i in range(n):
        # Left bound: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right bound: 1 - (x + r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom bound: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top bound: 1 - (y + r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints with optimized structure and closure capturing
    for i in range(n):
        for j in range(i + 1, n):
            # Use a lambda that captures i and j correctly
            # Closure capture with parameter binding
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # First optimization phase with tight constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-12})
    
    # Strategic reconfiguration: identify the most constrained spatially circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial interaction metric for circles
        # Compute all pairwise distances and min distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists_matrix = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distance from each circle to others
        min_dist_per_circle = np.min(dists_matrix, axis=1)
        
        # Find the circle with the smallest minimum distance (most spatially constrained)
        most_constrained_idx = np.argmin(min_dist_per_circle)
        most_constrained_r = radii[most_constrained_idx]
        # Apply strong perturbation to release it
        perturbation_factor = 1.2 * (most_constrained_r / np.mean(radii))
        perturbation = np.random.rand(n, 2) * (0.06 + 0.4 * np.random.rand()) * perturbation_factor
        
        # Create perturbed configuration
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        
        # Second phase optimization with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-12})
    
    # Final stage: global radius expansion while maintaining strict constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute current radii sum and target expansion
        current_sum = np.sum(radii)
        target_growth = 0.011  # Slight increase for potential gain
        
        # Compute optimal expansion distribution
        # Use a combination of uniform and proportional expansion to maintain spatial balance
        expansion_ratio = max(1.0, (target_growth / (n - 1)) * (current_sum / 1.0))
        new_radii = radii.copy()
        
        # Compute new radii with expansion on the most constrained circle
        # First scale base radii
        base_radius_increase = expansion_ratio * (np.mean(radii) - radii)
        new_radii += base_radius_increase
        
        # Apply stronger expansion to the most constrained circle
        expansion = expansion_ratio * 1.15 * (radii[most_constrained_idx] - 1e-4)
        new_radii[most_constrained_idx] += expansion
        
        # Apply spatial constraint check for expansion feasibility
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(new_radii, 1e-6, 0.5)
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check for overlaps with tolerance enforcement
            valid = True
            temp_dists = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (expanded_v[3*i+2] + expanded_v[3*j+2]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, reduce expansion by 10% and restart
                new_radii = (new_radii - radii) * 0.9 + radii
                # Ensure it doesn't fall below min
                new_radii = np.clip(new_radii, 1e-6, 0.5)
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final fine-tuning optimization
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
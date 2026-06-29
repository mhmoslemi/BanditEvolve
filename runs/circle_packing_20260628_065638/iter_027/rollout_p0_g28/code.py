import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized, adaptive, and non-uniform spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Adaptive randomness: scale randomness on row and col proximity to edges
        x_offset = np.random.uniform(-0.07, 0.07) * (1 - (np.abs(col - cols + 1) / cols))
        y_offset = np.random.uniform(-0.07, 0.07) * (1 - (np.abs(row - rows + 1) / rows))
        x_shift = (0.5 / cols) * (1 if (row % 2 == 1) else 0)
        
        # Apply staggered row shift
        x = base_x + x_offset + x_shift
        y = base_y + y_offset
        
        # Ensure bounds are respected with safety checks
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        xs.append(x)
        ys.append(y)
    
    # Adaptive initial radii: higher radii near center, lower near edges
    r0_base = np.random.uniform(0.32, 0.38)
    r0 = np.zeros(n)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        dist_from_center_x = np.abs(col - cols/2) / cols
        dist_from_center_y = np.abs(row - rows/2) / rows
        dist_from_center = np.sqrt(dist_from_center_x**2 + dist_from_center_y**2)
        
        # Scale radius inversely proportional to distance from center
        r0[i] = r0_base * (1.0 - 1.5 * dist_from_center)
    
    r0 = np.clip(r0, 1e-4, 0.45)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    # Ensure the bounds list has 3*n entries for the vector of length 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i and optimized capture
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using lambda with captured i,j but with optimized capture
    for i in range(n):
        for j in range(i + 1, n):
            # Use function factory to capture i and j in the lambda
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # First optimization pass with moderate constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})

    # Radical geometric reconfiguration via spatial hashing and controlled expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash with adaptive scaling for enhanced randomization
        # Spatial hash uses a combination of Voronoi tiling and radial pattern matching
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        
        # For each circle, apply a controlled spatial perturbation 
        # based on proximity to other circles and current radius
        for i in range(n):
            base_scale = 0.3 * (radii[i] / np.mean(radii))
            x_perturbation = np.clip(spatial_hash[i, 0] * base_scale, -0.03, 0.03)
            y_perturbation = np.clip(spatial_hash[i, 1] * base_scale, -0.03, 0.03)
            perturbed_v[3*i] += x_perturbation
            perturbed_v[3*i+1] += y_perturbation
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Topological reordering with constrained radius expansion and gradient-based refinement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate adaptive constraints based on spatial distribution
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find smallest circle by evaluating proximity to all others
        # with weighted spatial constraint function
        spatial_weights = np.exp(-dists / (np.mean(radii) + 1e-6))
        weighted_radii = spatial_weights * radii
        smallest_radius_idx = np.argmin(weighted_radii)
        smallest_radius = weighted_radii[smallest_radius_idx]
        
        # Calculate expansion factor using adaptive heuristic based on total system pressure
        current_total = np.sum(radii)
        target_growth = 0.008  # Increase expansion target
        expansion_factor = target_growth / (n) * (current_total / np.sum(radii)) * 1.4
        
        # Create new radii with controlled expansion on least constrained circle
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != smallest_radius_idx:
                # Apply stochastic expansion with adaptive scaling based on spatial density
                expansion_i = expansion_factor * (1 + 0.1 * np.random.rand()) * np.exp(-dists[i, smallest_radius_idx])
                new_radii[i] += expansion_i
        
        # Validate and refine expanded radii with local gradient refinement
        # Use a hybrid approach: first validate, then perform refinement step
        iterations = 0
        max_iterations = 2
        while iterations < max_iterations:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between circles with tolerance
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
                # If overlap detected, reduce expansion slightly
                # Use inverse proportional scaling based on density near the constrained circle
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
        
        # Update decision vector with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with new configuration but use a gradient-based method for fine-tuning
        res = minimize(neg_sum_radii, v_new, method="BFGS", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
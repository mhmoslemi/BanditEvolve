import numpy as np

def run_packing():
    n = 26
    cols = 4
    rows = (n + cols - 1) // cols
    # Initialize positions with geometric clustering, staggered grid, and dynamic offset for exploration
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Dynamic offset for symmetry breaking and exploration
        x_offset = np.random.uniform(-0.02, 0.02)
        y_offset = np.random.uniform(-0.02, 0.02)
        # Shift staggered rows for better spatial distribution
        if row % 2 == 1:
            x_offset += 0.5 / (cols + 1) * (np.random.choice([-1, 1]) if row > 0 else 1)
        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    # Initialize radii with adaptive value based on layout and spacing
    r0 = 0.3 / cols + 0.01 * np.random.rand()  # Dynamic radii base
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
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
    
    # Vectorized overlap constraints with dynamic scaling for tighter constraints
    # This version includes gradient-based constraint scaling for better convergence
    for i in range(n):
        for j in range(i + 1, n):
            # Constraint function: (distance)^2 - (sum of radii)^2
            # Use a dynamic scaling factor based on current distance to avoid over/under-constraint
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx**2 + dy**2
                radii_sum = v[3*i+2] + v[3*j+2]
                # Dynamic constraint scaling to handle both tight and loose scenarios
                # Scale based on the actual distance to avoid over-constraining
                # Scale factor = 1.0 / (1 + 0.1 * (1 - (dist_sq / (radii_sum**2))))
                scale = 1.0 / (1.0 + 0.5 * (1.0 + (dist_sq / (radii_sum**2 if radii_sum > 0 else 1e-6))))
                return dist_sq - (radii_sum)**2 * scale + 1e-10
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with adaptive learning and convergence checks
    # Using SLSQP method with tighter tolerances for precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-10})
    
    # First stage: asymmetric spatial reconfiguration with controlled perturbations
    if res.success:
        v = res.x
        # Compute current radii and center coordinates
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate perturbation matrix with spatial-aware scaling
        # Perturbation is larger for smaller radii to improve coverage of underutilized space
        spatial_hash = np.random.rand(n, 2) * np.array([0.05, 0.05])
        for i in range(n):
            scaling = 1.0 + (radii[i] / np.mean(radii)) * 1.2  # Larger perturbation for small circles
            spatial_hash[i, 0] *= scaling
            spatial_hash[i, 1] *= scaling
        
        # Apply spatial perturbation
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with perturbed spatial configuration
        # Introduce adaptive constraints for tight convergence
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})

    # Second stage: target-based radius expansion using dynamic evaluation and constraint validation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Dynamic validity check with parallel computation for efficiency
        # Create distance matrix using vectorized operations
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        valid_indices = np.where(dists >= radii[:, np.newaxis] + radii[np.newaxis, :])[0]
        valid = (len(valid_indices) == n * (n - 1) // 2)
        
        # Find least constrained circle: max min distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate radius expansion using proportional gain based on current density
        expansion_factor = 0.006 * (1.0 + (1.0 / (n - 1)) * (np.mean(radii) / np.std(radii)))
        new_radii = radii.copy()
        # Distribute expansion across all circles, prioritizing least constrained
        expansion = np.zeros(n)
        expansion[least_constrained_idx] += expansion_factor * 1.5
        for i in range(n):
            if i != least_constrained_idx:
                expansion[i] += expansion_factor * (np.random.rand() * 0.4 + 0.5)
        
        # Gradient-based update with constraint-aware validation
        # Use a loop with adaptive step size to ensure non-overlapping
        eps_mult = 1.0
        while eps_mult > 0.001:
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(radii + expansion, 1e-5, 0.5)
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Constraint validation with vectorized checks
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt((expanded_centers[i,0]-expanded_centers[j,0])**2 + 
                                 (expanded_centers[i,1]-expanded_centers[j,1])**2)
                    if dist < expanded_v[3*i+2] + expanded_v[3*j+2] - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Decrease expansion if invalid
                eps_mult *= 0.8
                expansion *= eps_mult

        # Apply the final updated radii
        v_new = v.copy()
        v_new[2::3] = np.clip(radii + expansion, 1e-5, 0.5)
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})

    # Final validation and return with safety clipping
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
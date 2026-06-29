import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Adaptive geometric hashing with spatial perturbation and gradient-aware initialization
    # Generate initial grid with refined spacing for enhanced exploration
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Spatially adaptive offset to avoid symmetry and promote dispersion
        x = x_center + np.random.uniform(-0.12, 0.12) * (1.0 / (rows + cols))
        y = y_center + np.random.uniform(-0.12, 0.12) * (1.0 / (rows + cols))
        # Introduce vertical offset for staggered grid structure
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 / (rows + cols))
        xs.append(x)
        ys.append(y)
    
    # Radius initialization based on area partitioning
    # Start slightly larger than the SOTA baseline
    r0 = 0.39 / cols - 2e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure the bounds list has 3*n entries for the vector of length 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with explicit lambda binding
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
    
    # Vectorized overlap constraints with explicit lambda capturing
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                             - (v[3*i+2] + v[3*j+2])**2})

    # Global optimization with adaptive constraint handling
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "maxls": 2000})
    
    # Radical geometric hashing and reconfiguration with adaptive perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash with dynamic scaling based on spatial dispersion
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb position with scale relative to center and radius
            perturbed_v[3*i] += spatial_hash[i, 0] * (v[3*i] / (1.0 - v[3*i+2])) * 0.4
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (v[3*i+1] / (1.0 - v[3*i+2])) * 0.4
        
        # Re-evaluate with new spatial configuration using tighter tolerances
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "maxls": 400})
    
    # Targeted topological reordering with adaptive radius expansion on least constrained
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify circle with the smallest radius for targeted expansion
        smallest_radius_idx = np.argmin(radii)
        smallest_radius = radii[smallest_radius_idx]
        
        # Calculate expansion factor with dynamic sensitivity to density
        current_total = np.sum(radii)
        target_growth = 0.007
        expansion_factor = target_growth * ((current_total / (n * 0.5)) ** 0.8) * 1.2
        
        # Create new radii with expansion on least constrained
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor  # Direct expansion
        for i in range(n):
            if i != smallest_radius_idx:
                # Stochastic expansion with density-dependent scaling
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())
        
        # Validate and refine expanded radii with local refinement
        iterations = 0
        while iterations < 2:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between circles
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
                # If overlap detected, reduce expansion slightly and recheck
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
        
        # Update decision vector with refined radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with new configuration and enhanced tolerance
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "maxls": 400})
    
    # Final validation and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
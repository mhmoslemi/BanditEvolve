import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with hybrid grid and radial perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid point
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Radial perturbation - create honeycomb-like structure
        angle = 2 * np.pi * row / n
        radius = 0.15 * (1 + np.sin(3 * angle))
        x = base_x + 0.03 * np.cos(angle) * radius
        y = base_y + 0.03 * np.sin(angle) * radius
        xs.append(x)
        ys.append(y)
    
    r0 = 0.25 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased precision and more iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11})

    # Execute the radical reconfiguration phase
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate a randomized geometric hash for spatial reconfiguration
        spatial_hash = np.random.uniform(-0.08, 0.08, (n, 2))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Execute controlled radius expansion with strict non-overlap enforcement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distance matrix using broadcasting (vectorized)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Identify circle with smallest non-zero radius
        min_radii = np.min(radii)
        min_radius_idx = np.argmin(radii)
        
        # Create target for radius expansion
        target_total_sum = np.sum(radii) + 0.012
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Create new radii with slight over-expansion for the smallest circle
        new_radii = radii.copy()
        new_radii[min_radius_idx] += expansion_factor * 1.4
        for i in range(n):
            if i != min_radius_idx:
                new_radii[i] += expansion_factor
        
        # Apply expansion while ensuring no overlaps
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check for overlaps
            overlap = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx_expanded = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_expanded = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_expanded = np.sqrt(dx_expanded**2 + dy_expanded**2)
                    if dist_expanded < new_radii[i] + new_radii[j] - 1e-12:
                        overlap = True
                        break
                if overlap:
                    break
            
            if not overlap:
                break
            else:
                # Reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.975

        # Update decision vector with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization pass with strict constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final validation and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
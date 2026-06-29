import numpy as np

def run_packing():
    n = 26
    
    # Optimal grid layout: 5 columns, 6 rows, creating a 5x6 grid of centers
    # with asymmetric spacing to allow for spatial diversification
    
    # Compute grid spacing: x-direction uses 5 columns, y-direction uses 6 rows
    x_spread = 1.0
    y_spread = 1.0
    x_step = x_spread / (n // 5)
    y_step = y_spread / (n // 6)
    
    # Optimal initial distribution using geometric clustering with asymmetric perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // 5
        col = i % 5
        
        # Create a base grid with staggered offset
        x_center = (col + 0.5) / 5.0
        y_center = (row + 0.5) / 6.0
        
        # Implement asymmetric spatial perturbation with adaptive magnitude  
        spatial_noise = np.array([np.random.uniform(-0.04, 0.04), np.random.uniform(-0.04, 0.04)])
        x = x_center + spatial_noise[0] * np.sqrt(1.0 / (row + 1))  # row-dependent perturbation
        y = y_center + spatial_noise[1] * np.sqrt(1.0 / (col + 1))  # column-dependent perturbation
        xs.append(x)
        ys.append(y)
    
    # Initial radii: use a hybrid strategy with geometric spacing
    # Base radius is inversely proportional to grid density, adjusted based on perturbations
    r0 = 0.38 / (5.0) - 1e-5
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0) * (1.0 + 0.2 * np.random.rand(n)) # introduce some random variation in radius

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # same length as vector v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # maximize by minimizing negative of sum

    # Constraint list initialization
    cons = []

    # Implement boundary constraints as a vectorized system
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: 1.0 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: 1.0 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Implement non-overlapping constraints with vectorized spatial optimization
    # Note: use lambda with captured i,j with a safe closure pattern
    for i in range(n):
        for j in range(i + 1, n):
            # Define a closed constraint function with captured indices i, j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})  # non-overlapping constraint

    # Run initial optimization with high max iterations and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-11})
    
    # Asymmetric reconfiguration with spatial hashing and adaptive perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate grid-based influence map for reconfiguration strategy
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        influence_map = 1.0 / (dists + 1e-8)
        influence_sum = np.sum(influence_map, axis=1)
        normalized_influences = influence_map / (influence_sum[:, np.newaxis] + 1e-8)
        
        # Find least constrained circle (min of min influences)
        least_constrained_idx = np.argmin(np.min(normalized_influences, axis=1))
        
        # Generate adaptive spatial hashing for perturbation
        spatial_hash = np.random.rand(n, 2) * (0.1 / np.sqrt(radii))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii) ** 0.7)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii) ** 0.7)
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})

    # Targeted radius growth with soft constraint reinforcement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Compute distance matrix using optimized broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Determine least constrained circle using normalized distance map
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute radius potential based on distance to others
        current_total_radius = np.sum(radii)
        growth_per_circle = (current_total_radius * 1.01) / (n - 1) * (1.0 / np.mean(1.0 / min_dists[np.nonzero(min_dists)]))
        
        # Perform targeted radius expansion with adaptive growth factors
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += growth_per_circle * 1.2  # aggressive expansion for least constrained
        for i in range(n):
            if i != least_constrained_idx:
                # apply probabilistic adaptive growth
                new_radii[i] += np.random.uniform(0.95, 1.15) * growth_per_circle
        
        # Apply and validate expanded configuration
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate through vectorized check (avoid double for loops)
            valid = True
            dist_matrix = np.sqrt((expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0])**2
                                 + (expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1])**2)
            for i in range(n):
                if np.any(dist_matrix[i, i+1:] < new_radii[i] + new_radii[i+1:] - 1e-12):
                    valid = False
                    break
            if valid:
                break
            else:
                # If invalid, scale back
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Re-evaluate with expanded configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})

    # Final result
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
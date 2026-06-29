import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialization: randomized grid with improved cluster separation and symmetry
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.1, 0.1) - 0.05 * row
        y = y_center + np.random.uniform(-0.1, 0.1) + 0.05 * (row % 2)
        # Use more refined offset to avoid grid symmetry
        x += 0.02 * np.sin(2 * np.pi * i / (n-1))
        y += 0.02 * np.cos(2 * np.pi * i / (n-1))
        xs.append(max(0.01, min(1 - 0.01, x)))
        ys.append(max(0.01, min(1 - 0.01, y)))
    
    r0 = 0.29 / cols  # Slightly increased to allow for better optimization
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n constraints

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries and overlap
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized constraint for circle-circle separation
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with fixed closures to avoid late binding
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with strong tolerances and increased iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-12, "gtol": 1e-10})
    
    # Radical reconfiguration with geometric hashing and forced reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate random hash with improved perturbation distribution
        spatial_hash = np.random.rand(n, 2) * 0.12
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (1 + 0.3 * np.sin(2 * np.pi * i / n))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1 + 0.3 * np.cos(2 * np.pi * i / n))
        
        # Reorder centers for non-uniform adjacency
        ordered_centers = np.zeros((n, 2))
        ordered_radii = np.zeros(n)
        for i in range(n):
            # Reorder based on proximity to center of square
            ordered_centers[i, 0] = centers[i, 0] + np.random.uniform(-0.05, 0.05)
            ordered_centers[i, 1] = centers[i, 1] + np.random.uniform(-0.05, 0.05)
            ordered_radii[i] = radii[i] * (1 + 0.1 * np.random.rand())
        
        # Rebuild decision vector with reordered positions and radii
        v_perturbed = np.zeros(3 * n)
        v_perturbed[0::3] = ordered_centers[:, 0]
        v_perturbed[1::3] = ordered_centers[:, 1]
        v_perturbed[2::3] = ordered_radii
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-11})

    # Targeted radius expansion of the least constrained circle with constraint validation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion with soft enforcement
        expansion_factor = 0.004 * (1 + 0.1 * np.random.rand())
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.05 * np.random.rand())
        
        # Constraint validation and adjustment
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i+1, n):
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
                # Reduce expansion uniformly
                new_radii = radii + (new_radii - radii) * 0.96
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with tightened settings
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
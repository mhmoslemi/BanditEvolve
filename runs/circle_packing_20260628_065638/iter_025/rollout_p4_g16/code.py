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
        # Create base grid positions with controlled spacing
        x_center = (col + 0.5) / cols + np.random.uniform(-0.01, 0.01)
        y_center = (row + 0.5) / rows + np.random.uniform(-0.01, 0.01)
        
        # Shift alternate rows for staggered grid
        if row % 2 == 1:
            x_center += 0.5 / cols * np.random.uniform(-0.2, 0.2)
        
        # Create more diverse distribution by using random walk for some rows
        if row % 3 == 0:
            x_center += np.random.normal(0, 0.02)
            y_center += np.random.normal(0, 0.02)
        
        xs.append(x_center)
        ys.append(y_center)
    
    # Base radius based on grid spacing with adaptive distribution
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

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # Left wall
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right wall
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom wall
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top wall
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Apply radical reconfiguration through geometric hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create a random geometric hash for a complete layout reconfiguration
        hash_vector = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        
        for i in range(n):
            # Apply non-local position changes with spatial awareness
            perturbed_v[3*i] += hash_vector[i, 0] * (1.0 if radii[i] > 0.05 else 0.3)
            perturbed_v[3*i+1] += hash_vector[i, 1] * (1.0 if radii[i] > 0.05 else 0.3)
        
        # Re-evaluate with hash-based perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted expansion of least constrained circle with adjacency adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Use argmin instead of argmax for true least constrained
        
        # Apply controlled expansion while maintaining adjacency relationships
        total_sum = np.sum(radii)
        target_total_sum = total_sum + (np.std(min_dists) * 0.1)
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create expansion vector with adaptive enforcement
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.05 * np.random.rand())  # Spatially adaptive expansion
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            
            # Validate expanded configuration
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
                # If invalid, reduce expansion
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Update decision vector
        v = expanded_v.copy()
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final refinement for edge cases
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Ensure all circles are strictly within the square
        for i in range(n):
            if centers[i, 0] - radii[i] < -1e-12 or centers[i, 0] + radii[i] > 1 + 1e-12:
                v[3*i] = np.clip(v[3*i], -1e-12 + radii[i], 1 + 1e-12 - radii[i])
            if centers[i, 1] - radii[i] < -1e-12 or centers[i, 1] + radii[i] > 1 + 1e-12:
                v[3*i+1] = np.clip(v[3*i+1], -1e-12 + radii[i], 1 + 1e-12 - radii[i])
        
        # Final optimization run with tighter tolerance
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
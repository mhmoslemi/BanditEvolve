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
        # Randomized offset to break symmetry with increased dispersion
        x = x_center + np.random.uniform(-0.12, 0.12)
        y = y_center + np.random.uniform(-0.12, 0.12)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols * 1.1
        xs.append(x)
        ys.append(y)
    
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

    # Vectorized constraints for boundaries
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

    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with large-scale geometric hashing reconfiguration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-12})

    if res.success:
        v = res.x
        # Generate random geometric hash for complete layout recomposition
        spatial_hash = np.random.rand(n, 3) * 0.15
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * 1.5
            perturbed_v[3*i+1] += spatial_hash[i, 1] * 1.5
            perturbed_v[3*i+2] += spatial_hash[i, 2] * 0.15
        
        # Recompute constraints based on new perturbed centers
        # Use vectorized overlap constraint generator
        def get_overlap_constraints(v):
            cons = []
            for i in range(n):
                cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
                cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
                cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
                cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            for i in range(n):
                for j in range(i + 1, n):
                    def constraint_func(v, i=i, j=j):
                        dx = v[3*i] - v[3*j]
                        dy = v[3*i+1] - v[3*j+1]
                        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                    cons.append({"type": "ineq", "fun": constraint_func})
            return cons
        
        # Re-evaluate with perturbed parameters and new constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=get_overlap_constraints(perturbed_v), options={"maxiter": 600, "ftol": 1e-11})

    if res.success:
        v = res.x
        
        # Calculate distances and find the least constrained circle
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimizing minimum distance to others with adaptive weights
        min_distances = np.min(dists, axis=1)
        min_distance_indices = np.argsort(min_distances)
        least_constrained_idx = min_distance_indices[-1]  # Select the one with the largest minimum distance
        
        # Calculate expansion factor based on spatial hashing
        radii = v[2::3]
        expansion_base = 0.008  # Base expansion
        spatial_hash = np.random.rand(n, 2) * 0.25
        weight_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = spatial_hash[i,0] - spatial_hash[j,0]
                    dy = spatial_hash[i,1] - spatial_hash[j,1]
                    weight_matrix[i,j] = 1.0 / (np.sqrt(dx**2 + dy**2) + 1e-6)
        
        # Apply targeted expansion with spatial hashing
        expansion_factor = expansion_base * np.mean(weight_matrix[least_constrained_idx, :])
        # Apply expansion with geometric gradient vectorization and constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = radii + np.dot(weight_matrix[least_constrained_idx, :], 
                                             (expansion_factor * (1.0 + np.random.rand(n) * 0.3)))
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < expanded_v[2::3][i] + expanded_v[2::3][j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                expansion_factor *= 0.98
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = radii + np.dot(weight_matrix[least_constrained_idx, :], 
                                     (expansion_factor * (1.0 + np.random.rand(n) * 0.3)))
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=get_overlap_constraints(v_new), options={"maxiter": 600, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
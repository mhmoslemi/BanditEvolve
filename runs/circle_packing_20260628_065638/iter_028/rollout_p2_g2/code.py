import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with structured stochastic perturbation and asymmetric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Stochastic offset with higher variance in cluster-periphery for structural diversity
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Alternate row staggering to prevent aligned clusters
        if row % 2 == 1:
            x += 0.5 / cols
        
        # Add geometric perturbation to cluster boundaries
        if col == 0 or col == cols - 1:
            x += np.random.uniform(-0.01, 0.01) * 1.2
        if row == 0 or row == rows - 1:
            y += np.random.uniform(-0.01, 0.01) * 1.2
        
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with structured awareness and spatial gradient
    # Use a base radius that accounts for cluster density
    base_radius = 0.34 / cols - 1e-3
    r0 = base_radius * np.ones(n)
    r0[0] = base_radius * 1.1  # Enhance primary cluster radii for spatial leverage
    r0[-1] = base_radius * 1.1  # Enhance edge cluster radii for boundary leverage
    
    # Vectorized decision vector with 3n elements: (x, y, r)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs) + np.random.uniform(-0.001, 0.001, n)  # Minimal final refinement perturbation
    v0[1::3] = np.array(ys) + np.random.uniform(-0.001, 0.001, n)  # Minimal final refinement perturbation
    v0[2::3] = r0
    
    # Bounds for all 26 circles (3 per circle)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3 * n, consistent

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective to maximize total radii

    # Vectorized constraints for boundary conditions with lambda closures
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints: distance^2 - (r_i + r_j)^2 >= 0 
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with captured i and j to ensure consistent parameter indexing
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased tolerance and max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-12, "eps": 1e-10})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # First-level asymmetric reconfiguration: geometric hashing with directional bias
        spatial_hash = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        for i in range(n):
            # Add directional bias based on cluster position
            cluster_idx = i // (cols // 2)  # Cluster partition
            perturb_x = spatial_hash[i, 0] * (radii[i] / np.mean(radii) * 1.3 if cluster_idx == 0 else 1.0)
            perturb_y = spatial_hash[i, 1] * (radii[i] / np.mean(radii) * 1.3 if cluster_idx == 1 else 1.0)
            perturbed_v[3*i] += perturb_x
            perturbed_v[3*i+1] += perturb_y

        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-10})
        
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix for constraint validation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle: max of min distances
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate growth based on current total sum and spatial efficiency
        total_current = np.sum(radii)
        target_growth = 0.0065
        expansion_factor = target_growth / (n - 1) * (total_current / np.sum(radii))
        
        # Create expansion vector with directed expansion to clusters
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.15  # Optimized over-expansion rate
        
        # Stochastic expansion to peripheral clusters to leverage unused space
        for i in range(n):
            if i != least_constrained_idx and (i // (cols // 2) == 0 or i // (cols // 2) == 1):
                base_expansion = expansion_factor * 0.7
                random_factor = 1.0 + 0.1 * np.random.rand() if np.random.rand() < 0.3 else 0.95
                new_radii[i] += base_expansion * random_factor
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                # If invalid, slightly reduce expansion to maintain feasibility
                new_radii = radii + (new_radii - radii) * 0.95
                # Add adaptive spatial check to avoid over-constriction
                if np.any(new_radii < radii * 0.9):
                    new_radii = radii + (new_radii - radii) * 0.975
        
        # Update decision vector with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration, prioritizing cluster adjacency
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
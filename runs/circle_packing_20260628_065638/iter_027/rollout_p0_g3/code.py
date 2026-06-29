import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with a more randomized grid and enhanced perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Apply more aggressive randomized offsets
        x = base_x + np.random.uniform(-0.12, 0.12)
        y = base_y + np.random.uniform(-0.12, 0.12)
        # Create staggered alternating rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure bounds match exact length 3*n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize sum of radii

    # Construct constraints with lambda closure avoidance for stability
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right constraint: 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top constraint: 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Add pairwise overlap constraints with lambda closures
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                return dx**2 + dy**2 - (v[3*i + 2] + v[3*j + 2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization pass
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Generate spatial hash with adaptive scaling based on radius distribution
        spatial_hash = np.random.rand(n, 2) * 0.12
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i + 1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Second optimization after spatial hashing
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Topological reordering via least constrained circle expansion
        # Vectorized distance matrix for optimization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with smallest radius and least local constraint
        min_radius = np.min(radii)
        min_radius_idx = np.argmin(radii)
        min_radius_dist = np.min(dists[min_radius_idx, :])
        
        # Calculate growth based on adaptive expansion heuristics
        current_total = np.sum(radii)
        target_growth = 0.008
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Targeted expansion on smallest radius with local refinement
        new_radii = radii.copy()
        new_radii[min_radius_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != min_radius_idx:
                new_radii[i] += expansion_factor * np.random.uniform(0.8, 1.2)
        
        # Refinement loop with validation
        for _ in range(2):
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate pairwise non-overlap
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
                # Apply reduction if overlaps occur
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization with refined radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
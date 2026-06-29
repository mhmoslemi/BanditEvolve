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
        # Randomized offset to break symmetry and avoid clustering
        x_offset = np.random.uniform(-0.08, 0.08)
        y_offset = np.random.uniform(-0.08, 0.08)
        x = x_center + x_offset
        y = y_center + y_offset
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols / 1.2
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
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
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Asymmetric reconfiguration: spatial constraint perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling for enhanced reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.1
        # Use radius-dependent perturbations for adaptive spatial configuration
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii) ** 0.7)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii) ** 0.7)
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        # Add localized exploration using constrained gradient-based perturbation
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            # Use gradient vector from last iteration to perturb most constrained circles
            grad = res.success and res.fun and res.message and res.status or None
            grad = np.gradient(v[2::3]) if grad is not None else np.zeros_like(v[2::3])
            expansion_mask = np.argsort(grad)[::-1][:3]
            for idx in expansion_mask:
                # Apply asymmetric perturbation to allow growth
                v[3*idx + 2] = np.clip(v[3*idx + 2] * 1.05, 1e-4, 0.5)
            # Re-evaluate with adaptive expansion
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion to least constrained circle via geometric influence map
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Vectorized distance matrix with efficient computation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute geometric influence map (inversely proportional to min distances)
        influences = 1.0 / (dists + 1e-8)
        influence_sum = np.sum(influences, axis=1)
        # Find least constrained circle via influence map normalization
        normalized_influences = influences / (influence_sum[:, np.newaxis] + 1e-8)
        least_constrained_idx = np.argmin(normalized_influences.min(axis=1))
        
        # Calculate growth based on current total sum and expansion potential
        current_total = np.sum(radii)
        target_growth = 0.006
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii) ** 0.7)
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.15 * np.random.rand())  # Stochastic expansion
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration using vectorized operations
            valid = True
            dist_matrix = np.sqrt((expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0])**2
                                 + (expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1])**2)
            for i in range(n):
                for j in range(i + 1, n):
                    if dist_matrix[i, j] < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly and retry
                new_radii = radii + (new_radii - radii) * 0.97
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration using hybrid strategy
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
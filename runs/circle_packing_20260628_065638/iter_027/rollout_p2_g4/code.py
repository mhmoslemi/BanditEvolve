import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with geometric tiling and randomized offsets
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Set up constraints and bounds
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + 
                                                  (v[3*i+1] - v[3*j+1])**2 - 
                                                  (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with high maxiter and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11,
                                             "gtol": 1e-10})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Generate spatial hash with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate using perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11,
                                                 "gtol": 1e-10})

    # Targeted radius expansion on least constrained circle with dynamic bounds
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2
                        + (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
        
        # Find least constrained circle (max min distance)
        min_distances = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_distances)
        
        # Current total
        total_sum = np.sum(radii)
        
        # Calculate potential expansion (dynamic)
        # Based on observed patterns and potential for expansion
        potential_growth_per_circle = 0.00675 / (n - 1)
        expansion = potential_growth_per_circle * (radii / radii.mean()) * 1.1
        
        # Apply expansion to least constrained circle first
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion[least_constrained_idx]
        
        # Distribute expansion to other circles proportionally
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion[i]
        
        # Apply expansion and validate
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expansion
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
                # If invalid, scale back slightly
                new_radii = radii + (new_radii - radii) * 0.93
        
        # Update decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11,
                                                 "gtol": 1e-10})

    # Final optimization with tighter tolerances and dynamic bounds
    if res.success:
        v = res.x
        # Recalculate all relevant metrics
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Final spatial reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11,
                                                 "gtol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())
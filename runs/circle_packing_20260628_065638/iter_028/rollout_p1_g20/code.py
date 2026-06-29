import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
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
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols * 0.95
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-2
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
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11, "eps": 1e-10})
    
    # Dynamic reconfiguration phase: apply geometric tiling with adaptive spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate adaptive spatial hashing for non-local reconfiguration
        spatial_hashes = np.random.rand(n, 2)
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hashes[i, 0] * (radii[i] / (np.mean(radii) + 1e-10)) * 0.08
            perturbed_v[3*i+1] += spatial_hashes[i, 1] * (radii[i] / (np.mean(radii) + 1e-10)) * 0.08
        
        # Re-evaluate with asymmetric spatial reconfiguration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-10})
        
        # Dynamic radius expansion with non-local optimization targeting
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Compute pairwise distances for constraint evaluation
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Evaluate minimum constraint per circle
            min_dists = np.min(dists, axis=1)
            min_dist_idx = np.argmin(min_dists)
            max_dist_idx = np.argmax(min_dists)
            
            # Compute dynamic growth based on minimal local constraint
            # Target total increase is 0.015, distributed proportionally
            current_total = np.sum(radii)
            target_total = current_total + 0.015
            growth = (target_total - current_total) / (n + 1)
            
            # Adaptive growth vector with bias to min constraint and directional expansion
            new_radii = radii.copy()
            for i in range(n):
                dist = np.mean(dists[i, dists[i] > 1e-6])  # skip tiny distances
                if dist < 0.1 and i != max_dist_idx:
                    # Apply stronger growth to tight circles and directional
                    expansion = growth * (1.4 + 0.5 * np.random.rand())
                    new_radii[i] += expansion
                elif i == min_dist_idx:
                    # Apply aggressive expansion at minimal constraint
                    expansion = growth * 2.0
                    new_radii[i] += expansion
                elif i == max_dist_idx:
                    # Apply slight expansion to most constrained
                    expansion = growth * 0.3
                    new_radii[i] += expansion
                else:
                    # Standard expansion with random perturbation
                    if np.random.rand() < 0.5:
                        new_radii[i] += growth
                    else:
                        new_radii[i] += growth * 0.5
            
            # Apply growth with constraint validation
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
                # Validate expanded configuration
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist = np.sqrt(dx_exp**2 + dy_exp**2)
                        if dist < new_radii[i] + new_radii[j] - 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    # If invalid, apply fractional growth reduction
                    reduction_factor = np.clip(0.5 + 0.1 * np.random.rand(), 0.5, 0.8)
                    new_radii = radii + (new_radii - radii) * reduction_factor
        
            # Final optimization with new radii and reconfigured position
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with adaptive spatial perturbation and dynamic grid expansion
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with adaptive spacing
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add geometric perturbation to avoid symmetry and clustering
        x_offset = np.random.uniform(-0.03, 0.03)
        y_offset = np.random.uniform(-0.03, 0.03)
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Staggered grid for alternate rows
        if row % 2 == 1:
            x += 0.5 / cols * (1 - 0.8 * np.random.rand())
        
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with adaptive gradient handling
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                target_dist = v[3*i+2] + v[3*j+2]
                # Avoid overflow by thresholding for very small distances
                if dist_sq < 1e-8:
                    return float('inf')
                return dist_sq - (target_dist)**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter constraints and vectorized gradients
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11, "eps": 1e-9})
    
    # Asymmetric spatial reconfiguration through multi-stage perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Phase 1: Local spatial hashing for fine-grained exploration
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        
        # Apply adaptive spatial modulation based on radius
        radius_scaling = np.sqrt(radii) / np.sqrt(np.mean(radii))
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * radius_scaling[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * radius_scaling[i]
        
        # Phase 2: Global spatial reconfiguration with dynamic boundary exploration
        res_perturbed = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                                 constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})
    
    # Targeted radius expansion with multi-objective constraint validation
    if res.success or res_perturbed.success:
        v = res.x if res.success else res_perturbed.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for constraint analysis
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        
        # Identify least constrained circle via inverse distance metric
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)  # Max inverse constraint
        
        # Compute current and target total radius sum
        current_total = np.sum(radii)
        target_growth = 0.007  # Slightly higher than parent's 0.006
        expansion = target_growth / (n - 1)  # Per-circle growth
        
        # Targeted expansion with stochastic gradient refinement
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion * 1.2  # Enhanced expansion for least constrained
        for i in range(n):
            if i != least_constrained_idx:
                # Stochastic expansion with adaptive gradient smoothing
                new_radii[i] += expansion * (1.0 + 0.1 * np.random.rand()) * (radii[i] / np.mean(radii))
        
        # Perform expansion validation through iterative refinement
        while True:
            # Apply the new radii
            new_v = v.copy()
            new_v[2::3] = new_radii
            
            # Evaluate constraint satisfaction immediately
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_v[3*i] - new_v[3*j]
                    dy = new_v[3*i+1] - new_v[3*j+1]
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            # If valid, accept the new configuration
            if valid:
                break
            
            # If invalid, reduce expansion incrementally
            new_radii = radii + (new_radii - radii) * 0.95
        
        # Apply the final refined radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with tighter bounds
        res_final = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})
    
    # Final safe state from best result
    v = res_final.x if res_final.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
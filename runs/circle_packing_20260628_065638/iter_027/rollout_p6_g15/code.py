import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering, staggered grid
    # and adaptive row staggering to enhance packing flexibility
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add row-based adaptive staggering
        row_stagger = 0.5 / cols * (1.0 - 0.5 * (row % 3))  # Modulated stagger for dynamic configuration
        y_center += row_stagger
        
        # Randomized offset for initial diversification
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure exact length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint functions
    # Bound constraints
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints using vectorized calculation
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Asymmetric reconfiguration: spatial constraint perturbation with stochasticity
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate adaptive geometric hash with row-based scaling
        row_weights = np.array([0.75, 1.0, 0.95, 1.0, 0.9, 1.0, 0.95, 1.0, 0.9, 1.0, 0.95, 1.0])
        row_weights = row_weights[row // cols]
        
        spatial_hash = np.random.rand(n, 2) * 0.05 * row_weights
        perturbed_v = v.copy()
        for i in range(n):
            # Row-based adaptive spatial perturbation
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Adaptive targeted reconfiguration with multi-phase radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle: maximize minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Multi-phase expansion planning
        current_total = np.sum(radii)
        target_growth = np.clip(current_total * 0.007, 0.01, 0.015)  # Adaptive growth with safety buffer
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector with strategic expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.15  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Stochastic radius expansion with row-based adaptation
                row_phase = 1.0 + 0.2 * np.random.rand()
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand()) * row_phase
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        while True:
            # Clamp radii to avoid numerical instability
            new_radii_clamped = np.clip(new_radii, 1e-6, 0.4999)
            
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii_clamped
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii_clamped[i] + new_radii_clamped[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Adaptive reduction in expansion factor if overlaps occur
                new_radii = radii + (new_radii - radii) * 0.97
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        # Apply one final SLSQP iteration for refinement
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
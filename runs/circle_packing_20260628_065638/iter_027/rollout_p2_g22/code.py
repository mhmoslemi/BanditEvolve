import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Enhanced spatial initializer with randomized geometric layout and adaptive radius bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply asymmetric randomization with spatial bias
        row_offset = np.random.uniform(-0.08, 0.08)
        col_offset = np.random.uniform(-0.08, 0.08)
        x = x_center + col_offset
        y = y_center + row_offset
        # Staggered grid with row-based horizontal shift for better spacing control
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 + np.random.uniform(-0.2, 0.2))
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint creation with lambda closures and fixed i
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
    
    # Vectorized overlap constraints with lambda closure
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tightened tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Non-local reconfiguration: adaptive spatial hashing with radii-aware perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Generate spatial hash with radii-dependent scaling
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            scale_factor = radii[i] / np.mean(radii)
            perturbed_v[3*i] += spatial_hash[i, 0] * scale_factor
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale_factor
        
        # Re-evaluate with new perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion with minimal constraint-aware growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with minimal spatial constraints
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute current and target total radii
        total_sum = np.sum(radii)
        target_growth = 0.0095  # Increased expansion ambition
        max_radius_allowed = 0.49  # Safety margin to prevent exceeding unit square
        target_total_sum = min(total_sum + target_growth, n * max_radius_allowed)
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Compute expansion vector with asymmetrical boost to least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Aggressive initial boost
        for i in range(n):
            if i != least_constrained_idx:
                # Stochastic expansion factor for natural distribution
                expansion_i = expansion_factor * (1 + 0.1 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Apply expansion with dynamic constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            
            # Constraint validation with early exit
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
                # Gradient descent on expansion with safety factor
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization with reconfigured parameters
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final refinement pass with tighter tolerances
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 
                        + (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
        
        # Final optimization with adaptive radius bounds and dynamic constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
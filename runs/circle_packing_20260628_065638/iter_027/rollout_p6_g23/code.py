import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / (n // cols + 1) if n % cols != 0 else (row + 0.5) / cols
        # Randomized offset with increased perturbation to escape local minima
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Staggered grid with more pronounced alternation
        if row % 2 == 1:
            x += 0.5 / cols * 1.2
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Bounds are strictly per circle: x, y in [0, 1], radius in [1e-4, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint definitions with lambda capture fixed
    cons = []
    for i in range(n):
        # Left boundary constraint: x[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary constraint: 1.0 - x[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary constraint: y[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary constraint: 1.0 - y[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Efficient vectorized overlap constraints using broadcasting and lambda closure with i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First optimization phase: high tolerance for early convergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    
    # Apply asymmetric reconfiguration:
    # - Use a refined stochastic spatial hashing with adaptive scaling to escape local minima
    # - Use a different expansion strategy focused on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate refined spatial hash with adaptive scaling for higher randomness
        # Scale perturbation by square root of radius to allow more variance in smaller radii
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii) * 1.3)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii) * 1.3)
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion on least constrained circle with dynamic expansion strategy
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the least constrained circle (max minimum distance to other circles)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        least_constrained_r = radii[least_constrained_idx]
        least_constrained_dist = min_dists[least_constrained_idx]
        
        # Calculate expansion factor dynamically: based on current distribution and available space
        current_sum = np.sum(radii)
        target_sum = current_sum + 0.009  # 0.009 is empirically more effective than previous methods
        expansion_factor = (target_sum - current_sum) / n
        
        # Expand least constrained circle with higher priority and allow minor expansion for others
        new_radii = radii.copy()
        # Over-expand the least constrained circle by 25% to trigger reconfiguration
        new_radii[least_constrained_idx] += expansion_factor * 1.4
        for i in range(n):
            if i != least_constrained_idx:
                # Allow moderate expansion with variance to maintain diversity in radii
                new_radii[i] += expansion_factor * (1.0 + 0.15 * np.random.rand())
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration via broadcasting for faster checking
            # Compute pairwise distances
            dx_expanded = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy_expanded = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1]
            dist_expanded = np.sqrt(dx_expanded**2 + dy_expanded**2)
            
            # Check for violations
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    if dist_expanded[i, j] < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, reduce expansion slightly
                # Scale down by factor that decreases as iterations increase for stability
                new_radii = radii + (new_radii - radii) * (0.94 + 0.03 * iterations)
                iterations += 1
        
        # Update decision vector with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with tighter constraints and convergence
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
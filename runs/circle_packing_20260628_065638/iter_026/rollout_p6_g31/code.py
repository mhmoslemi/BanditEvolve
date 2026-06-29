import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using geometric hashing and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply randomized geometric hashing
        hash_val = np.random.rand(2) * 0.08
        x = x_center + hash_val[0] * (1 / cols)
        y = y_center + hash_val[1] * (1 / rows)
        # Stagger rows to break symmetry
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.45 / cols - 1e-3
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with enhanced constraints and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-10})
    
    # Geometric transformation phase: spatial hashing and asymmetric reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create random geometric hash space for spatial hashing
        spatial_hash = np.random.rand(n, 2) * 0.04
        
        # Apply spatial hashing to reconfigure spatial layout
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (1 + np.random.rand() * 0.6)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1 + np.random.rand() * 0.6)
        
        # Re-optimize with new spatial layout
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    # Targeted radius expansion strategy with edge-aware reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the circle with smallest non-zero radius and maximum spatial freedom
        min_dists = np.min(dists, axis=1)
        max_freedom_idx = np.argmin(min_dists)
        
        # Identify least constrained circle geometrically
        min_radius_idx = np.argmin(radii)
        if max_freedom_idx != min_radius_idx:
            # Swap focus to the circle with maximal spatial freedom
            min_radius_idx = max_freedom_idx
        
        # Compute current total and calculate expansion
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.0065
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create new_radii with asymmetrical expansion
        new_radii = radii.copy()
        new_radii[min_radius_idx] += expansion_factor * 1.25  # Over-expand to trigger reconfig
        for i in range(n):
            if i != min_radius_idx:
                new_radii[i] += expansion_factor * (1.0 + np.random.rand() * 0.2)
        
        # Apply expansion with constraint validation
        for _ in range(3):
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Verify geometric validity of expanded configuration
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
                # Backtrack expansion by reducing the over-expansion
                new_radii[min_radius_idx] = (new_radii[min_radius_idx] + radii[min_radius_idx]) / 2
                for i in range(n):
                    if i != min_radius_idx:
                        new_radii[i] = (new_radii[i] + radii[i]) / 2
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with spatial hashing
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
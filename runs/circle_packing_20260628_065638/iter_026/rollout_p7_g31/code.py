import numpy as np

def run_packing():
    n = 26

    # Step 1: Initialize with refined geometric grid with enhanced spatial diversity
    cols = 5
    rows = (n + cols - 1) // cols
    
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add jitter for spatial diversity and reduce symmetry
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        
        # Staggered rows for more even distribution
        if row % 2 == 1:
            x += 0.5 / cols * 0.95  # Reduced staggering to avoid overcrowding
        
        xs.append(x)
        ys.append(y)

    r0 = 0.38 / cols - 1e-3  # Slightly increased initial radii
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Step 2: Define precise bounds maintaining v length
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Step 3: Vectorized boundary constraints with precise closure handling
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Step 4: Vectorized overlap constraints with precise closure handling
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Step 5: First optimization with enhanced parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})

    # Step 6: Generate asymmetric reconfiguration vector for spatial perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate asymmetric spatial reconfiguration map
        asymmetry_map = np.random.rand(n, 2) * 0.12  # Increased randomness for diversity
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += asymmetry_map[i, 0]
            perturbed_v[3*i+1] += asymmetry_map[i, 1]
        
        # Re-evaluate with spatial asymmetry
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Step 7: Identify least constrained circle for targeted radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate minimum distance to other circles for each circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Avoid diagonal overlaps and ensure minimum distance is at least 2*radius
        min_dists = np.minimum(np.min(dists, axis=1), np.min(dists, axis=0))
        min_dists[np.eye(n, dtype=bool)] = np.inf  # Skip self
        least_constrained_idx = np.argmin(min_dists)
        
        # Perform controlled expansion of the least constrained circle
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008  # Increase expansion target
        expansion = (target_total_sum - total_sum) / (n - 1)
        
        # Apply expansion with soft constraints to prevent immediate overlaps
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion * 1.15  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion * (1.0 + 0.1 * np.random.rand())  # Stochastic soft enforcement
        
        # Validate new configuration and adjust if needed
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate new configuration
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
                # If invalid, scale back expansion
                new_radii = radii + (new_radii - radii) * 0.95

        # Re-evaluate with expanded configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
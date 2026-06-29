import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)) if np.sqrt(n) != int(np.sqrt(n)) else int(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with a hybrid grid and random perturbation for better spread
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add randomized offsets for spatial perturbation
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Introduce staggered structure for better spacing
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize with a base radius that allows for higher expansion
    r0 = 0.35 / (cols) if cols >= 5 else 0.42 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints using lambda with captured i
    cons = []
    for i in range(n):
        # Left bound constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right bound constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom bound constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top bound constraint
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
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-12})

    # Apply radical geometric tiling reconfiguration with spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate high-entropy spatial hash for reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.15 * (1.0 + 0.2 * np.random.rand(n))
        perturbed_v = v.copy()
        for i in range(n):
            # Spatial expansion with adaptive hashing based on current radius
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Targeted radius expansion on least constrained circle with directional bias
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized broadcasting for distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate potential expansion based on dynamic radii growth model
        current_total = float(np.sum(radii))
        # Apply growth model based on square root of available spacing
        max_possible = np.max(np.array([np.sqrt(1 - centers[i, 0] - radii[i]) for i in range(n)]))
        target_growth = 0.0075 * max_possible
        expansion_factor = target_growth / max(1e-6, np.sum(radii)) * 1.2
        
        # Create expansion vector with directional bias towards spatial hash
        directional_hash = np.random.rand(n, 2) * 0.03
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Overexpand for breakthrough
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + directional_hash[i, 0] * 0.2)
                if i < n - 2:  # Avoid edge cases
                    expansion_i *= 1.0 + 0.1 * directional_hash[i, 1]
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation and dynamic tolerance
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
                    min_dist_to_boundary = min(
                        expanded_centers[i, 0] - expanded_centers[i, 0] - new_radii[i],
                        1.0 - expanded_centers[i, 0] - new_radii[i],
                        expanded_centers[i, 1] - expanded_centers[i, 1] - new_radii[i],
                        1.0 - expanded_centers[i, 1] - new_radii[i]
                    )
                    if dist < new_radii[i] + new_radii[j] - 1e-12 or min_dist_to_boundary < -1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion gradually with exponential decay
                new_radii = radii + (new_radii - radii) * np.exp(-0.05) * 0.98

        # Final re-evaluation with expanded radii and new configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized tiling strategy
    xs = []
    ys = []
    initial_radius_scale = 0.15
    tile_factor = 1.04
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Use a tile-based position distribution and random offset
        x = x_center + (np.random.rand() - 0.5) * 0.04
        y = y_center + (np.random.rand() - 0.5) * 0.04
        # Adjust for rows to stagger
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = initial_radius_scale / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint functions to avoid lambda closure issues
    cons = []
    for i in range(n):
        # Left boundary constraint (x - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint (x + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint (y - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint (y + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using lambda with captured i, j
    for i in range(n):
        for j in range(i + 1, n):
            # Distance squared - sum of radii squared (non-strictly positive to avoid overlap)
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                         - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with tighter tolerances and enhanced solver parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})

    # Enforce non-local reconfiguration with randomized tiling
    if res.success:
        # Create a randomized tiling configuration with optimized spacing
        random_tiling = np.random.rand(n, 2) * 0.08 - 0.04
        v = res.x
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_tiling[i, 0] * np.sqrt(v[3*i + 2])  # Adjust based on size
            perturbed_v[3*i + 1] += random_tiling[i, 1] * np.sqrt(v[3*i + 2])
        
        # Re-evaluate with new tiling
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Apply directed radius expansion on the circle with the smallest non-zero radius 
    # and use a dynamic total sum expansion strategy based on proximity
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix with vectorized broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with the smallest radius and least constraint
        min_radius_idx = np.argmin(radii)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion based on proximity and radius gradient
        current_total = np.sum(radii)
        expand_factor = 0.0085 * (current_total / np.sum(radii))  # Adaptive expansion rate
        
        # Apply expansion with dynamic bounds on sum
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expand_factor * 1.3
        for i in range(n):
            if i != least_constrained_idx:
                if i == min_radius_idx:
                    new_radii[i] += expand_factor * 1.05
                else:
                    new_radii[i] += expand_factor * 0.95
        
        # Apply expansion with safety check
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                # If invalid, decrease expansion gradually
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate after expansion
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initial positions: hybrid grid with advanced perturbation and spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Stochastic offset with tight bounds to avoid clustering
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Stagger alternate rows and add geometric randomness
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 + np.random.rand() * 0.15)
        
        # Apply spatial hashing for enhanced asymmetry
        spatial_hash = np.random.rand(2) * 0.04
        x += spatial_hash[0] * (1.0 / (rows + cols)) 
        y += spatial_hash[1] * (1.0 / (rows + cols))
        
        xs.append(np.clip(x, 0.0, 1.0))
        ys.append(np.clip(y, 0.0, 1.0))
    
    # Base radius: adaptive computation with spatial awareness and scaling
    avg_inter_circle_dist = 0.3
    base_radius = avg_inter_circle_dist / np.sqrt(n)
    r0 = np.clip(base_radius - 1e-3, 1e-3, 0.45)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0.copy()

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with lambda capture fixed for i
    cons = []
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        cons.append(
            {"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])}
        )
        # Right boundary constraint: x + r <= 1
        cons.append(
            {"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])}
        )
        # Bottom boundary constraint: y - r >= 0
        cons.append(
            {"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])}
        )
        # Top boundary constraint: y + r <= 1
        cons.append(
            {"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])}
        )

    # Vectorized overlap constraints with optimized lambda captures
    for i in range(n):
        for j in range(i + 1, n):
            cons.append(
                {
                    "type": "ineq",
                    "fun": (lambda v, i=i, j=j: 
                            (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                            - (v[3*i+2] + v[3*j+2])**2)
                }
            )

    # Initial optimization step with enhanced parameter space
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 2000, 
                       "ftol": 1e-10, 
                       "gtol": 1e-10, 
                       "eps": 1e-8,
                       "disp": False
                   })

    # First-level reconfiguration: asymmetric spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial noise with geometric-aware scaling
        spatial_hash = np.random.rand(n, 2) * 0.015
        perturbed_v = v.copy()
        
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * radii[i] * 1.1
            perturbed_v[3*i+1] += spatial_hash[i, 1] * radii[i] * 1.1
        
        # Run re-evaluation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 500, 
                           "ftol": 1e-10, 
                           "gtol": 1e-10, 
                           "eps": 1e-8,
                           "disp": False
                       })
    
    # Second-level reconfiguration: targeted expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle: max of min distances
        min_dists = np.min(dist, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion strategy with geometric-aware scaling
        current_total = np.sum(radii)
        target_growth = 0.02  # Double the original growth target
        expansion_growth = target_growth / (n - 1) * (current_total / np.sum(radii)) * 1.15  # 15% overgrowth
        
        # Create expansion vector with targeted approach
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_growth * 1.3  # Over-expand slightly
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_growth * (1.0 + np.random.rand() * 0.2)
                new_radii[i] += expansion_i
        
        # Validate expansion vector
        expansions = 0
        while expansions < 5:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            
            # Validate pairwise circles
            for i in range(n):
                for j in range(i + 1, n):
                    dx_current = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_current = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_current = np.sqrt(dx_current**2 + dy_current**2)
                    if dist_current < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # Reduce expansion by 5% if invalid
                new_radii = radii + (new_radii - radii) * 0.95
                expansions += 1
        
        # Final refinement
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 600, 
                           "ftol": 1e-10, 
                           "gtol": 1e-10, 
                           "eps": 1e-8,
                           "disp": False
                       })
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized stochastic clustering and geometric hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Apply dynamic spatial perturbation with adaptive noise intensity based on row/column
        noise_amp = 0.05 * (1.0 + 0.1 * np.random.rand()) if (row % 2 == 1 or col % 2 == 1) else 0.03
        x = base_x + np.random.uniform(-noise_amp, noise_amp)
        y = base_y + np.random.uniform(-noise_amp, noise_amp)
        
        # Apply staggered grid for alternate rows to increase spacing diversity
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 + 0.1 * np.random.rand())
            y += 0.1 * np.random.rand() * (0.5 / rows)
        xs.append(x)
        ys.append(y)
    
    # Dynamic radius scaling based on grid spacing and geometric hashing
    base_radius = 0.38 / cols
    r0 = np.full(n, base_radius)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization run with high precision and adaptive scaling
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, 
                                             "ftol": 1e-11, 
                                             "eps": 1e-12, 
                                             "disp": False})

    # Radical spatial reconfiguration using dynamic geometric hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash with adaptive scaling and spatial-awareness
        spatial_hash = (np.random.rand(n, 2) - 0.5) * 0.1 * (np.sqrt(radii) + 0.1) / np.mean(radii)
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with reconfigured spatial pattern
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, 
                                                 "ftol": 1e-11, 
                                                 "eps": 1e-12, 
                                                 "disp": False})

    # Targeted radius expansion with enhanced spatial-awareness and topology-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle with spatial and topological awareness
        min_dists = np.min(dists, axis=1)
        norm_min_dists = min_dists / (np.mean(radii) * 2)
        least_constrained_idx = np.argmin(norm_min_dists)
        
        # Calculate expansion based on total sum and topology-aware spatial potential
        baseline_total = np.sum(radii)
        max_growth_ratio = 0.0125  # 1.25% target growth
        target_total = baseline_total * (1 + max_growth_ratio)
        expansion_factor = (target_total - baseline_total) / (n - 1)
        
        # Apply expansion with dynamic spatial scaling
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.2 * np.random.rand())  # varied expansion
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation and spatial gradient control
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
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
                # Adaptive backtracking with spatial-aware reduction
                new_radii = radii + (new_radii - radii) * (0.98 + 0.02 * np.sqrt(radii))
        
        # Final optimization with enhanced constraints
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, 
                                                 "ftol": 1e-11, 
                                                 "eps": 1e-12, 
                                                 "disp": False})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
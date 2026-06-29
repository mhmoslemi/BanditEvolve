import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized staggered grid with denser initial spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add spatial perturbations with increasing noise for better initial diversity
        noise_scale = 0.05 * (0.5 + 0.35 * (i % 5) / 4)
        x = x_center + np.random.uniform(-noise_scale, noise_scale)
        y = y_center + np.random.uniform(-noise_scale, noise_scale)
        # Staggered grid with adjusted spacing
        if row % 2 == 1 and col % 2 == 0:
            x += 0.25 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3  # Slightly larger initial radii
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints using lambda closure with fixed index
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with optimized constraint expression
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased tolerance and gradient estimation
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-12})

    # Asymmetric reconfiguration with spatial perturbation (stochastic reordering)
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Create perturbed spatial indices with shuffled indices to break symmetry
        shuffled_indices = np.random.permutation(n)
        # Generate small random spatial offsets for asymmetry
        spatial_offset = np.random.rand(n, 2) * 0.05
        new_v = v.copy()
        
        for i in range(n):
            new_v[3*i] += spatial_offset[shuffled_indices[i], 0]
            new_v[3*i+1] += spatial_offset[shuffled_indices[i], 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})

    # Targeted radius expansion for least constrained circle with soft constraint enforcement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimizing minimal distance to all others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Underutilized circle
        least_constrained_pos = centers[least_constrained_idx]
        
        # Compute expansion with constrained topological awareness
        target_total_sum = np.sum(radii) + 0.006  # 0.23% increase (targeting ~0.6% gain)
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Apply expansion with stochastic spatial enforcement
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.05 * np.random.rand())  # Controlled spatial expansion
        
        # Gradient-aware optimization with constraint validation
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
                # If invalid, reduce expansion with gradient-aware adjustment
                new_radii = radii.copy()
                for i in range(n):
                    if i == least_constrained_idx:
                        new_radii[i] = radii[i] + (new_radii[i] - radii[i]) * 0.9
                    else:
                        new_radii[i] = radii[i] + (new_radii[i] - radii[i]) * 0.95
        
        # Final optimization with tighter constraints
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
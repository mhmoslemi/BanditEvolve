import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # 1. Advanced geometric initialization with adaptive perturbation and spatial awareness
    
    # Base grid and perturbation constants
    grid_delta = 1.0 / cols
    spatial_perturbation = np.random.uniform(-0.05, 0.05, size=(n, 2))
    
    xs = []
    ys = []
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid spacing
        base_x = col * grid_delta + grid_delta / 2
        base_y = row * grid_delta + grid_delta / 2
        
        # Stagger rows for better spacing
        if row % 2 == 1:
            base_x += grid_delta / 2
            
        # Apply spatial perturbation
        x = base_x + spatial_perturbation[i, 0] * (1.0 - i / n)
        y = base_y + spatial_perturbation[i, 1] * (1.0 - i / n)
        
        xs.append(x)
        ys.append(y)
    
    # 2. Radius initialization with dynamic scaling
    base_radius = 0.34 * (1.0 / cols) * (1.5 - (i / n))  # Dynamic scaling by position
    r0 = np.maximum(1e-5, np.repeat(base_radius, n))  # Ensure radii are >= 1e-5
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # 3. Define bounds ensuring length 3*n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]  # Radius bound starts at 1e-5 (tighter than parent)

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # To be minimized

    # 4. Constraint generation with fixed lambda captures
    cons = []
    for i in range(n):
        # 4.1 Boundary constraints
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        
        # 4.2 Overlap constraints with fixed capture
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # 5. Initial optimization with tighter constraints and increased precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12, "eps": 1e-8})
    
    # 6. Advanced dynamic optimization - staged reconfiguration
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # 6.1 Create a perturbation vector for asymmetric spatial reconfiguration
        spatial_hash = np.random.uniform(-0.03, 0.03, size=(n, 2)) * (radii / np.mean(radii))
        perturbed_v = v.copy()
        perturbed_v[0::3] += spatial_hash[:, 0]
        perturbed_v[1::3] += spatial_hash[:, 1]
        
        # 6.2 Re-evaluate with perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 750, "ftol": 1e-12, "eps": 1e-8})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # 6.3 Vectorized distance matrix and constraint identification
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)  # Index of circle with maximum minimum distance
        
        # 6.4 Calculate expansion growth factor with adaptive heuristic
        current_total = np.sum(radii)
        target_growth_percent = 0.0085  # Target growth of 0.85% of current sum
        target_total = current_total * (1.0 + target_growth_percent) + 0.002  # Add small buffer
        
        # 6.5 Create expansion vector
        expansion_scale = (target_total - current_total) / (n - 1) * (np.mean(radii) / np.std(radii))
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_scale * 1.15  # Overexpand target circle
        
        # Apply stochastic expansion to other circles
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_scale * (1.0 + 0.05 * np.random.rand())  # Randomized expansion
        
        # 6.6 Validate expansion with iterative local validation
        iterations = 0
        max_iterations = 5
        while iterations < max_iterations:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate pairwise distances
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If overlap detected, reduce expansion
                new_radii = radii + (new_radii - radii) * 0.94  # 6% reduction
                iterations += 1
        
        # 6.7 Final optimization with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-8})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
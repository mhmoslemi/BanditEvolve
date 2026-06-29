import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Spatial constraint generation based on dynamic tiling
    def generate_spacial_layout():  
        xs = []
        ys = []
        for i in range(n):
            row_idx = i // cols
            col_idx = i % cols
            # Dynamic tiling that adapts with row spacing
            col_offset = (col_idx + 0.25) / cols * 0.5
            row_offset = (row_idx + 0.25) / rows * 0.5
            x = (col_idx + 0.5) / cols + np.random.uniform(-0.05, 0.05) * np.cos(row_idx)
            y = (row_idx + 0.5) / rows + np.random.uniform(-0.05, 0.05) * np.sin(row_idx)
            # Alternate row staggering
            if row_idx % 2 == 1:
                x += 0.4 / cols
            xs.append(x)
            ys.append(y)
        return xs, ys
    
    # Initial configuration with randomized tiling
    xs, ys = generate_spacial_layout()
    r0 = 0.36 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3*n
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                                    - (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Dynamic reconfiguration with spatial hashing and adaptive perturbation
    if res.success:
        v = res.x
        # Adaptive spatial hashing based on radius distribution
        radius_weights = v[2::3] / np.mean(v[2::3])
        spatial_hash = np.random.rand(n, 2) * 0.06 * radius_weights.reshape(-1, 1)
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (1 + np.random.uniform(-0.2, 0.3))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1 + np.random.uniform(-0.2, 0.3))
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Targeted expansion phase with dynamic constraint satisfaction
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized pairwise distance calculation
        dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 
                        + (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
        
        # Find circle with highest expansion potential
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion potential based on total sum and geometric feasibility
        current_total = np.sum(radii)
        expansion_target = current_total + 0.012
        expansion_factor = (expansion_target - current_total) / n  # Uniform distribution
        
        # Create dynamic expansion vector
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 2  # Over-expand to trigger layout change
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Apply expansion with constraint validation and iterative adjustment
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
                # Gradual reduction of expansion
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Final optimization pass with expanded configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Final configuration with spatial constraint tightening
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Final optimization with tighter constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 200, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
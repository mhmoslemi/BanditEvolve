import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Use spatial hash for initial radius estimation with dynamic spatial awareness
    base_radius = 0.38 / cols - np.random.uniform(0.002, 0.01)
    r0 = np.clip(np.array([base_radius] * n + [np.random.uniform(0.001, 0.003)] * 3 + [np.random.uniform(0.002, 0.004)] * 18), 
                 1e-4, 0.5)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure 3*n entries for consistency

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with strict tolerance and closure handling
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using lambda with captured i,j (with fixed i,j bindings)
    overlap_constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # Avoid dynamic closure issues by binding parameters explicitly
            def create_overlap_func(i, j):
                def _func(v):
                    idx1 = 3*i
                    idx2 = 3*j
                    dx = v[idx1] - v[idx2]
                    dy = v[idx1+1] - v[idx2+1]
                    return dx*dx + dy*dy - (v[idx1+2] + v[idx2+2])**2
                return _func
            overlap_constraints.append(create_overlap_func(i, j))
    for c in overlap_constraints:
        cons.append({"type": "ineq", "fun": c})
    
    # Initial optimization with tightened tolerances and increased iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11,
                                             "gtol": 1e-11, "eps": 1e-12})
    
    # Stochastic spatial reconfiguration using gradient-aware hashing (adaptive to current geometry)
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate gradient-aware spatial hashing that perturbs based on local curvature
        spatial_hash_factor = np.sqrt(radii) / np.mean(radii) * 0.03
        spatial_hash = np.random.rand(n, 2) * spatial_hash_factor
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i]**1.1 / np.mean(radii**1.1))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i]**1.1 / np.mean(radii**1.1))
        
        # Re-evaluate with new spatial configuration, using adaptive optimization
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11,
                                                 "gtol": 1e-11, "eps": 1e-12})

    # Targeted radius expansion on least constrained circle with spatial and radii-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance matrix using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (max of min distances to others)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        least_constrained_radius = radii[least_constrained_idx]
        
        # Compute potential for expansion using adaptive gradient and spacing
        # Estimate expansion factor based on spacing to all others, adjusted by current total
        spacing_to_all = np.mean(np.min(dists, axis=1))
        max_possible_radius = spacing_to_all - np.mean(radii)
        expansion_growth = max(0.004, max_possible_radius - np.mean(radii))
        
        # Create expansion vector with targeted growth + spatial-aware perturbation
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_growth * 1.25  # Over-expand slightly
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_growth * (1.0 + 0.15 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        iterations = 0
        while iterations < 6:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between expanded circles
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
                # If overlap, reduce expansion by 3.5% to stabilize
                new_radii = radii + (new_radii - radii) * 0.965
                iterations += 1
        
        # Update decision vector with final expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final re-optimize with refined spatial constraints and increased tolerance
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11,
                                                 "gtol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
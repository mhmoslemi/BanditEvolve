import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))  # 5 columns for 26 circles
    rows = (n + cols - 1) // cols
    
    # Generate spatial constraints using a randomized tiling scheme with non-local perturbations
    # Introduce a non-uniform spatial grid that clusters small circles and creates spatial imbalance
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Apply asymmetric scaling for cluster formation and non-local clustering
        if row < n//3:
            # Cluster small circles in the lower section
            x = x_center + np.random.uniform(-0.1, 0.1)
            y = y_center + np.random.uniform(-0.1, 0.1)
        elif row > 2*n//3:
            # Cluster large circles in the upper section
            x = x_center + np.random.uniform(-0.1, 0.1)
            y = y_center + np.random.uniform(-0.1, 0.1)
        else:
            # Random distribution for middle clusters
            x = x_center + np.random.uniform(-0.08, 0.08)
            y = y_center + np.random.uniform(-0.08, 0.08)
        
        # Stagger alternate rows for non-uniform spacing
        if row % 2 == 1:
            x += 0.5 / cols * 1.1  # Slightly shifted for asymmetric tiling
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius distribution with asymmetric gradient and radius amplification
    # Start with smaller radii in the lower cluster and larger in the upper
    r0 = np.full(n, 0.25 / cols)  # base radii
    r0[0:n//3] = r0[0:n//3] * 0.9  # lower cluster smaller
    r0[n//3:2*n//3] = r0[n//3:2*n//3] * 1.3  # middle cluster larger
    r0[2*n//3:] = r0[2*n//3:] * 1.5  # upper cluster largest
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    # Ensure all bounds have 3*n length
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # [x_low, x_high], [y_low, y_high], [r_low, r_high]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized boundaries with spatial hashing (non-local perturbations)
    # Use closure with lambda and bound indices (capture i)
    cons = []
    for i in range(n):
        # Left bound + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right bound - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom bound + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top bound - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with spatial hashing and constraint tightening
    for i in range(n):
        for j in range(i + 1, n):
            # Add constraint that distance^2 >= (r_i + r_j)^2 with tight tolerance
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + 
                             (v[3*i+1] - v[3*j+1])**2 - 
                             (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization to establish baseline with aggressive iteration and precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-10, "eps": 1e-12})
    
    # Enforce radical non-local reconfiguration with dynamic spatial tiling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create dynamic spatial constraints via randomized tiling
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))  # scale with relative size
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))  # scale with relative size
        
        # Re-evaluate using new spatial perturbation for non-local reconfiguration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted expansion of least constrained circle with dynamic radius amplification
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Efficiently compute all pairwise distances using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distance per circle with vectorized operations
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        least_r = radii[least_constrained_idx]
        
        # Calculate relative expansion based on current sum and potential for growth
        current_sum = radii.sum()
        target_growth = 0.0065
        expansion_factor = target_growth / (n - 1) * (current_sum / radii.mean())  # scale by average
        
        # Create expansion vector with targeted expansion on least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Stochastic and spatial-aware perturbation
                # Favor expansion for circles with smaller min_distances / max_distances
                expansion_i = expansion_factor * (1.0 + 0.15 * np.random.rand()
                                                  * max(0, (min_dists[i] / (n * 0.025) )) )
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation using efficient broadcasted checks
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Fast validation using vectorized constraints
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
                # If invalid, scale back expansion gradually
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Final optimization with refined radii and spatial configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Perform final optimization with constrained expansion
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with optimized randomization for spatial diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with adaptive width to break symmetry
        x_offset = np.random.uniform(-0.075, 0.075)
        y_offset = np.random.uniform(-0.075, 0.075)
        x = x_center + x_offset
        y = y_center + y_offset
        # Stagger alternate rows for spatial diversity
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Set initial radii with more aggressive initial allocation
    # Radius spacing is determined by grid and offset size
    r0 = 0.37 / cols - 1e-3  # slightly higher base radius than previous
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        # Left constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j:
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2
            })

    # Initialize optimization with tighter tolerance and more iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-11})

    # Asymmetric reconfiguration with structured randomization and guided expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate structured randomization pattern for asymmetric perturbation
        # Randomized spatial hashing with gradient adjustment for constrained circles
        # Use radius-based scaling to amplify effect on smaller circles
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            scale = 1.0 + np.log(radii[i] / np.mean(radii)) * 0.5  # radius-aware scaling
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * scale
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})

    # Targeted radius expansion on least constrained circle with soft-constrained adaptation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Efficient distance calculation using broadcasting and vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the circle with the largest minimum distance (least constrained)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate targeted expansion factor with adaptive growth heuristic
        current_total = np.sum(radii)
        target_growth = 0.0075
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create new radii with expansion on most under-constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expand slightly
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic adjustment
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        max_trials = 3
        for trial in range(max_trials):
            # Apply expansion
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between circles
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
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.96
        
        # Update decision vector with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with new configuration and tighter constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
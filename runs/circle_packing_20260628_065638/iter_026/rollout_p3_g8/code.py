import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized grid clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Generate base grid positions with offset for randomness and spacing
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add randomized offset to break symmetry and allow flexible arrangement
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        
        # Stagger alternate rows for more compact packing
        if row % 2 == 1:
            x += 0.5 / cols
        
        # Bound x and y values to ensure they stay within square
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimate
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for decision variables
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Define objective: maximize sum of radii is equivalent to minimizing negative sum
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Define boundary constraints
    cons = []
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Define pairwise circle overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased iterations and precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, 
                                              "eps": 1e-12, "disp": False})
    
    # Apply asymmetric reconfiguration: random geometric hashing
    if res.success:
        v = res.x
        # Introduce controlled randomness to explore new configurations
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, 
                                                  "eps": 1e-12, "disp": False})

    # Targeted radius expansion with spatial-aware adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Efficient vectorized distance computation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate maximum potential expansion based on remaining space
        expansion_factor = 0.008 / (n - 1)  # Controlled to avoid overlaps
        
        # Expand the least constrained circle with soft constraints
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.35  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Minor random perturbations to avoid clustering
                new_radii[i] += expansion_factor * (0.95 + 0.1 * np.random.rand())
        
        # Validate and apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validity check
            valid = True
            # Precompute all pairwise distances
            dx = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            for i in range(n):
                for j in range(i + 1, n):
                    if dists[i, j] < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion to keep within bounds
                new_radii = radii + (new_radii - radii) * 0.97
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Apply final optimization with refined configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, 
                                                  "eps": 1e-12, "disp": False})

    # Final cleanup to ensure robustness
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
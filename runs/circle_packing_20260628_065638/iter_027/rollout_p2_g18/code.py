import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with a hybrid grid + random offset and randomized stagger
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add randomized offset to scatter
        x_off = np.random.uniform(-0.04, 0.04)
        y_off = np.random.uniform(-0.04, 0.04)
        
        # Staggered grid for rows to avoid alignment
        stagger = 0.5 / cols * (row % 2)
        x = x_center + x_off
        y = y_center + y_off + stagger
        
        xs.append(x)
        ys.append(y)
    
    # Base radius based on grid layout with adjustment for 5 columns
    r0 = 0.32 / cols - 0.5 * 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Construct constraints using lambda with closure to avoid lambda capture issues
    cons = []
    for i in range(n):
        # Left edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Construct pairwise distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Use nested lambda to capture fixed i,j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with aggressive parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-8})

    # Apply dynamic spatial hashing and reconfiguration for better diversity
    def apply_spatial_hashing(v):
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed = v.copy()
        for i in range(n):
            perturbed[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        return perturbed
    
    if res.success:
        v = res.x
        # First pass with spatial hashing
        v = apply_spatial_hashing(v)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted expansion on least constrained circle with dynamic constraint adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized pairwise distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Base target total growth and expansion pattern
        current_total = np.sum(radii)
        target_growth = 0.01
        expansion_rate = target_growth / (n - 1)
        
        # Apply expansion with asymmetric pattern
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_rate * 1.3  # Over-expand to trigger layout change
        for i in range(n):
            if i != least_constrained_idx:
                expansion = expansion_rate * (1.0 + 0.2 * np.random.rand())
                new_radii[i] += expansion
        
        # Create expanded vector and validate
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
                # If invalid, reduce expansion
                new_radii = radii + (new_radii - radii) * 0.98

        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final optimization with tighter constraints and validation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Re-apply spatial hashing to trigger new configuration
        v = apply_spatial_hashing(v)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final check
    if res.success:
        v = res.x
    else:
        v = v0

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
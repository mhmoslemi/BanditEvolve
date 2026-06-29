import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with stochastic grid + staggered rows + perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add randomized offset for initial spread
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
        
        # Stagger rows for better distribution
        if row % 2 == 1:
            x += 0.5 / cols
        
        xs.append(x)
        ys.append(y)
    
    # Set initial guess with tighter radius based on spacing
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds with exact length 3*n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to maximize the sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint for left boundary
    def left_boundary(v, i):
        return 1.0 - v[3*i] - v[3*i+2]
    # Constraint for right boundary
    def right_boundary(v, i):
        return v[3*i] - v[3*i+2]
    # Constraint for bottom boundary
    def bottom_boundary(v, i):
        return 1.0 - v[3*i+1] - v[3*i+2]
    # Constraint for top boundary
    def top_boundary(v, i):
        return v[3*i+1] - v[3*i+2]

    # Create boundary constraints with fixed i
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: left_boundary(v, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: right_boundary(v, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: bottom_boundary(v, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: top_boundary(v, i)})
    
    # Create overlap constraints using vectorized distance calculation with stable lambda closures
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with capture to avoid closure issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization run with tight tolerances and moderate max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12, "eps": 1e-10})

    # Asymmetric reconfiguration step with stochastic spatial hashing
    if res.success:
        v = res.x
        # Generate stochastic reconfiguration vector with more control
        spatial_hash = np.random.rand(n, 2) * 0.12
        perturbed_v = v.copy()
        for i in range(n):
            # Apply more aggressive perturbation to non-boundary circles
            if np.abs(v[3*i] - 0.5) < 0.4 and np.abs(v[3*i+1] - 0.5) < 0.4:
                perturbed_v[3*i] += spatial_hash[i, 0] * 1.5
                perturbed_v[3*i+1] += spatial_hash[i, 1] * 1.5
            else:
                perturbed_v[3*i] += spatial_hash[i, 0]
                perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with more aggressive spatial reconfiguration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-10})

    # Targeted radius expansion with isolation metric
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized calculation of minimum distance to each circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dist, axis=1)
        isolated_idx = np.argmax(min_dists)
        
        # Calculate expansion factor with more aggressive target
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.0085  # Larger expansion
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create expansion array ensuring no radius exceedes 0.46
        new_radii = radii.copy()
        new_radii[isolated_idx] = np.clip(radii[isolated_idx] + expansion_factor * 1.2, 1e-4, 0.46)
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] = np.clip(radii[i] + expansion_factor * 0.9, 1e-4, 0.46)
        
        # Apply expansion and check for validity with vectorized validation
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Vectorized validation to find overlapping circles
        dx = expanded_v[0::3][:, np.newaxis] - expanded_v[0::3][np.newaxis, :]
        dy = expanded_v[1::3][:, np.newaxis] - expanded_v[1::3][np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        radii_a = expanded_v[2::3][:, np.newaxis]
        radii_b = expanded_v[2::3][np.newaxis, :]
        overlaps = np.where(dists < (radii_a + radii_b - 1e-12), True, False)

        # Find overlapping circles
        overlapping_indices = np.where(overlaps.any(axis=1))[0]
        if len(overlapping_indices) > 0:
            # If overlaps, reduce the expansion factor proportionally
            expansion_factor *= 0.8
            new_radii = radii.copy()
            new_radii[isolated_idx] = np.clip(radii[isolated_idx] + expansion_factor * 1.2, 1e-4, 0.46)
            for i in range(n):
                if i != isolated_idx:
                    new_radii[i] = np.clip(radii[i] + expansion_factor * 0.9, 1e-4, 0.46)
            expanded_v[2::3] = new_radii
        
        # Apply expanded configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-10})

    # Final cleanup and validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
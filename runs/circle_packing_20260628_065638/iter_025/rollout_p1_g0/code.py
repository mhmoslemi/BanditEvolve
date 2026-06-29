import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized staggered grid + enhanced randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Apply double-layered random distortion with exponential decay
        x_offset = np.random.exponential(scale=0.05) * np.random.choice([-1, 1])
        y_offset = np.random.exponential(scale=0.03) * np.random.choice([-1, 1])
        
        # Alternate row shifting for staggered grid
        if row % 2 == 1:
            x_center += 0.5 / cols
        
        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with optimized lambda parameter passing
    cons = []
    for i in range(n):
        # Left bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing and optimized lambda
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Apply stochastic geometric hashing reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash for spatial redistribution
        hash_angles = np.random.uniform(0, 2 * np.pi, n)
        hash_distances = np.random.uniform(0.01, 0.05, n)
        hash_dirs = np.stack([np.cos(hash_angles), np.sin(hash_angles)], axis=1)
        
        # Apply reconfiguration to centers
        new_v = v.copy()
        for i in range(n):
            dx = hash_dirs[i, 0] * hash_distances[i]
            dy = hash_dirs[i, 1] * hash_distances[i]
            new_v[3*i] += dx
            new_v[3*i+1] += dy
        
        # Re-evaluate configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    # Targeted expansion of least constrained circle with adaptive validation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Efficient distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Apply controlled expansion with validation loop
        expansion_total = 0.005  # Target expansion for total sum
        expansion_factor = expansion_total / (n - 1)
        
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate non-overlapping
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (new_radii[i] + new_radii[j]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            # Check boundary conditions
            if valid:
                for i in range(n):
                    if expanded_v[3*i] - expanded_v[3*i+2] < -1e-12:
                        valid = False
                        break
                    if expanded_v[3*i] + expanded_v[3*i+2] > 1.0 + 1e-12:
                        valid = False
                        break
                    if expanded_v[3*i+1] - expanded_v[3*i+2] < -1e-12:
                        valid = False
                        break
                    if expanded_v[3*i+1] + expanded_v[3*i+2] > 1.0 + 1e-12:
                        valid = False
                        break
            
            if valid:
                break
            else:
                # Gradual scale back of expansion
                new_radii = radii + (new_radii - radii) * 0.95

        # Final optimization with enhanced constraints
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
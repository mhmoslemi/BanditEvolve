import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized staggered grid and geometric density control
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base position calculation with row staggering
        x_center_base = (col + 0.5) / cols
        y_center_base = (row + 0.5) / rows
        
        # Add controlled random spatial jitter for geometric diversity
        x_jitter = np.random.uniform(-0.03, 0.03)
        y_jitter = np.random.uniform(-0.03, 0.03)
        
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x_center_base += 0.5 / cols
        
        # Final center coordinates
        x = x_center_base + x_jitter
        y = y_center_base + y_jitter
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with density-dependent scaling
    r0 = 0.33 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Build constraints with closed-loop lambda capture
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})  # Left constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})  # Right constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})  # Bottom constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})  # Top constraint
    
    # Vectorized overlap constraints with geometric sensitivity
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                # Add radius-dependent spatial sensitivity for edge cases
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2 * (1.0 + 0.01 * np.cos(np.sqrt(dx*dx + dy*dy)))
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-8})

    # Shake heuristic for escaping local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Evaluate current configuration's overlap
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find minimum distance for each circle
        min_distances = np.min(dists, axis=1)
        # Find smallest circle with maximum minimum distance
        least_constrained_idx = np.argmax(min_distances)
        smallest_radius = radii[least_constrained_idx]
        
        # Apply controlled perturbation to the least constrained circle
        perturbation = np.random.rand(2) * 0.05
        new_x = centers[least_constrained_idx, 0] + perturbation[0]
        new_y = centers[least_constrained_idx, 1] + perturbation[1]
        new_r = smallest_radius
        
        # Create a new perturbed configuration with the modified circle
        v_new = v.copy()
        v_new[3*least_constrained_idx] = new_x
        v_new[3*least_constrained_idx + 1] = new_y
        
        # Re-evaluate perturbed version
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-8})
    
    # Additional targeted radius expansion with geometric awareness
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute all pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle with geometric growth potential
        min_dist_per_circle = np.min(dists, axis=1)
        total_distance = np.sum(dists[np.triu_indices(n, k=1)])  # Only upper triangle
        growth_potential = (total_distance / n) - np.sum(radii)
        
        # Calculate growth based on total potential and geometric expansion factor
        growth_factor = 1.2  # Control amount of expansion
        expansion_step = growth_factor * (0.001 + growth_potential * 0.001)
        
        # Distribute expansion across all circles while maintaining feasibility
        for _ in range(2):  # Two rounds of expansion to allow deeper optimization
            new_radii = radii.copy()
            for i in range(n):
                # Add proportional expansion based on available space
                available_space = (np.min(dists[i, (i+1):]) - radii[i]) / (1 + np.sum(radii[i]))
                new_radii[i] += expansion_step * (1 + 0.1 * np.random.rand()) * available_space
                
            # Check feasibility
            feasible = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                v = v.copy()
                v[2::3] = new_radii
                radii = new_radii
                centers = np.column_stack([v[0::3], v[1::3]])
                res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-9})
            else:
                break
    
    # Final check and optimization
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
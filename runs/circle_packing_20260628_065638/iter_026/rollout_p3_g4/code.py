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
        # Randomized offset with increased variance to break symmetry
        x = x_center + np.random.uniform(-0.12, 0.12)
        y = y_center + np.random.uniform(-0.12, 0.12)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols * 1.1
        xs.append(x)
        ys.append(y)
    
    r0 = 0.45 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for position and radius
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective: minimize negative sum of radii to maximize total
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint functions for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Constraint functions for circle-circle distance
    for i in range(n):
        for j in range(i+1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization run with high precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-12})
    
    # Trigger asymmetric reconfiguration using spatial hashing
    if res.success:
        v = res.x
        spatial_hash = np.random.rand(n, 2) * 0.12
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})

    # Targeted radius expansion with more refined spatial-aware adjustment
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute distances using vectorized broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle via spatial diversity and constraint strength
        min_dists = np.min(dists, axis=1)
        # Weighted constraint strength for circle selection (lower min distance implies lower constraint)
        constraint_strength = np.sum(dists, axis=1)
        weighted_dists = min_dists / (constraint_strength + 1e-8)
        least_constrained_idx = np.argmax(weighted_dists)
        
        # Calculate more aggressive expansion factor with spatial awareness
        total_sum = np.sum(radii)
        expansion_factor = 0.0085 / (n - 1)  # Controlled expansion
            
        # Create expansion vector with soft enforcement
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.35  # Slight over-expansion
        for i in range(n):
            # Add small random perturbation to nearby circles
            if i != least_constrained_idx:
                # Apply exponential decay on impact of expansion
                decay = np.exp(-np.abs(i - least_constrained_idx) / 3)
                expansion_i = expansion_factor * decay * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation and adaptive tolerance
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration with tighter tolerance
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
                # If invalid, adjust expansion by a percentage (more aggressive than parent)
                new_radii = radii + (new_radii - radii) * 0.93
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration with tighter tolerance
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
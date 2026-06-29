import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering, staggered grid, and directional bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Base randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Add directional bias for alternating rows (staggered layout)
        if row % 2 == 1:
            x += 0.5 / cols * (0.9 + np.random.rand() * 0.2)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda-capturing
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

    # Vectorized overlap constraints with advanced geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First-phase optimization with tight tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-10})
    
    # Trigger asymmetric reconfiguration with geometric hashing
    if res.success:
        v = res.x
        # Compute current configuration's spatial constraints
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Determine constraint slack for every circle
        slack = np.zeros(n)
        for i in range(n):
            slack[i] = np.min(dists[i, i+1:]) - np.sum([v[3*i+2] + v[3+j+2] for j in range(i+1, n)]) + 0.005
        
        # Identify the circle with the most constraint slack
        max_slack_idx = np.argmax(slack)

        # Create geometric hash for spatial reconfiguration
        hash_map = np.random.rand(n, 2) * 0.10
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0] * (1.0 if i == max_slack_idx else 0.5)
            perturbed_v[3*i+1] += hash_map[i, 1] * (1.0 if i == max_slack_idx else 0.5)

        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-11})
    
    # Targeted expansion on most under-constrained circle with directional bias
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        dists = np.sqrt(((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2).sum(axis=2))
        min_dists = np.min(dists, axis=1)
        most_under_constrained_idx = np.argmax(min_dists)
        
        # Calculate current minimal constraint margin for the most under-constrained
        target_radius_idx = most_under_constrained_idx
        other_radii = np.delete(radii, target_radius_idx)
        sum_other_radii = np.sum(other_radii)
        target_radius = radii[target_radius_idx] + 0.0045
        new_total_sum = sum_other_radii + target_radius
        expansion_factor = (new_total_sum - np.sum(radii)) / (n - 1)
        
        # Apply controlled expansion with directional bias
        new_radii = radii.copy()
        new_radii[target_radius_idx] += expansion_factor * 1.15
        for i in range(n):
            if i != target_radius_idx:
                expansion = expansion_factor * (1.0 + 0.2 * np.random.rand())
                new_radii[i] += expansion
        
        # Validate updated radii and re-evaluate
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
        
        # Gradient approximation and warm-start for faster convergence
        def expanded_neg_sum_radii(v):
            return -np.sum(v[2::3])

        res = minimize(expanded_neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12, "eps": 1e-11})
        
        # Final refinement with spatial bias reinforcement
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            
            # Additional spatial bias for most under-constrained
            if most_under_constrained_idx != -1:
                for i in range(n):
                    v[3*i] += np.random.uniform(-0.02, 0.02)
                    v[3*i+1] += np.random.uniform(-0.02, 0.02)
                res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 100, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
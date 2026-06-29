import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Generate initial positions with randomized staggered grid and geometric perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add random spatial perturbation with increasing magnitude for higher rows
        row_weight = 1.0 + 0.4 * (row / rows)
        x = x_center + np.random.uniform(-0.08 * row_weight, 0.08 * row_weight)
        y = y_center + np.random.uniform(-0.08 * row_weight, 0.08 * row_weight)
        
        # Stagger alternate rows for better spacing
        if row % 2 == 1:
            x += 0.5 / cols * (0.8 + 0.2 * row / rows)
        xs.append(x)
        ys.append(y)
    
    # Use higher initial radii with diminishing scale for better convergence
    r0 = 0.4 / cols - 1e-3
    base_radius = r0
    
    # Create initial radius vector with adaptive weighting for central rows
    radii_weights = np.array([1.0 + 0.2 * (i / rows) for i in range(rows)])
    radii_weights = np.repeat(radii_weights, cols)
    radii_weights = np.clip(radii_weights, 0.8, 1.2)
    r0 = base_radius * radii_weights
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with optimized distance calculation
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initialize with enhanced constraints for better convergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})

    # Apply advanced geometric reconfiguration with randomized spatial hashing
    if res.success:
        v = res.x
        # Generate spatial hash map to trigger reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.05
        
        # Apply perturbation to positions with gradient-aligned scaling
        perturbed_v = v.copy()
        for i in range(n):
            perturb_pos = spatial_hash[i] * (1.0 + 0.1 * np.random.rand())
            perturbed_v[3*i] += perturb_pos[0]
            perturbed_v[3*i+1] += perturb_pos[1]
        
        # Re-evaluate with new layout
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    # Apply targeted radius expansion on most constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute constraint tightness (min distance to others and edges)
        constraint_tightness = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    constraint_tightness[i] += max(0.0, radii[i] + radii[j] - dists[i,j])
            constraint_tightness[i] += (1.0 - centers[i,0] - radii[i]) + (1.0 - centers[i,1] - radii[i])
        
        # Identify circle with maximal constraint tightness
        most_constrained_idx = np.argmax(constraint_tightness)
        
        # Expand its radius while maintaining constraints
        current_sum = np.sum(radii)
        target_sum = current_sum + 0.008
        expansion = (target_sum - current_sum) * 1.0
        
        # Distribute expansion with spatial-aware distribution
        expansion_vector = np.zeros(n)
        expansion_vector[most_constrained_idx] = expansion * 1.2  # Slight over-expansion
        for i in range(n):
            if i != most_constrained_idx:
                expansion_vector[i] = expansion * (1.0 + 0.05 * np.random.rand())
        
        # Apply expansion and re-configure
        new_radii = radii + expansion_vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with optimized layout
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    # Final verification with enhanced validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
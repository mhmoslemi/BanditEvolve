import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initial layout with geometric hashing
    xs = []
    ys = []
    for i in range(n):
        # Base grid position
        base_col = i % cols
        base_row = i // cols
        x_center = (base_col + 0.5) / cols
        y_center = (base_row + 0.5) / rows
        
        # Add non-local geometric hashing
        hash_offset = np.random.uniform(-0.15, 0.15, 2)
        x = x_center + hash_offset[0]
        y = y_center + hash_offset[1]
        # Create staggered rows
        if base_row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radii
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

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11})
    
    # Topological reconfiguration with geometric hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Geometric hashing for non-local reconfiguration
        hash_radius = 0.02
        hash_angles = np.random.uniform(0, 2 * np.pi, n)
        hash_offsets = np.random.uniform(-hash_radius, hash_radius, (n, 2))
        
        # Create new spatial configuration
        new_v = v.copy()
        for i in range(n):
            cx = np.cos(hash_angles[i]) * hash_offsets[i, 0]
            cy = np.sin(hash_angles[i]) * hash_offsets[i, 1]
            new_v[3*i] += cx
            new_v[3*i+1] += cy
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Identify least constrained circle with minimum inter-circle distances
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find most under-constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        min_dist = min_dists[least_constrained_idx]
        min_radius = radii[least_constrained_idx]
        total_radius_sum = np.sum(radii)
        
        # Target controlled expansion factor with safety margin
        expansion_factor = 0.003  # 0.3% of current sum
        new_total_radius_sum = total_radius_sum + expansion_factor
        expansion_per_circle = expansion_factor / (n - 1)
        
        # Create expansion vector with careful enforcement
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_per_circle * 1.5  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_per_circle
        
        # Validate new configuration
        while True:
            temp_v = v.copy()
            temp_v[2::3] = new_radii
            temp_centers = np.column_stack([temp_v[0::3], temp_v[1::3]])
            
            # Check all pairs for overlap
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = temp_centers[i, 0] - temp_centers[j, 0]
                    dy = temp_centers[i, 1] - temp_centers[j, 1]
                    dist = np.hypot(dx, dy)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            # If valid, break, else reduce expansion slightly
            if valid:
                break
            else:
                # Reduce expansion by 10%
                new_radii = radii + (new_radii - radii) * 0.9
                # Force minimum radius threshold
                new_radii = np.maximum(new_radii, 1e-6)
        
        # Apply expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with tight tolerances
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
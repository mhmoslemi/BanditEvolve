import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        
        # Base grid calculation
        base_x = (col_idx + 0.5) / cols
        base_y = (row_idx + 0.5) / rows
        
        # Add randomized noise
        noise_x = np.random.uniform(-0.05, 0.05)
        noise_y = np.random.uniform(-0.05, 0.05)
        
        # Add row alternation for stagger
        stagger = 0.0 if (row_idx % 2 == 0) else 0.5 / cols
        
        # Final position
        x = base_x + noise_x + stagger
        y = base_y + noise_y
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-4
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
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "maxls": 100})
    
    # Spatial hashing reconfiguration
    if res.success:
        v = res.x.copy()
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hash for spatial transformation
        hash_factor = 0.08
        hash_offset = 0.05
        spatial_hash = np.random.rand(n, 2) * hash_factor + hash_offset
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Reoptimize with perturbed positions
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "maxls": 100})

    # Targeted radius expansion with spatial hashing
    if res.success:
        v = res.x.copy()
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances using vectorization for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with least constraint (least nearest neighbor distance)
        min_interaction = np.min(dists, axis=1)
        isolated_idx = np.argmin(min_interaction)
        
        # Apply expansion with spatial re-optimization
        expansion_base = 0.0035
        expansion_factor = expansion_base * (1.0 + 0.1 * np.random.rand())
        
        # Create new radii array with expansion applied
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_factor * 1.2  # Slight over-expansion
        
        # Spatially reconfigure to promote interaction with expanded circle
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())
        
        # Update decision vector
        v_expanded = v.copy()
        v_expanded[2::3] = new_radii
        
        # Reoptimize with new configuration
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "maxls": 100})
        
        # Final validation pass
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            
            # Validate against boundary constraints
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if (x - r < -1e-12 or x + r > 1 + 1e-12
                    or y - r < -1e-12 or y + r > 1 + 1e-12):
                    radii = np.clip(radii, 1e-6, None)
                    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                                   constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
                    v = res.x
                    centers = np.column_stack([v[0::3], v[1::3]])
        
        # Final radius clipping
        v = res.x
        radii = np.clip(v[2::3], 1e-6, None)
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
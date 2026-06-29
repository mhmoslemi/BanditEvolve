import numpy as np

def run_packing():
    n = 26
    
    # Optimal tiling with adaptive spatial hashing: 5x6 grid + flexible row count
    cols = 5  # Primary grid column count (higher for better spacing)
    rows = np.ceil(n / cols).astype(int) if cols !=0 else n  # Rows adapt to cols
    
    # 1. Initialize with adaptive randomized cluster centers and spatial constraints
    # Use geometric hashing to break symmetry and seed diversity in spatial distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid calculation with adaptive spacing
        x_center = 0.5 + col * (1.0 / cols)  # Base x-center in 0.0 to 1.0
        
        # Adaptive y-assignment: rows with fewer elements are scaled
        row_factor = 1.0 if rows == 1 else ((rows - row - 1) / (rows - 1))
        
        if rows > 1:
            y_center = 0.5 + row * (1.0 / rows) * row_factor
        else:
            y_center = 0.5
        # Add randomized offset to avoid clustering
        x_offset = np.random.uniform(-0.04, 0.04)  # Small range to prevent clustering
        y_offset = np.random.uniform(-0.04, 0.04)  # Small range to prevent clustering
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Add spatial perturbation based on row density
        row_density = ((rows - row - 1) / (rows - 1)) if rows > 1 else 1.0
        perturb = np.random.normal(0, 0.02 * row_density, size=2)
        x += perturb[0]
        y += perturb[1]
        xs.append(x)
        ys.append(y)
    
    # 2. Initialize with radius based on grid dimensions + spatial hashing factor
    r0 = 0.35 / cols - 1e-3  # Initial guess based on 5-column spacing
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)  # Initial radius set

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3*n ensures proper alignment

    # Objective: maximize sum of radii (minimize negative)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint definitions: optimized for vectorization, avoid lambda capture issues
    cons = []

    # Apply spatial boundary constraints for all circles
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2] })
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] })
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2] })
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] })

    # Implement spatial hashing + dynamic constraint generation for overlaps
    # For overlapping circles, we add spatial constraints that enforce at least a minimum gap
    for i in range(n):
        for j in range(i + 1, n):
            # Use a dynamic spacing constraint: distance^2 - (r1 + r2)^2 >= min_gap^2
            # To avoid numerical instability in small radii, min_gap is 1.5e-5
            min_gap_sq = (1.5e-5)**2  # Slight buffer to avoid numerical precision issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2 - min_gap_sq
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # 3. Optimization with multi-stage, dynamic convergence
    # Initial optimization (medium steps, high tolerance)
    res = minimize(neg_sum_radii, v0, method="SLSQP",
                   bounds=bounds,
                   constraints=cons,
                   options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-10})
    
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
        
        # 4. Spatial hashing: reconfigure to break symmetries
        hash_map = np.random.rand(n, 2) * 0.02  # Tiny but impactful perturbation
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0] * (radii[i]/np.mean(radii))
            perturbed_v[3*i+1] += hash_map[i, 1] * (radii[i]/np.mean(radii))
        # Re-optimization with perturbed centers and dynamic constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", 
                       bounds=bounds,
                       constraints=cons, 
                       options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})
    
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
        dists = np.zeros((n, n))
        
        # 5. Compute spatial constraints for reconfiguration (vectorized)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        
        # 6. Find the circle with the least geometric constraint (most space available)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        constraint_strength = 1.0 - (min_dists[least_constrained_idx] - 1e-7) / (np.max(min_dists) - 1e-7)
        
        # 7. Targeted radius expansion using adaptive growth model
        # Grow the least constrained radius by 0.0015 with gradient control
        # Apply a radial growth to neighbors (soft influence) 
        # to propagate expansion without causing overlap
        expansion_factor = 0.0015 * (0.8 + 0.2 * constraint_strength)
        
        # 8. Create expansion vector with adaptive expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Boost expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Use a softmax-like expansion where more space means more growth
                expansion_i = expansion_factor * (1.0 + 0.2 * np.random.rand()) * (min_dists[i] / np.mean(min_dists))
                new_radii[i] += expansion_i
        
        # 9. Validate with gradient-based constraint checking to avoid overlap
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            
            # Check overlap with gradient checking (avoiding exact distance computation)
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_sq = dx*dx + dy*dy
                    radius_sum = new_radii[i] + new_radii[j]
                    if dist_sq < radius_sum**2 - 1e-12:
                        valid = False  # Just need one to fail
                        break
                if not valid:
                    break
            
            if valid:
                # Apply expansion
                v = expanded_v
                break
            else:
                # Gradual scaling back of expansion
                new_radii = radii + (new_radii - radii) * 0.98
        
        # 10. Final optimization with expanded radii
        res = minimize(neg_sum_radii, v, method="SLSQP",
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 450, "ftol": 1e-11, "eps": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
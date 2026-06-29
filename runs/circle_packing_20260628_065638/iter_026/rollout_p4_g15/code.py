import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with a randomized geometric grid with hashing
    xs = []
    ys = []
    hash_seed = np.random.rand(n)
    
    for i in range(n):
        col = i % cols
        row = i // cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Apply hash-based perturbation for randomized spatial configuration
        jitter_x = np.sin(hash_seed[i] * 10) * 0.06
        jitter_y = np.cos(hash_seed[i] * 13) * 0.06
        
        # Stagger alternate rows
        if row % 2 == 1:
            base_x += 0.5 / cols
        
        x = base_x + jitter_x
        y = base_y + jitter_y
        
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

    # Create vectorized constraint functions
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "maxls": 100})

    # Apply geometric hashing transformation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Random spatial hash with sine/cosine for more natural perturbation
        spatial_hash = np.random.rand(n, 2) - 0.5
        spatial_hash *= np.array([0.05, 0.05])
        perturbed_v = v.copy()
        
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "maxls": 100})
    
    # Targeted radius expansion on smallest circle while enforcing strict non-overlap
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify smallest circle
        smallest_idx = np.argmin(radii)
        min_dist = np.min(dists[smallest_idx])
        
        # Calculate expansion factor with safety margin
        max_allowed_radius = 0.5 - min(0.05, np.min(centers[:, 0] - 0.5, np.min(centers[:, 0] + 0.5, 
                                                                          np.min(centers[:, 1] - 0.5, 
                                                                                 np.min(centers[:, 1] + 0.5)))) - 0.01)
        if radii[smallest_idx] < max_allowed_radius:
            expansion_factor = (max_allowed_radius - radii[smallest_idx]) * 1.4
        else:
            expansion_factor = 0
        
        # Apply expansion to smallest radius and others with some randomness
        expansion = expansion_factor * (1.0 + 0.2 * np.random.rand())
        new_radii = radii.copy()
        new_radii[smallest_idx] += expansion
        
        # Ensure all other circles get some minimal expansion
        for i in range(n):
            if i != smallest_idx:
                new_radii[i] += expansion * 0.75 * (1.0 + 0.05 * np.random.rand())
        
        # Re-calculate distances to ensure no overlaps
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                # Reduce expansion slightly if invalid
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Apply the expanded configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with the new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "maxls": 100})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
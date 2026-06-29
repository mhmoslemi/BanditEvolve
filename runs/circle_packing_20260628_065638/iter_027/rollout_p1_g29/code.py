import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    total_area = 1.0

    # Initialize with geometrically informed seed positions and spatial hashing for diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce non-local geometric hashing for more diverse configurations
        # We generate a random hash for each circle to seed its position
        hash_offset = np.random.rand(2) * 0.2
        x = x_center + hash_offset[0]
        y = y_center + hash_offset[1]
        # Stagger alternating rows to prevent grid-like clustering
        if row % 2 == 1:
            x += 0.25 / cols  # Scaled based on grid column size
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with density-aware estimation based on grid spacing, allowing for spatial compression
    r0 = 0.36 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Build constraints explicitly with vectorized handling and closure binding to ensure correct execution
    cons = []
    for i in range(n):
        # Left wall constraint: x_i - r_i >= 0
        def left_bound(v, i=i):
            return v[3*i] - v[3*i+2]
        cons.append({"type": "ineq", "fun": left_bound})
        # Right wall constraint: 1.0 - x_i - r_i >= 0
        def right_bound(v, i=i):
            return 1.0 - v[3*i] - v[3*i+2]
        cons.append({"type": "ineq", "fun": right_bound})
        # Bottom wall constraint: y_i - r_i >= 0
        def bottom_bound(v, i=i):
            return v[3*i+1] - v[3*i+2]
        cons.append({"type": "ineq", "fun": bottom_bound})
        # Top wall constraint: 1.0 - y_i - r_i >= 0
        def top_bound(v, i=i):
            return 1.0 - v[3*i+1] - v[3*i+2]
        cons.append({"type": "ineq", "fun": top_bound})
    
    # Vectorize overlap constraints with closure-binding to ensure correct v indexing
    for i in range(n):
        for j in range(i+1, n):
            def dist_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": dist_func})

    # First optimization run with increased max iterations and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-12})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Radical spatial reconfiguration through asymmetric geometric rehashing
        # Generate a new spatial hash with adaptive scaling based on current density
        spatial_hash = np.random.rand(n, 2) * 0.15
        # Apply spatial hash scaling based on radius diversity
        radius_ratio = np.std(radii) / np.mean(radii)
        perturb_factor = 0.5 * radius_ratio
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_hash[i, 0] * perturb_factor
            new_v[3*i+1] += spatial_hash[i, 1] * perturb_factor
        
        # Re-evaluate with spatial reconfiguration and tighter tolerances
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})
        
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Targeted expansion with dynamic radius allocation and enhanced topology awareness
            # Calculate distance matrix for all pairs in a vectorized way
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)  # Identify most underutilized circle

            # Compute expansion potential using spatial dynamics
            current_total = np.sum(radii)
            target_total = current_total * 1.007  # 0.7% total expansion goal
            expansion_factor = (target_total - current_total) / (n - 1)
            
            # Stochastic expansion with topology-aware growth and radius-based perturbation
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.15  # slight over-expansion
            for i in range(n):
                if i != least_constrained_idx:
                    new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())
            
            # Apply expansion while maintaining constraint validity
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
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
                    # If invalid, decrease expansion slightly based on constraint proximity
                    # Use a more subtle decay to preserve valid configurations
                    new_radii = radii + (new_radii - radii) * 0.98

            # Final optimization with enhanced parameters to stabilize the new configuration
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
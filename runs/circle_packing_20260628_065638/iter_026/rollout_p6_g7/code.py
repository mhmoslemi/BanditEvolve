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
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.3 / cols
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

    cons = []
    for i in range(n):
        # Left constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing and constraint caching
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r1 = v[3*i+2]
                r2 = v[3*j+2]
                return dist_sq - (r1 + r2)**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-11})

    # Disruptive geometric transformation: randomized spatial hashing + targeted expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix for all pairwise circles
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Compute adjacency matrix based on current configuration
        adj = dists <= (radii + radii.reshape(-1, 1))
        
        # Apply geometric hashing for space reordering
        from scipy.sparse import csr_matrix, csgraph
        graph = csr_matrix(adj)
        components = csgraph.connected_components(graph)[1]
        component_hash = np.random.rand(n, 2) * 0.06
        
        # Apply stochastic reordering by component
        for i in range(n):
            v[3*i] += component_hash[components[i], 0] * 1.0
            v[3*i+1] += component_hash[components[i], 1] * 1.0
        
        # Re-optimize with new spatial configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
        
        # Targeted radius expansion on circle with minimal radius but maximal spatial potential
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 + 
                            (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
            min_dists = np.min(dists, axis=1)
            min_radius_idx = np.argmin(radii)
            min_dist_idx = np.argmin(min_dists)
            
            # Apply spatial expansion bias towards least constrained circle
            if min_radius_idx == min_dist_idx:
                # Select the radius with smallest value but not zero
                non_zero_radii = radii[radii > 1e-8]
                smallest_nonzero_idx = np.argmin(non_zero_radii)
                min_radius_idx = smallest_nonzero_idx
                
            # Compute expansion target
            current_total = np.sum(radii)
            target_total = current_total + 0.0090
            expansion_factor = max((target_total - current_total) / (n - 1), 0.0001)
            
            # Distribute expansion with spatial awareness
            new_radii = radii.copy()
            new_radii[min_radius_idx] += expansion_factor * 1.3
            for i in range(n):
                if i != min_radius_idx:
                    new_radii[i] += expansion_factor
            
            # Apply expansion with constraint validation
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                valid = True
                
                for i in range(n):
                    for j in range(i+1, n):
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
                    # If invalid, lower expansion slightly
                    expansion_factor *= 0.95
                    new_radii = radii + (new_radii - radii) * expansion_factor / (target_total - current_total)
            
            # Update decision vector
            v = expanded_v
        
        # Final optimization pass with tighter tolerances
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a hybrid of randomized geometric clustering and 
    # randomized lattice structure to avoid symmetry and enable reconfiguration
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Shift odd rows for staggered grid
        if row % 2 == 1:
            x += 0.5 / cols * (np.random.rand() - 0.5)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3*n, matches v

    def neg_sum_radii(v):
        # Optimization objective: maximize sum of radii
        return -np.sum(v[2::3])

    # Vectorized constraint definitions
    cons = []
    for i in range(n):
        # Left side boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right side boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom side boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top side boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using vectorized operations
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                # Use numpy vectorized operations to compute squared distance
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization phase
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Targeted spatial transformation with geometric hashing and reordering
    if res.success:
        v = res.x
        # Store initial radii and apply geometric hashing
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Use geometric hashing for spatial reconfiguration
        # Compute component-based spatial hash for reordering
        from scipy.sparse import csr_matrix, csgraph
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        adj = dists <= (radii + radii.reshape(-1, 1))
        graph = csr_matrix(adj)
        components = csgraph.connected_components(graph)[1]
        
        # Create a spatial hash to trigger geometric transformation
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[components[i], 0]
            perturbed_v[3*i+1] += spatial_hash[components[i], 1]
        
        # Reoptimize with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on smallest radius circle with adjacency awareness
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute pairwise distances using vectorized operations
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the circle with smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        # Compute total current sum
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Adjust radii with soft spatial constraints to avoid over-expansion
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * 1.2
        
        # Apply expansion with adjacency-aware constraint enforcement
        for i in range(n):
            if i != smallest_radius_idx:
                # Add a small random perturbation to avoid local minima
                expansion_i = expansion_factor * (1.0 + 0.05 * np.random.randn())
                new_radii[i] += expansion_i
        
        # Apply expansion and re-verify
        while True:
            # Check feasibility before applying
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    if np.sqrt(dx**2 + dy**2) < new_radii[i] + new_radii[j] - 1e-12:
                        new_radii = radii + (new_radii - radii) * 0.95
                        break
                else:
                    continue
                break
            
            # If valid, accept new radii
            if all(np.sqrt((centers[i] - centers[j])**2) >= new_radii[i] + new_radii[j] - 1e-11 
                   for i in range(n) for j in range(i + 1, n)):
                break
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Perform final re-optimization with adjusted constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
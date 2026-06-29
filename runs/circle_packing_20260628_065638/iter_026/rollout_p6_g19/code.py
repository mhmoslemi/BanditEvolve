import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with a hybrid grid + random spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Base positions with grid-aligned layout
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows for staggered layout
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Base radii: optimize for dense packing and allow expansion
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
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply geometric hashing and spatial transformation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute adjacency matrix for all circle pairs
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        adj = dists <= (radii + radii.reshape(-1, 1))
        
        # Use spatial hashing to break symmetry and trigger layout changes
        from scipy.sparse import csr_matrix, csgraph
        graph = csr_matrix(adj)
        components = csgraph.connected_components(graph)[1]
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[components[i], 0] * 0.25
            perturbed_v[3*i+1] += spatial_hash[components[i], 1] * 0.25
        
        # Re-evaluate with perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Trigger disruptive transformation on the circle with the smallest radius
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distances between all circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify the circle with the smallest radius (least constrained)
        smallest_radius_idx = np.argmin(radii)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Apply asymmetric expansion on smallest radius circle
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.007
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Adjust radii using directional expansion to break symmetry
        new_radii = radii.copy()
        # Overexpand smallest radius to force layout reconfiguration
        new_radii[smallest_radius_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != smallest_radius_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.2 * np.random.rand())
        
        # Update decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    # Final cleanup and optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distances for final validation
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Ensure all circles meet boundary constraints with tight tolerances
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            if not (0.0 + 1e-9 <= x - r and x + r <= 1.0 - 1e-9
                    and 0.0 + 1e-9 <= y - r and y + r <= 1.0 - 1e-9):
                v[3*i] = np.clip(x, 1e-9, 1.0 - 1e-9)
                v[3*i+1] = np.clip(y, 1e-9, 1.0 - 1e-9)
                v[3*i+2] = np.clip(r, 1e-6, 0.5)
        
        # Final optimization with constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
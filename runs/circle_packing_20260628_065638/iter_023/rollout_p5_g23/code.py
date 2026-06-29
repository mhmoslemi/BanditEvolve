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
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
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

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Asymmetric reconfiguration: apply randomized geometric hashing with adjacency-aware perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        # Compute distance matrix
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Compute adjacency matrix based on current configuration
        adj = dists <= (radii + radii.reshape(-1, 1))
        # Compute graph components to enable topological reordering
        from scipy.sparse import csr_matrix, csgraph
        graph = csr_matrix(adj)
        components = csgraph.connected_components(graph)[1]
        # Apply random geometric hashing to each component
        component_hash = np.random.rand(n, 2) * 0.06
        for i in range(n):
            v[3*i] += component_hash[components[i], 0] * (1.0 - radii[i] / 0.4)
            v[3*i+1] += component_hash[components[i], 1] * (1.0 - radii[i] / 0.4)
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Targeted radius expansion on smallest non-zero radius with global topology reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        # Compute distance matrix
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Compute adjacency matrix based on current configuration
        adj = dists <= (radii + radii.reshape(-1, 1))
        # Compute graph components to enable topology reordering
        from scipy.sparse import csr_matrix, csgraph
        graph = csr_matrix(adj)
        components = csgraph.connected_components(graph)[1]
        # Identify the circle with smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        # Apply controlled radius expansion based on topological position
        total_sum = np.sum(radii)
        expansion_factor = 0.007 / (n - 1) * (1.0 + 1e-3 * np.random.rand())
        
        # Create adjusted radius vector
        new_radii = radii.copy()
        # Expand smallest radius more to trigger topological change
        new_radii[smallest_radius_idx] += expansion_factor * 2.0
        for i in range(n):
            if i != smallest_radius_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new adjacency constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
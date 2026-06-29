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
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Apply geometric dissection on the three most constrained circles
    if res.success:
        v = res.x
        # Compute distance matrix for all pairwise circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Compute adjacency matrix for all circle pairs
        adj = dists <= (v[2::3] + v[2::3].reshape(-1, 1))
        # Compute constraint severity for each circle
        constraint_severity = np.sum(adj, axis=1)
        # Identify three most constrained circles
        most_constrained_indices = np.argsort(constraint_severity)[::-1][:3]
        # Isolate these three and reconfigure their spatial relationships
        # Temporarily reposition them to break local minima
        temp_v = v.copy()
        for idx in most_constrained_indices:
            # Apply strategic repositioning to unlock new configuration
            temp_v[3*idx] += np.random.uniform(-0.04, 0.04)
            temp_v[3*idx+1] += np.random.uniform(-0.04, 0.04)
            temp_v[3*idx+2] += np.random.uniform(-0.003, 0.003)
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, temp_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Gradual radius expansion on the least constrained circle while preserving topology
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        # Calculate expansion factor to increase the least constrained circle's radius
        total_sum = np.sum(radii)
        expansion_factor = 0.007 / (n - 1)  # Controlled expansion to unlock new configuration
        
        # Adjust radii to increase least constrained circle's radius
        new_radii = radii.copy()
        # Gradual expansion with adjacency-aware topology preservation
        for i in range(n):
            if i == least_constrained_idx:
                new_radii[i] += expansion_factor * 1.2  # Targeted expansion
            else:
                # Expand slightly, but keep expansion factor below critical thresholds
                new_radii[i] += expansion_factor * 0.9
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = np.clip(new_radii, 1e-4, 0.5)
        
        # Re-evaluate with expanded radii and maintained topology
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
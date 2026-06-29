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
    
    # Apply targeted geometric dissection on the three most constrained circles
    if res.success:
        v = res.x
        # Calculate distances between all circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Compute adjacency matrix for all circle pairs
        adj = dists <= (v[2::3] + v[2::3].reshape(-1, 1))
        # Compute adjacency-based constraint scores for all circles
        constraint_scores = np.sum(adj, axis=1)
        # Identify three circles with the most constrained spatial relationships
        most_constrained_indices = np.argsort(constraint_scores)[::-1][:3]
        
        # Isolate and reconfigure the three most constrained circles
        for idx in most_constrained_indices:
            # Apply controlled perturbation to their positions
            v[3*idx] += np.random.uniform(-0.03, 0.03)
            v[3*idx+1] += np.random.uniform(-0.03, 0.03)
            # Apply small radius adjustment
            v[3*idx+2] += np.random.uniform(-0.002, 0.002)
        
        # Re-evaluate with modified configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on the least constrained circle with adjacency-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        # Calculate distances between all circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Compute adjacency matrix for all circle pairs
        adj = dists <= (radii + radii.reshape(-1, 1))
        # Calculate minimum distances for all circles
        min_dists = np.min(dists, axis=1)
        # Identify the least constrained circle
        least_constrained_idx = np.argmax(min_dists)
        # Find the circle with the smallest radius
        smallest_radius_idx = np.argmin(radii)
        # Compute current total sum
        total_sum = np.sum(radii)
        # Calculate expansion factor for controlled radius increase
        expansion_factor = 0.006 / (n - 1)  # Controlled expansion to unlock new configuration
        
        # Create adjusted radius vector with adjacency-based expansion
        new_radii = radii.copy()
        # Expand the least constrained circle
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        # Expand the smallest radius circle more
        new_radii[smallest_radius_idx] += expansion_factor * 1.2
        # Expand all other circles proportionally
        for i in range(n):
            if i != least_constrained_idx and i != smallest_radius_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = np.clip(new_radii, 1e-4, 0.5)
        
        # Re-evaluate with expanded radii and new adjacency constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
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
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-12})

    # Targeted geometric dissection: isolate three most constrained circles
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate pairwise distances and identify most constrained circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        min_dists = np.min(dists, axis=1)
        constrained_indices = np.argsort(min_dists)[-3:]  # Get three most constrained
        
        # Store current configuration for reconfiguration
        baseline_v = v.copy()
        
        # Reconfigure the three most constrained circles with fixed spacing
        for idx in constrained_indices:
            # Fix the position of the constrained circle
            # Apply geometric offset to free up space
            v[3*idx] = centers[idx, 0] + np.random.uniform(-0.03, 0.03)
            v[3*idx+1] = centers[idx, 1] + np.random.uniform(-0.03, 0.03)
            # Reduce its radius to allow expansion
            v[3*idx+2] = max(1e-4, radii[idx] - 0.003)
        
        # Re-optimization with new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})
    
    # Targeted radius expansion on least constrained circle with adjacency-aware expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor to increase the least constrained circle's radius
        total_sum = np.sum(radii)
        expansion_factor = 0.012 / (n - 1)  # Controlled expansion
        
        # Adjust radii to increase least constrained circle's radius
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.4  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = np.clip(new_radii, 1e-4, 0.5)
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})

    v = res.x if res.success else baseline_v if "baseline_v" in locals() else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
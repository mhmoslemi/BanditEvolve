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
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
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
    
    # Vectorized overlap constraints with random geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Hybrid reconfiguration: randomize spatial constraints with geometric hashing
    if res.success:
        v = res.x
        # Create a random geometric hash map for new configuration
        random_hash = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted geometric dissection: isolate and reconfigure two interacting circles
    if res.success:
        v = res.x
        # Identify two circles with minimal distance between them
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_matrix[i, j] = np.sqrt(dx*dx + dy*dy)
                dist_matrix[j, i] = dist_matrix[i, j]
        # Find the two circles with the smallest distance
        min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        i, j = min_dist_idx
        
        # Save their initial positions and radii
        pos_i = np.array([v[3*i], v[3*i+1]])
        pos_j = np.array([v[3*j], v[3*j+1]])
        r_i = v[3*i+2]
        r_j = v[3*j+2]
        
        # Fix their positions temporarily for reconfiguration
        for k in range(n):
            if k == i or k == j:
                continue
            v[3*k] = v[3*k] + np.random.uniform(-0.01, 0.01)
            v[3*k+1] = v[3*k+1] + np.random.uniform(-0.01, 0.01)
        
        # Re-optimize with fixed positions for i and j
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
        
        # Restore the fixed positions
        if res.success:
            v[3*i] = pos_i[0]
            v[3*i+1] = pos_i[1]
            v[3*j] = pos_j[0]
            v[3*j+1] = pos_j[1]
        
        # Apply controlled radius expansion to the least constrained circle
        if res.success:
            v = res.x
            radii = v[2::3]
            # Find the circle with the smallest non-zero radius
            smallest_radius_idx = np.argmin(radii)
            # Expand its radius while maintaining constraints
            expansion_factor = 0.002
            v[3*smallest_radius_idx + 2] += expansion_factor
            # Re-optimize with adjusted parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
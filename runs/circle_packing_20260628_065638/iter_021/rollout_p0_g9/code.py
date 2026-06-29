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
        x = x_center + np.random.uniform(-0.02, 0.02)
        y = y_center + np.random.uniform(-0.02, 0.02)
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
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-10})

    # Apply shake heuristic to smallest circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the smallest circles to shake
        smallest_indices = np.argsort(radii)[:5]
        # Apply small random perturbations to their positions
        for i in smallest_indices:
            v[3*i] += np.random.uniform(-0.02, 0.02)
            v[3*i+1] += np.random.uniform(-0.02, 0.02)
            v[3*i+2] += np.random.uniform(-0.002, 0.002)
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-10})

    # Targeted geometric dissection: isolate and reconfigure the two most interacting circles
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the two most dynamically interacting circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
                dists[j, i] = dists[i, j]
        # Find top two most interacting circles
        interactions = dists[np.triu_indices(n, k=1)]
        top_idx = np.argsort(interactions)[-2:]
        i1, i2 = top_idx[0], top_idx[1]
        
        # Fix radii of these two circles and re-optimize
        fixed_radii = radii.copy()
        fixed_radii[i1] = fixed_radii[i2] = 0.0
        v_fixed = v.copy()
        v_fixed[3*i1+2] = v_fixed[3*i2+2] = 0.0
        
        # Re-optimize with fixed radii
        res = minimize(neg_sum_radii, v_fixed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-10})
    
    # Controlled radius expansion on the least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the circle with the least constraint (largest allowable expansion)
        allowable_expansion = 0.5 - radii
        least_constrained_idx = np.argmax(allowable_expansion)
        
        # Apply directional expansion to this circle
        expansion_factor = 0.015
        max_radius = 0.5
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = min(radii[least_constrained_idx] + expansion_factor, max_radius)
        
        # Update decision vector
        v_new = v.copy()
        v_new[3*least_constrained_idx+2] = new_radii[least_constrained_idx]
        
        # Re-evaluate with expanded radii and new adjacency constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
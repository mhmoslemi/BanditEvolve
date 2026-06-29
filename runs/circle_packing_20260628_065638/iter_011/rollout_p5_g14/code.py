import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and asymmetry
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Randomized offset with row-dependent asymmetry
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
        # Alternate row staggering for asymmetry
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
    
    # Vectorized overlap constraints
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
    
    # Enforce radical spatial reconfiguration with fractal-based placement
    if res.success:
        v = res.x
        # Create a new initial guess with a more compact, randomized layout
        new_xs = np.random.uniform(0.1, 0.9, n)
        new_ys = np.random.uniform(0.1, 0.9, n)
        new_v = np.empty(3 * n)
        new_v[0::3] = new_xs
        new_v[1::3] = new_ys
        new_v[2::3] = v[2::3]  # Keep previous radii to maintain feasibility
        
        # Re-run optimization with the new arrangement
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    
    # Targeted radius expansion of the outermost circle
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Find the outermost circle based on distance to boundary
        dist_to_boundary = np.zeros(n)
        for i in range(n):
            x, y, r = centers[0][i], centers[1][i], radii[i]
            dist_to_boundary[i] = min(x - r, 1.0 - x - r, y - r, 1.0 - y - r)
        outermost_idx = np.argmin(dist_to_boundary)
        # Expand its radius slightly while adjusting position to stay within bounds
        v[3*outermost_idx + 2] += 0.005
        # Limit expansion to prevent penetration
        for i in range(n):
            if i == outermost_idx:
                v[3*i] = np.clip(v[3*i], 1e-6, 1.0 - v[3*i + 2])
                v[3*i + 1] = np.clip(v[3*i + 1], 1e-6, 1.0 - v[3*i + 2])
            else:
                v[3*i] = np.clip(v[3*i], 1e-6, 1.0 - v[3*i + 2])
                v[3*i + 1] = np.clip(v[3*i + 1], 1e-6, 1.0 - v[3*i + 2])
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
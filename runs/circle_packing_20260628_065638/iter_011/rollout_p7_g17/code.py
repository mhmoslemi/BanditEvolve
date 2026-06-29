import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Dynamic fractal-based placement to break symmetry and avoid local optima
    xs = []
    ys = []
    for i in range(n):
        # Base grid placement
        base_col = i % cols
        base_row = i // cols
        x = (base_col + 0.5) / cols
        y = (base_row + 0.5) / rows
        
        # Fractal-like displacement based on row and column indices
        x += (np.sin(base_row * 2 * np.pi / cols) + 1) / 4 * 0.05
        y += (np.cos(base_col * 2 * np.pi / rows) + 1) / 4 * 0.05
        
        # Alternate row staggering for asymmetry
        if base_row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
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
    
    # Radical spatial reconfiguration: introduce fractal displacement to trigger new layout
    if res.success:
        v = res.x
        # Apply controlled fractal displacement to outermost circles to test new topology
        outer_indices = np.argsort(np.sqrt((v[0::3] - 0.5)**2 + (v[1::3] - 0.5)**2))[:5]
        for i in outer_indices:
            v[3*i] += np.random.uniform(-0.05, 0.05)
            v[3*i+1] += np.random.uniform(-0.05, 0.05)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted expansion of outermost circle with strict boundary enforcement
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate distance to boundary for all circles
        min_dist_to_boundary = np.zeros(n)
        for i in range(n):
            x, y, r = centers[0][i], centers[1][i], radii[i]
            min_dist_to_boundary[i] = min(x - r, 1.0 - x - r, y - r, 1.0 - y - r)
        # Identify the outermost circle (smallest min distance)
        outermost_idx = np.argmin(min_dist_to_boundary)
        # Expand its radius while maintaining boundary constraints
        v[3*outermost_idx + 2] += 0.002
        # Adjust position to respect boundaries
        for _ in range(10):
            x, y, r = centers[0][outermost_idx], centers[1][outermost_idx], radii[outermost_idx]
            # Move towards the nearest boundary
            x += np.sign(1.0 - x - r - (x - r)) * 0.001
            y += np.sign(1.0 - y - r - (y - r)) * 0.001
            v[3*outermost_idx] = np.clip(x, 1e-6, 1.0 - 1e-6)
            v[3*outermost_idx + 1] = np.clip(y, 1e-6, 1.0 - 1e-6)
            # Re-evaluate with adjusted parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 10, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
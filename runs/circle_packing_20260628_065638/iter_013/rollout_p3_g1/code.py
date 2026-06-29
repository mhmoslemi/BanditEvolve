import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Randomized geometric hashing to break symmetry
        x += np.random.uniform(-0.1, 0.1)
        y += np.random.uniform(-0.1, 0.1)
        # Add jitter to promote non-uniform distribution
        x += np.random.normal(0, 0.02)
        y += np.random.normal(0, 0.02)
        # Shift alternate rows for staggered grid
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
    
    # Trigger topological overhaul with randomized geometric hashing
    if res.success:
        v = res.x
        # Create new positions using randomized geometric hashing
        new_xs = []
        new_ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            # Add randomized jitter for hashing
            x += np.random.uniform(-0.1, 0.1)
            y += np.random.uniform(-0.1, 0.1)
            # Add normal jitter for fine-grained distribution
            x += np.random.normal(0, 0.02)
            y += np.random.normal(0, 0.02)
            # Shift alternate rows for staggered grid
            if row % 2 == 1:
                x += 0.5 / cols
            new_xs.append(x)
            new_ys.append(y)
        
        # Reinitialize with new positions
        new_v = np.empty(3 * n)
        new_v[0::3] = np.array(new_xs)
        new_v[1::3] = np.array(new_ys)
        new_v[2::3] = np.full(n, r0)
        
        # Re-evaluate with new positions and original constraints
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion: expand the smallest non-zero radius circle
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate constraint tightness (inverse of minimum distance to boundary)
        min_dist_to_boundary = np.zeros(n)
        for i in range(n):
            x, y, r = centers[0][i], centers[1][i], radii[i]
            min_dist_to_boundary[i] = min(x - r, 1.0 - x - r, y - r, 1.0 - y - r)
        # Identify the circle with the smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        # Expand its radius slightly and adjust its position to maintain feasibility
        v[3*smallest_radius_idx + 2] += 0.003
        v[3*smallest_radius_idx] += 0.005
        v[3*smallest_radius_idx+1] += 0.005
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
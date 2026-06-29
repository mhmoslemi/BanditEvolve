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

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Trigger a constrained topological shift: replace the parent's spatial arrangement
    # with a randomized geometric tiling scheme
    if res.success:
        v = res.x
        # Re-initialize positions with randomized geometric tiling
        new_xs = []
        new_ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            # Randomized offset to break symmetry
            x = x_center + np.random.uniform(-0.05, 0.05)
            y = y_center + np.random.uniform(-0.05, 0.05)
            # Shift alternate rows for staggered grid
            if row % 2 == 1:
                x += 0.5 / cols
            new_xs.append(x)
            new_ys.append(y)
        # Preserve radii from previous solution
        new_v = np.empty(3 * n)
        new_v[0::3] = np.array(new_xs)
        new_v[1::3] = np.array(new_ys)
        new_v[2::3] = v[2::3]
        # Re-evaluate with new positions
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion on the most under-constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius (most under-constrained)
        min_radius_idx = np.argmin(radii)
        total_sum = np.sum(radii)
        # Set a target total radius sum for controlled expansion
        target_total_sum = total_sum + 0.007
        # Expand the smallest radius while preserving constraints
        expansion = (target_total_sum - total_sum) / (n - 1)
        # Apply expansion to all other circles except the smallest one
        for i in range(n):
            if i != min_radius_idx:
                v[3*i + 2] += expansion
        # Re-evaluate after expansion
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
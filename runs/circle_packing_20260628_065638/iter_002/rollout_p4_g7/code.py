import numpy as np

def run_packing():
    n = 26
    cols = 5  # Manual adjustment to use hexagonal grid
    rows = (n + cols - 1) // cols  # Ensure enough rows for 26 circles
    
    # Initialize positions using a hexagonal grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Offset even rows for hexagonal packing
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Add soft constraints for boundary and overlap with penalty
    def boundary_penalty(v):
        penalty = 0.0
        for i in range(n):
            x = v[3*i]
            y = v[3*i+1]
            r = v[3*i+2]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += max(0, (x - r) * 1000) + max(0, (1 - x - r) * 1000) + max(0, (y - r) * 1000) + max(0, (1 - y - r) * 1000)
        return penalty

    def overlap_penalty(v):
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_sum = v[3*i+2] + v[3*j+2]
                if dist < r_sum - 1e-8:
                    penalty += max(0, (r_sum - dist - 1e-8) * 100)
        return penalty

    # Combine penalties into a single objective function with weights
    def neg_sum_radii_with_penalty(v):
        return -np.sum(v[2::3]) + boundary_penalty(v) + overlap_penalty(v)

    # Define hard constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Run optimization with increased iterations and tighter tolerance
    res = minimize(neg_sum_radii_with_penalty, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-8})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
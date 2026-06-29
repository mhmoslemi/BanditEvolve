import numpy as np

def run_packing():
    n = 26
    cols = 5  # Manual adjustment for a hexagonal grid
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

    # Constraint functions for boundaries
    def boundary_constraints(v, i):
        x = v[3*i]
        y = v[3*i+1]
        r = v[3*i+2]
        return [
            {"type": "ineq", "fun": lambda v: v[3*i] - v[3*i+2]},
            {"type": "ineq", "fun": lambda v: 1.0 - v[3*i] - v[3*i+2]},
            {"type": "ineq", "fun": lambda v: v[3*i+1] - v[3*i+2]},
            {"type": "ineq", "fun": lambda v: 1.0 - v[3*i+1] - v[3*i+2]}
        ]

    cons = []
    for i in range(n):
        cons.extend(boundary_constraints(v0, i))
    
    # Constraint functions for circle overlaps
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Add a penalty-based constraint to handle overlap violations
    def overlap_penalty(v):
        total_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[3*i+2] + v[3*j+2]
                if dist_sq < r_sum**2 - 1e-8:
                    total_penalty += max(0, r_sum**2 - dist_sq)
        return total_penalty

    # Add penalty to the objective function
    def penalized_neg_sum_radii(v):
        return -np.sum(v[2::3]) + 1000 * overlap_penalty(v)

    res = minimize(penalized_neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
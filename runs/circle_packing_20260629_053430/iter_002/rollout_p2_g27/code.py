import numpy as np

def run_packing():
    n = 26
    cols = 5  # Fixed hexagonal grid with 5 columns for better spacing
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
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Apply a controlled perturbation to the seed configuration
    perturbation = np.random.uniform(-0.01, 0.01, size=3*n)
    v = np.clip(v + perturbation, 0.0, 1.0)
    v[2::3] = np.clip(v[2::3], 1e-4, 0.5)

    # Re-run optimization with the perturbed configuration
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 600, "ftol": 1e-9})
    v = res.x if res.success else v

    # Final refinement with local search
    def local_search(v, maxiter=100):
        for _ in range(maxiter):
            idx = np.random.randint(0, n)
            x, y, r = v[3*idx], v[3*idx+1], v[3*idx+2]
            # Small random move for local refinement
            dx, dy = np.random.uniform(-0.01, 0.01, size=2)
            new_x = np.clip(x + dx, 0.0, 1.0)
            new_y = np.clip(y + dy, 0.0, 1.0)
            new_r = np.clip(r + np.random.uniform(-0.001, 0.001), 1e-4, 0.5)
            v[3*idx], v[3*idx+1], v[3*idx+2] = new_x, new_y, new_r
        return v

    v = local_search(v)

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
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

    # Apply logarithmic transformation to coordinates to enhance global search space
    def log_polar_transform(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        # Log-polar transformation: log(r) and polar angle theta
        r_log = np.log(r + 1e-12)
        theta = np.arctan2(y, x)
        return np.concatenate([r_log, theta, r])

    def inverse_log_polar_transform(v):
        r_log = v[0::3]
        theta = v[1::3]
        r = v[2::3]
        # Inverse transformation
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.concatenate([x, y, np.exp(r_log) - 1e-12])

    # Initial optimization with log-polar coordinates
    v_log_polar = log_polar_transform(v0)
    cons = []
    for i in range(n):
        # Bounds on log-polar coordinates
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - 1e-2})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 10.0 - v[3*i]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - (-np.pi)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: np.pi - v[3*i+1]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v_log_polar, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v_log_polar
    v = inverse_log_polar_transform(v)

    # Final refinement in original space
    bounds_final = []
    for _ in range(n):
        bounds_final += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_final = []
    for i in range(n):
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_final(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons_final.append({"type": "ineq", "fun": constraint_func_final})

    res_final = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds_final,
                         constraints=cons_final, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_final.x if res_final.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
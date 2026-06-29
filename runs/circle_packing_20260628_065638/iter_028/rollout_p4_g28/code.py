import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize with a randomized yet structured grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols + 0.02 * np.sin(0.7 * i)
        y_center = (row + 0.5) / rows + 0.02 * np.cos(0.7 * i)
        # Staggered grid with dynamic shift
        x = x_center + np.random.uniform(-0.02, 0.02)
        y = y_center + np.random.uniform(-0.02, 0.02)
        if row % 2 == 1:
            x += 0.5 / cols * (0.9 + 0.1 * np.random.rand())
        xs.append(x)
        ys.append(y)
    
    # Initial radius calculation from spatial distribution
    r0 = 0.34 / (cols + 1) - 1e-3
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

    # Vectorized overlap constraints with enhanced gradient precision
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with high accuracy and multiple reinitializations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1700, "ftol": 1e-11, "eps": 1e-12})

    # Advanced shake heuristic for escaping local optima
    def shake_heuristic(v):
        if not res.success:
            return v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Identify the smallest non-zero radius circles (candidates for shaking)
        sorted_indices = np.argsort(radii)
        small_radii = sorted_indices[:2]  # Target smallest 2 circles
        # Jiggle small circles with radius-adaptive perturbation
        for idx in small_radii:
            # Perturb positions
            dx = np.random.uniform(-0.02, 0.02) * (radii[idx] / 0.3) 
            dy = np.random.uniform(-0.02, 0.02) * (radii[idx] / 0.3)
            v[3 * idx] += dx
            v[3 * idx + 1] += dy
            # Expand radius slightly while checking bounds
            v[3*idx + 2] += np.random.uniform(0.001, 0.003) * (radii[idx] < 0.002)
            # Clamp radii to enforce lower bounds
            v[3*idx + 2] = np.clip(v[3*idx + 2], 1e-3, 0.5)
        return v

    # First shake with fine gradient adjustments
    v = shake_heuristic(res.x)
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Second shake for deeper local escape
    v = shake_heuristic(res.x)
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
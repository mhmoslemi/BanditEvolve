import numpy as np

def run_packing():
    n = 26
    # Use a hexagonal grid seeding for better initial arrangement
    radius_initial = 0.02
    centers = []
    for i in range(n):
        row = i // 3
        col = i % 3
        x = col * (1.5 * radius_initial) + 0.5 * radius_initial
        y = row * (np.sqrt(3) * radius_initial) + 0.5 * radius_initial
        if i >= 9:
            y += (np.sqrt(3) * radius_initial)
        if i >= 18:
            y += (np.sqrt(3) * radius_initial)
        centers.append([x, y])
    # Adjust positions to fit inside the unit square
    max_x = max(c[0] for c in centers)
    max_y = max(c[1] for c in centers)
    scales = [1.0 / max_x, 1.0 / max_y]
    centers = np.array(centers) * np.min(scales)
    # Create initial guess
    v0 = np.empty(3 * n)
    v0[0::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
    v0[2::3] = np.full(n, radius_initial)
    # Define bounds
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    # Objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    # Constraints
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
    # Optimize
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
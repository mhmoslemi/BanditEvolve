import numpy as np

def run_packing():
    n = 26
    # Initial guess with grid-based placement
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = 0.5 / cols - 1e-3  # Initial radius

    # Bounds: x, y in [0, 1], radius in [1e-4, 0.5]
    bounds = [(0.0, 1.0) for _ in range(n * 3)]

    # Objective: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraints
    constraints = []
    for i in range(n):
        # Left and right boundaries
        constraints.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        constraints.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Top and bottom boundaries
        constraints.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        constraints.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Circle-circle distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
            constraints.append({"type": "ineq", "fun": constraint})

    # Optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-8})

    # Extract results
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
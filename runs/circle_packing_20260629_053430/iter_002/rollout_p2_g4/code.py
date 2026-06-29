import numpy as np

def run_packing():
    n = 26
    cols = 5  # Use a fixed hexagonal grid with 5 columns for better spacing
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
        cons.append({"type": "ineq", "fun": (lambda i: lambda v: v[3*i] - v[3*i+2])(i)})
        cons.append({"type": "ineq", "fun": (lambda i: lambda v: 1.0 - v[3*i] - v[3*i+2])(i)})
        cons.append({"type": "ineq", "fun": (lambda i: lambda v: v[3*i+1] - v[3*i+2])(i)})
        cons.append({"type": "ineq", "fun": (lambda i: lambda v: 1.0 - v[3*i+1] - v[3*i+2])(i)})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                          "fun": (lambda i, j: lambda v:
                                  (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                                  - (v[3*i+2] + v[3*j+2])**2)(i, j)})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Apply controlled perturbation to force new exploration path
    perturb = 0.05
    v = v + np.random.normal(0, perturb, size=v.shape)
    v = np.clip(v, 0, 1)
    v[2::3] = np.clip(v[2::3], 1e-4, 0.5)

    # Final optimization
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
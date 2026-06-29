import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    # Use hexagonal grid initialization for better spacing
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii_grid = np.sqrt(1 / (2 * np.pi)) * np.sqrt(n / (np.pi * 2))
    x_grid = radii_grid * np.cos(angles)
    y_grid = radii_grid * np.sin(angles)
    
    # Offset the grid to fit within the unit square
    x_grid = (x_grid + 1) / 2
    y_grid = (y_grid + 1) / 2
    
    # Initialize the decision vector
    v0 = np.empty(3 * n)
    v0[0::3] = x_grid
    v0[1::3] = y_grid
    v0[2::3] = np.full(n, 0.02)  # Start with small radii

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
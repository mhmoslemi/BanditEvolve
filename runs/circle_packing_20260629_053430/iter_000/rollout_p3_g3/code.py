import numpy as np

def run_packing():
    n = 26
    # Use a hexagonal grid seeding with spiral placement for better initial distribution
    # Create a hexagonal grid with spacing based on initial radius estimate
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    grid = np.zeros((n, 2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if r % 2 == 0:
                x = c / cols
            else:
                x = (c + 0.5) / cols
            y = r / rows
            grid[idx] = [x, y]
            idx += 1

    # Initial radius estimate based on hexagonal packing density
    initial_radius = 0.25 / np.sqrt(2)  # Estimate based on hexagonal packing in unit square
    v0 = np.empty(3 * n)
    v0[0::3] = grid[:, 0]
    v0[1::3] = grid[:, 1]
    v0[2::3] = initial_radius

    # Bounds for each circle: x, y in [0, 1], radius in [1e-4, 0.5]
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraints: non-overlapping circles and within unit square
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

    # Use SLSQP optimization method with increased iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})

    # Use the optimized solution if successful, otherwise fall back to initial guess
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
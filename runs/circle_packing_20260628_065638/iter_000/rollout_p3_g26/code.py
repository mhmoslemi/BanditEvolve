import numpy as np

def run_packing():
    n = 26
    # Initialize with a hexagonal grid seeding for better initial packing
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Hexagonal grid coordinates with staggered rows
    xs = []
    ys = []
    for r in range(rows):
        for c in range(cols):
            # Even rows: center-aligned
            if r % 2 == 0:
                x = c / cols
                y = r / rows
            else:
                # Odd rows: shifted right
                x = (c + 0.5) / cols
                y = r / rows
            xs.append(x)
            ys.append(y)
    
    # If we have more points than the grid, fill the remaining with random positions
    while len(xs) < n:
        x = np.random.rand()
        y = np.random.rand()
        xs.append(x)
        ys.append(y)
    
    # Trim to exactly 26 points
    xs = xs[:n]
    ys = ys[:n]
    
    # Initial radii based on grid spacing
    r0 = 0.5 / np.sqrt(n) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

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
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
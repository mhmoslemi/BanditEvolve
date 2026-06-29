import numpy as np

def run_packing():
    n = 26
    # Use a hexagonal grid seeding approach for better initial configuration
    cols = int(np.ceil(np.sqrt(n)))
    hex_rows = int(np.ceil(np.sqrt(n * 2 / 3)))
    hex_cols = int(np.ceil(n / hex_rows))
    
    # Generate initial positions using a hexagonal grid
    xs = []
    ys = []
    for i in range(hex_rows):
        for j in range(hex_cols):
            x = j + 0.5 * (i % 2)
            y = i * np.sqrt(3) / 2
            xs.append(x)
            ys.append(y)
    
    # Trim or pad to exactly 26 circles
    xs = np.array(xs[:n] + xs[:n][::-1][:n - len(xs)])
    ys = np.array(ys[:n] + ys[:n][::-1][:n - len(ys)])
    
    # Scale to unit square and shift to center
    xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs)) * 0.95
    ys = (ys - np.min(ys)) / (np.max(ys) - np.min(ys)) * 0.95
    xs += (1 - np.max(xs)) / 2
    ys += (1 - np.max(ys)) / 2
    
    # Initialize radii
    r0 = 0.05
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
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
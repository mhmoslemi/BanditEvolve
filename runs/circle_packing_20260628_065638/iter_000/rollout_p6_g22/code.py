import numpy as np

def run_packing():
    n = 26
    cols = 5  # Fixed number of columns for hexagonal grid
    rows = 6  # Sufficient rows for 26 circles with hexagonal packing
    
    # Generate hexagonal grid initial positions
    xs = np.zeros(n)
    ys = np.zeros(n)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 0:
                xs[idx] = col / (cols - 1)
                ys[idx] = row / (rows - 1)
            else:
                xs[idx] = (col + 0.5) / (cols - 1)
                ys[idx] = row / (rows - 1)
            idx += 1
    # Trim to 26 circles if needed
    xs = xs[:n]
    ys = ys[:n]
    
    r0 = 0.05  # Initial radius based on hexagonal packing
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
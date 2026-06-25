import numpy as np

def run_packing():
    n = 26
    # Generate initial positions using a hexagonal grid pattern
    cols = 6
    rows = 5
    positions = []
    for row in range(rows):
        for col in range(cols):
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if row % 2 == 1:
                x += 0.5 / cols
            positions.append((x, y))
    centers = np.array(positions[:n], dtype=float)
    
    # Initial radius guess based on spacing between centers
    initial_radius = 0.1
    radii = np.full(n, initial_radius, dtype=float)
    
    # Define decision vector and bounds
    v0 = np.empty(3 * n)
    v0[::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
    v0[2::3] = radii
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n
    
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
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint})
    
    # Run optimizer
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    # Hexagonal grid seeding with spiral adjustment
    cols = 5
    rows = 6
    grid = np.zeros((n, 2))
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 0:
                x = (col + 0.5) / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            grid[idx] = [x, y]
            idx += 1
    # Add spiral points
    for i in range(cols * rows, n):
        angle = 2 * np.pi * i / n
        radius = 0.3 + 0.1 * i / n
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        grid[i] = [x, y]
    
    # Initial radii based on grid spacing
    r0 = np.zeros(n)
    for i in range(n):
        min_dist = np.inf
        for j in range(n):
            if i != j:
                dx = grid[i][0] - grid[j][0]
                dy = grid[i][1] - grid[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
        r0[i] = min_dist * 0.45
    
    v0 = np.empty(3 * n)
    v0[0::3] = grid[:, 0]
    v0[1::3] = grid[:, 1]
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
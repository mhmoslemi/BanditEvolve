import numpy as np

def run_packing():
    n = 26
    # Use a hexagonal grid-like initialization for better spacing
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Generate initial positions in a hexagonal grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5 * (row % 2)) / cols
        y = row / rows
        xs.append(x)
        ys.append(y)
    
    # Initial radius based on cell size
    cell_diag = np.sqrt((1/cols)**2 + (1/rows)**2)
    r0 = cell_diag / 4 - 1e-3
    
    # Decision vector v = [x0, y0, r0, x1, y1, r1, ...]
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Bounds for each circle: x, y in [0,1], r in [1e-4, 0.5]
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to maximize sum of radii (minimize negative sum)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraints
    cons = []
    
    # Boundary constraints for each circle
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Circle-circle distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Define constraint function for each pair
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[3*i+2] + v[3*j+2]
                return dist_sq - r_sum*r_sum
            cons.append({"type": "ineq", "fun": constraint_func})

    # Optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "eps": 1e-8})
    
    # Extract results
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())
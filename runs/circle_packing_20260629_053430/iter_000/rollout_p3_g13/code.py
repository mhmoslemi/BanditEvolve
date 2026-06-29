import numpy as np

def run_packing():
    n = 26
    # Use hexagonal grid seeding for better initial arrangement
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Generate initial positions on a hexagonal grid
    x_coords = []
    y_coords = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Offset rows for hexagonal packing
        if row % 2 == 1:
            x = (col + 0.5) / cols
        else:
            x = (col) / cols
        y = (row) / rows
        x_coords.append(x)
        y_coords.append(y)
    
    # Initial radius estimation based on grid spacing
    r0 = 0.5 / max(cols, rows) - 1e-3
    
    # Decision vector v = [x0,y0,r0, x1,y1,r1, ...], length 3*n
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(x_coords)
    v0[1::3] = np.array(y_coords)
    v0[2::3] = r0 * np.ones(n)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    cons = []
    # Add constraints for circle boundaries
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Add constraints for circle-circle overlaps
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Perform optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-8})
    
    # Use initial guess if optimization fails
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
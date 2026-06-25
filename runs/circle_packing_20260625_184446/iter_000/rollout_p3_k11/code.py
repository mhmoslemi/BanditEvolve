import numpy as np

def run_packing():
    n = 26
    # Use a hexagonal grid seeding approach for better initial distribution
    radius_initial = 0.03  # Initial guess for radii
    centers = np.zeros((n, 2))
    radii = np.full(n, radius_initial, dtype=float)
    
    # Generate hexagonal grid points
    # For 26 circles, approximate grid of 5 rows (hexagonal packing)
    # Each row has 6, 5, 6, 5, 6 circles (total 28, adjust to 26)
    
    # Define hexagonal grid points with some offset for better spacing
    row_offset = np.sqrt(3) / 2 * radius_initial  # Vertical offset between rows
    col_offset = 1.0 * radius_initial  # Horizontal offset between columns
    
    # Adjust grid to fit within [0,1]x[0,1] square
    # Define grid points with adjusted spacing
    # Start from center for better symmetry
    col_start = 0.5 - 2.5 * radius_initial
    row_start = 0.5 - 1.5 * radius_initial
    
    for i in range(n):
        row = i // 6
        col = i % 6
        x = col_start + col * (1.0 + 1.5 * radius_initial)
        y = row_start + row * (1.0 + row_offset)
        
        # Adjust to stay within the square
        x = np.clip(x, radius_initial, 1.0 - radius_initial)
        y = np.clip(y, radius_initial, 1.0 - radius_initial)
        
        centers[i, 0] = x
        centers[i, 1] = y
    
    # Optimization setup
    v0 = np.empty(3 * n)
    v0[0::3] = centers.flatten()
    v0[1::3] = centers.flatten()
    v0[2::3] = radii
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # x, y, radius
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Constraints: distance between centers >= sum of radii
    for i in range(n):
        for j in range(i + 1, n):
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint})
    
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())
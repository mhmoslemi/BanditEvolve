import numpy as np

def run_packing():
    n = 26
    # Use a hexagonal grid seeding strategy for better initial distribution
    cols = int(np.ceil(np.sqrt(n)))
    hex_radius = 0.3  # Adjust based on expected packing density
    radii = np.full(n, 0.05)  # Initial guess for radii
    
    # Generate hexagonal grid coordinates
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(cols):
        for col in range(cols):
            if row % 2 == 0:
                x = col * (2 * hex_radius) + hex_radius
            else:
                x = col * (2 * hex_radius) + hex_radius * 1.5
            y = row * (np.sqrt(3) * hex_radius)
            if idx < n:
                centers[idx] = [x, y]
                idx += 1
            if idx >= n:
                break
        if idx >= n:
            break
    
    # Adjust positions to fit in the unit square
    scale = 1.0 / (np.max(centers) - np.min(centers))
    centers = centers * scale
    centers -= np.min(centers)
    
    # Initial guess for optimization
    v0 = np.empty(3 * n)
    v0[0::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
    v0[2::3] = radii
    
    # Define bounds for x, y, and radii
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    # Objective function to maximize the sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints: circles must be inside the square and not overlap
    cons = []
    for i in range(n):
        # Left and right boundaries
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Top and bottom boundaries
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Run the optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    
    # Extract results
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())
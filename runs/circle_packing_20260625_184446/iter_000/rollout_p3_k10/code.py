import numpy as np

def run_packing():
    n = 26
    # Use a hexagonal grid initialization with spiral ordering for better packing
    cols = int(np.ceil(np.sqrt(n)))
    spiral = []
    i, j = 0, 0
    direction = 1
    for _ in range(n):
        spiral.append((i + 0.5, j + 0.5))
        if (i + j) % 2 == 0:
            i += direction
        else:
            j += direction
        # Change direction when reaching the edge
        if i > cols - 1 or i < 0 or j > rows - 1 or j < 0:
            direction *= -1
            i, j = i + direction, j
    centers = np.array(spiral[:n], dtype=float)
    
    # Initial radius based on spacing between points
    initial_radius = 0.25
    # Add a small perturbation to break symmetry
    perturbation = np.random.uniform(-0.05, 0.05, size=(n, 2))
    centers += perturbation
    # Ensure centers are within the unit square
    centers = np.clip(centers, 0, 1)
    
    # Initial guess for radii
    radii = np.full(n, initial_radius, dtype=float)
    
    # Decision vector v = [x0,y0,r0, x1,y1,r1, ...], length 3*n
    v0 = np.empty(3 * n)
    v0[0::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
    v0[2::3] = radii
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, matches v
    
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
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint})
    
    # Use SLSQP with increased maxiter for better convergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
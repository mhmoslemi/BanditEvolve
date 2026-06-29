import numpy as np

def run_packing():
    n = 26
    cols = 5  # Adjust for a more efficient spatial distribution
    
    # Generate initial positions using a randomized Voronoi tessellation approach
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    r0 = 0.02  # Start with small radii for more flexibility
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

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

    # Global optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    
    # If optimization succeeds, perform a local refinement by expanding the most isolated circle
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        
        # Identify the circle with the maximum distance to others
        distances = np.zeros(n)
        for i in range(n):
            dist = 0.0
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dist += dx*dx + dy*dy
            distances[i] = np.sqrt(dist)
        
        max_distance_index = np.argmax(distances)
        max_radius_index = np.argmax(radii)
        
        # Prioritize the most isolated circle for expansion
        if max_distance_index != max_radius_index:
            max_radius_index = max_distance_index
        
        # Perturb the most isolated circle and re-optimize
        v[3*max_radius_index + 2] += 0.002  # Increase radius
        v[3*max_radius_index + 0] += 0.005  # Slight x-position shift
        v[3*max_radius_index + 1] += 0.005  # Slight y-position shift
        
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
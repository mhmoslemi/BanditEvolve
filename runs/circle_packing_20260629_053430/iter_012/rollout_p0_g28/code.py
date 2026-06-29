import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized overlap constraint function with modified distance function
    def vectorized_overlap_constraint(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dist_sq = np.sum((x[:, np.newaxis] - x[np.newaxis, :])**2 + (y[:, np.newaxis] - y[np.newaxis, :])**2, axis=2)
        min_dist_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
        return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)

    # Build constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Add overlap constraints using vectorized function
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dist_sq = (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9})
    v = res.x if res.success else v0

    # Phase 1: Initial optimization
    v = res.x if res.success else v0

    # Phase 2: Decouple position and radius of the smallest circle
    # Create new variables for positions and radii
    # Re-optimizing with radius of the smallest circle as a separate target
    indices = np.argsort(v[2::3])
    smallest_circle_index = indices[0]
    
    # Set the position of the smallest circle to a random point within the square
    new_v = v.copy()
    new_v[3*smallest_circle_index] = np.random.uniform(0.0, 1.0)
    new_v[3*smallest_circle_index+1] = np.random.uniform(0.0, 1.0)
    
    # Create new constraints with the smallest circle's radius as a separate variable
    new_bounds = []
    for i in range(n):
        if i == smallest_circle_index:
            # Radius of the smallest circle is now a separate variable
            new_bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
        else:
            new_bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    new_cons = []
    for i in range(n):
        if i == smallest_circle_index:
            # For the smallest circle, we have separate constraints
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        else:
            # For other circles, use the same constraints
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            new_cons.append({"type": "ineq", "fun": constraint_func})

    res_decoupled = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=new_bounds,
                             constraints=new_cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9})
    v = res_decoupled.x if res_decoupled.success else v

    # Phase 3: Final optimization with the new configuration
    res_final = minimize(neg_sum_radii, v, method="SLSQP", bounds=new_bounds,
                         constraints=new_cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9})
    v = res_final.x if res_final.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
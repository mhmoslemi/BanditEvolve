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

    # Vectorized overlap constraint with modified distance function
    def vectorized_overlap_constraint(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
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

    # Phase 2: Apply radical decomposition mutation
    # Isolate the smallest circle (index 0), reparameterize its radius, and reoptimize
    # Keep other circles' positions and radii fixed
    
    # Get current radii and identify smallest circle
    current_radii = v[2::3]
    smallest_idx = np.argmin(current_radii)
    
    # Create a new decision vector where we fix all circles except the smallest
    # and make the smallest circle's radius a new variable
    fixed_v = v.copy()
    fixed_v[3*smallest_idx+2] = 0.0  # temporarily set to zero for optimization
    
    # Create a new constraint for the smallest circle's radius
    def radius_constraint(v, idx=smallest_idx):
        return v[3*idx+2] - 1e-4  # ensure radius is at least 1e-4
    
    # Rebuild constraints with the smallest circle's radius as a free variable
    cons_radial = []
    for i in range(n):
        if i == smallest_idx:
            # Only add constraints for position bounds for the smallest circle
            cons_radial.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons_radial.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons_radial.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons_radial.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        else:
            # For all other circles, add position and radius constraints
            cons_radial.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons_radial.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons_radial.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons_radial.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Add overlap constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_radial(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons_radial.append({"type": "ineq", "fun": constraint_func_radial})
    
    # Add the radial constraint for the smallest circle
    cons_radial.append({"type": "ineq", "fun": radius_constraint})
    
    # Perform the optimization with the smallest circle's radius as the target
    res_radial = minimize(neg_sum_radii, fixed_v, method="SLSQP", bounds=bounds,
                          constraints=cons_radial, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_radial.x if res_radial.success else v

    # Phase 3: Refinement with final optimization
    res_final = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                         constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_final.x if res_final.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
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

    # Phase 2: Apply geometric transformation to seed configuration
    scale = 1.1
    rotated_v = v.copy()
    rotated_v[0::3] *= scale
    rotated_v[1::3] *= scale
    rotated_v[1::3] += 0.1  # slight vertical shift to break symmetry
    rotated_v[2::3] *= scale

    bounds_transformed = []
    for _ in range(n):
        bounds_transformed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_transformed = []
    for i in range(n):
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_transformed(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons_transformed.append({"type": "ineq", "fun": constraint_func_transformed})

    res_transformed = minimize(neg_sum_radii, rotated_v, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons_transformed, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_transformed.x if res_transformed.success else v

    # Phase 3: Radical decomposition mutation - isolate the smallest circle
    # Identify the smallest circle
    radii = v[2::3]
    smallest_idx = np.argmin(radii)

    # Fix positions of all circles except the smallest
    fixed_positions = v.copy()
    fixed_positions[3*smallest_idx] = v[3*smallest_idx]
    fixed_positions[3*smallest_idx+1] = v[3*smallest_idx+1]
    fixed_positions[3*smallest_idx+2] = 0  # Set radius to zero to fix it

    # Create new bounds and constraints for the mutated configuration
    bounds_mutated = []
    for i in range(n):
        if i == smallest_idx:
            bounds_mutated += [(0.0, 1.0), (0.0, 1.0), (0.0, 0.0)]  # Fix radius to zero
        else:
            bounds_mutated += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    cons_mutated = []
    for i in range(n):
        if i == smallest_idx:
            # For the smallest circle, only constrain position
            cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        else:
            cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Add overlap constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            if i == smallest_idx or j == smallest_idx:
                def constraint_func_mutated(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                    return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
                cons_mutated.append({"type": "ineq", "fun": constraint_func_mutated})
            else:
                def constraint_func_mutated(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                    return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
                cons_mutated.append({"type": "ineq", "fun": constraint_func_mutated})

    # Optimize only the radius of the smallest circle
    def neg_sum_radius(v):
        return -v[3*smallest_idx+2]
    
    res_mutated = minimize(neg_sum_radius, fixed_positions, method="SLSQP", bounds=bounds_mutated,
                          constraints=cons_mutated, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_mutated.x if res_mutated.success else v

    # Phase 4: Final refinement
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
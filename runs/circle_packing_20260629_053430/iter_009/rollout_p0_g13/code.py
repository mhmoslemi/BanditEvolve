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

    # Build constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Add overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Apply non-linear coordinate warping
    def warp_coords(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        # Apply logarithmic scaling to coordinates and radii
        x = np.log(x + 1e-6)
        y = np.log(y + 1e-6)
        r = np.log(r + 1e-6)
        # Apply inverse transformation to bring back to unit square
        x = (np.exp(x) - 0.5) / (np.max(np.exp(x)) - 0.5)
        y = (np.exp(y) - 0.5) / (np.max(np.exp(y)) - 0.5)
        r = np.exp(r) - 0.5
        # Adjust to ensure all radii are positive
        r = np.clip(r, 1e-4, 0.5)
        return np.concatenate([x, y, r])

    warped_v = warp_coords(v)
    bounds_warped = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n

    # Rebuild constraints for warped configuration
    cons_warped = []
    for i in range(n):
        cons_warped.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_warped.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_warped.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_warped.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_warped(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons_warped.append({"type": "ineq", "fun": constraint_func_warped})

    res_warped = minimize(neg_sum_radii, warped_v, method="SLSQP", bounds=bounds_warped,
                         constraints=cons_warped, options={"maxiter": 500, "ftol": 1e-9})
    v = res_warped.x if res_warped.success else warped_v

    # Apply localized perturbation to smallest circles
    radii = v[2::3]
    indices = np.argsort(radii)
    perturbation = 0.01
    v[3*indices[0]] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[0]+1] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[0]+2] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]+1] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]+2] += np.random.uniform(-perturbation, perturbation)

    # Final optimization
    bounds_final = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n
    cons_final = []
    for i in range(n):
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_final(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons_final.append({"type": "ineq", "fun": constraint_func_final})

    res_final = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds_final,
                         constraints=cons_final, options={"maxiter": 500, "ftol": 1e-9})
    v = res_final.x if res_final.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
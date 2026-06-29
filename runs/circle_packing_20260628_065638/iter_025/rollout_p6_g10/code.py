import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset for fine-grained spatial perturbation
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Staggered grid for better density
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Consistent length 3n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint definitions with lambda captures
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased resolution and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-13})
    
    # Apply targeted shake heuristic for circle escape from local optima
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Determine the 5 smallest radii
        small_radius_indices = np.argsort(radii)[:5]
        # Apply perturbations to small circles to trigger configuration reconfiguration
        for idx in small_radius_indices:
            # Perturb center with small random offset
            perturb = np.random.uniform(-0.01, 0.01, size=2)
            new_x = centers[idx, 0] + perturb[0]
            new_y = centers[idx, 1] + perturb[1]
            # Ensure within bounds and recompute radii
            # Create new decision vector with updated position
            new_v = v.copy()
            new_v[3*idx] = new_x
            new_v[3*idx+1] = new_y
            # Re-optimize with perturbed position
            shake_res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                                constraints=cons, options={"maxiter": 200, "ftol": 1e-11, "eps": 1e-12})
            if shake_res.success:
                v = shake_res.x

    # Final optimization with refined perturbed configuration
    v = res.x if res.success else v0
    if res.success:
        v = res.x
        # Final optimization with tight tolerances
        final_res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-12})
        v = final_res.x if final_res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
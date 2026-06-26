import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    for i in range(n):
        for j in range(i + 1, n):
            # Define a constraint function for circle-circle distance
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})

    # If optimization fails, perform a local refinement
    if not res.success:
        v = v0
        for _ in range(3):
            # Perform a local optimization with tighter bounds and constraints
            local_bounds = bounds.copy()
            for k in range(n):
                local_bounds[3*k] = (max(0.0, v[3*k] - 0.1), min(1.0, v[3*k] + 0.1))
                local_bounds[3*k + 1] = (max(0.0, v[3*k + 1] - 0.1), min(1.0, v[3*k + 1] + 0.1))
                local_bounds[3*k + 2] = (max(1e-4, v[3*k + 2] - 0.01), min(0.5, v[3*k + 2] + 0.01))
            
            local_cons = cons.copy()
            for k in range(n):
                for l in range(k + 1, n):
                    def local_constraint_func(v, k=k, l=l):
                        dx = v[3*k] - v[3*l]
                        dy = v[3*k+1] - v[3*l+1]
                        dist_sq = dx*dx + dy*dy
                        min_dist_sq = (v[3*k+2] + v[3*l+2])**2
                        return dist_sq - min_dist_sq
                    local_cons.append({"type": "ineq", "fun": local_constraint_func})
            
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=local_bounds,
                           constraints=local_cons, options={"maxiter": 100, "ftol": 1e-9})
            if res.success:
                v = res.x
                break

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
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

    # Define constraints
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

    # Pairwise distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})

    # Local refinement with penalty method
    if res.success:
        v = res.x
        # Add penalty for overlaps and boundaries
        def penalty_func(v):
            penalty = 0
            # Boundary penalties
            for i in range(n):
                if v[3*i] < 0:
                    penalty += -v[3*i]
                if v[3*i] > 1:
                    penalty += v[3*i] - 1
                if v[3*i+1] < 0:
                    penalty += -v[3*i+1]
                if v[3*i+1] > 1:
                    penalty += v[3*i+1] - 1
                if v[3*i+2] < 1e-4:
                    penalty += -v[3*i+2]
                if v[3*i+2] > 0.5:
                    penalty += v[3*i+2] - 0.5
            # Overlap penalties
            for i in range(n):
                for j in range(i + 1, n):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < v[3*i+2] + v[3*j+2] - 1e-8:
                        penalty += (v[3*i+2] + v[3*j+2] - dist) ** 2
            return -np.sum(v[2::3]) + 1e3 * penalty

        # Refine with penalty function
        res_refine = minimize(penalty_func, v, method="SLSQP", bounds=bounds,
                              options={"maxiter": 100, "ftol": 1e-9})
        v = res_refine.x if res_refine.success else v
    else:
        v = v0

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
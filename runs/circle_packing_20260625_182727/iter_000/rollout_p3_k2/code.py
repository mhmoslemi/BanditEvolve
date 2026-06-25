import numpy as np

def run_packing():
    n = 26
    # Initial coarse grid layout
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    r0 = 0.5 / cols - 1e-3
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Global constraints: circle boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Pairwise circle separation constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint})

    # First stage: global optimization with SLSQP
    res_global = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                         constraints=cons, options={"maxiter": 100, "ftol": 1e-8})

    # Second stage: local refinement with L-BFGS-B
    if res_global.success:
        v = res_global.x
        radii = v[2::3]
        # Create new bounds based on current solution
        new_bounds = []
        for i in range(n):
            new_bounds += [(max(0.0, v[3*i] - v[3*i+2]), min(1.0, v[3*i] + v[3*i+2])),
                           (max(0.0, v[3*i+1] - v[3*i+2]), min(1.0, v[3*i+1] + v[3*i+2])),
                           (max(1e-4, radii[i] - 1e-3), min(0.5, radii[i] + 1e-3))]
        # Refine with L-BFGS-B
        res_local = minimize(neg_sum_radii, v, method="L-BFGS-B", bounds=new_bounds,
                            constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
        v = res_local.x if res_local.success else res_global.x
    else:
        v = v0

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
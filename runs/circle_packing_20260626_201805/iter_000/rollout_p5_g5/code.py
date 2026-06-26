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

    # Define constraints with closure capture and explicit parameter passing
    def constraint_func(i, j, v):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(i, j, v)})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "maxfun": 100000})
    
    # Add a local optimization step with fixed centers to refine radii
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        radii = np.clip(radii, 1e-6, None)
        
        # Fix centers and optimize only radii
        fixed_centers = centers
        radii_bounds = [(1e-4, 0.5) for _ in range(n)]
        def objective(radii_vec):
            return -np.sum(radii_vec)
        
        def distance_constraint(i, j, radii_vec):
            dx = fixed_centers[i, 0] - fixed_centers[j, 0]
            dy = fixed_centers[i, 1] - fixed_centers[j, 1]
            return dx*dx + dy*dy - (radii_vec[i] + radii_vec[j])**2
        
        local_cons = []
        for i in range(n):
            local_cons.append({"type": "ineq", "fun": lambda r, i=i: r[i]})
            local_cons.append({"type": "ineq", "fun": lambda r, i=i: 0.5 - r[i]})
        
        for i in range(n):
            for j in range(i + 1, n):
                local_cons.append({"type": "ineq", "fun": lambda r, i=i, j=j: distance_constraint(i, j, r)})
        
        res_local = minimize(objective, radii, method="SLSQP", bounds=radii_bounds,
                             constraints=local_cons, options={"maxiter": 200, "ftol": 1e-10})
        v = np.concatenate([fixed_centers.flatten(), res_local.x])
        radii = res_local.x
    else:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())
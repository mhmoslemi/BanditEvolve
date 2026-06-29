import numpy as np

def run_packing():
    n = 26
    # Generate initial positions using a random placement to enable diverse topology
    np.random.seed(42)
    xs = np.random.uniform(0.1, 0.9, n)
    ys = np.random.uniform(0.1, 0.9, n)
    r0 = 0.02  # Initial radius estimate based on random distribution
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # Global optimization with adaptive constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Local refinement with targeted expansion of isolated circles
    def local_optimization(v):
        def local_neg_sum_radii(v):
            return -np.sum(v[2::3])
        local_bounds = []
        for _ in range(n):
            local_bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
        local_cons = []
        for i in range(n):
            local_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            local_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            local_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            local_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        for i in range(n):
            for j in range(i + 1, n):
                def local_constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                    return dist_sq - min_dist_sq
                local_cons.append({"type": "ineq", "fun": local_constraint_func})
        
        # Identify isolated circles and prioritize expanding them
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        isolation_scores = []
        for i in range(n):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            min_dist = np.min(dists[dists > 1e-8])
            isolation_scores.append(min_dist - radii[i])
        
        # Sort circles by isolation score and prioritize those with higher values
        sorted_indices = np.argsort(isolation_scores)[::-1]
        for idx in sorted_indices[:5]:  # Apply refinement to top 5 most isolated circles
            v[3*idx + 2] += 0.001  # Increment radius
            v[3*idx + 0] += np.random.uniform(-0.005, 0.005)  # Slight perturbation
            v[3*idx + 1] += np.random.uniform(-0.005, 0.005)
        
        res_local = minimize(local_neg_sum_radii, v, method="SLSQP", bounds=local_bounds,
                            constraints=local_cons, options={"maxiter": 200, "ftol": 1e-9})
        return res_local.x if res_local.success else v

    v = local_optimization(v)
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
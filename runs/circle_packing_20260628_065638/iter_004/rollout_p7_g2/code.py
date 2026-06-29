import numpy as np

def run_packing():
    n = 26
    cols = 5  # Manual adjustment for a hexagonal grid
    rows = (n + cols - 1) // cols  # Ensure enough rows for 26 circles
    
    # Initialize positions using a hexagonal grid pattern with random perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols + np.random.uniform(-0.05, 0.05)
        y = (row + 0.5) / rows + np.random.uniform(-0.05, 0.05)
        # Offset even rows for hexagonal packing
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

    # Local optimization to polish the solution
    def local_optimization(v):
        def local_neg_sum_radii(v):
            return -np.sum(v[2::3])
        # Reduce bounds to avoid overly restrictive constraints
        local_bounds = []
        for _ in range(n):
            local_bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
        # Reduce constraints for faster local search
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
        res_local = minimize(local_neg_sum_radii, v, method="SLSQP", bounds=local_bounds,
                            constraints=local_cons, options={"maxiter": 200, "ftol": 1e-9})
        return res_local.x if res_local.success else v

    # Global optimization
    res_global = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                          constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v_global = res_global.x if res_global.success else v0

    # Additional optimization: perturb and re-optimize based on Voronoi tessellation
    def voronoi_based_optimization(v):
        def local_neg_sum_radii(v):
            return -np.sum(v[2::3])
        # Create Voronoi diagram to identify isolated points
        centers = np.column_stack([v[0::3], v[1::3]])
        from scipy.spatial import Voronoi
        try:
            vor = Voronoi(centers)
        except:
            return v
        # Find the isolated point with the largest radius
        radii = v[2::3]
        max_radius_index = np.argmax(radii)
        # Perturb the isolated point to allow expansion
        v[3*max_radius_index + 2] += 0.001
        v[3*max_radius_index + 0] += 0.005
        v[3*max_radius_index + 1] += 0.005
        res_voronoi = minimize(local_neg_sum_radii, v, method="SLSQP", bounds=bounds,
                              constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
        return res_voronoi.x if res_voronoi.success else v

    v = voronoi_based_optimization(v_global)
    v = local_optimization(v)
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
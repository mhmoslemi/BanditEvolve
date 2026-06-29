import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a hexagonal grid pattern
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
        res_local = minimize(local_neg_sum_radii, v, method="SLSQP", bounds=local_bounds,
                            constraints=local_cons, options={"maxiter": 200, "ftol": 1e-9})
        return res_local.x if res_local.success else v

    # Global optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0
    v = local_optimization(v)

    # Advanced topological shift: Voronoi-based refinement
    # Step 1: Compute Voronoi tessellation of initial positions
    from scipy.spatial import Voronoi
    vor = Voronoi(np.column_stack([v[0::3], v[1::3]]))
    regions = vor.regions
    points = vor.vertices

    # Step 2: Identify the most isolated circle (highest Voronoi region area)
    voronoi_areas = []
    for region in regions:
        if not region:
            continue
        polygon = points[region]
        area = 0.5 * np.abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], 1) - polygon[:, 1] * np.roll(polygon[:, 0], 1)))
        voronoi_areas.append(area)
    max_area_index = np.argmax(voronoi_areas)
    max_area_center = np.column_stack([v[0::3], v[1::3]])[max_area_index]

    # Step 3: Move the most isolated circle and re-optimize
    v[3*max_area_index + 0] = max_area_center[0] + np.random.uniform(-0.01, 0.01)
    v[3*max_area_index + 1] = max_area_center[1] + np.random.uniform(-0.01, 0.01)
    v[3*max_area_index + 2] = v[3*max_area_index + 2] + 0.001

    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
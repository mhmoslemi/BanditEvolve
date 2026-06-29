import numpy as np

def run_packing():
    n = 26
    # Use a Voronoi-based initialization to get more diverse positions
    # Generate initial points using a grid with some random perturbation
    grid = np.random.rand(6, 6)
    grid = np.sort(grid, axis=1)
    grid = np.sort(grid, axis=0)
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    points = grid.reshape(-1, 2)
    points = np.column_stack((points[:, 0], points[:, 1]))
    # Perturb points to avoid exact repetition
    points += np.random.rand(*points.shape) * 0.05
    points = np.clip(points, 0.0, 1.0)
    
    # Initialize radii
    r0 = 0.5 / np.sqrt(n) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = points[:, 0]
    v0[1::3] = points[:, 1]
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

    # Global optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Local refinement: expand the most isolated circle
    def local_optimization(v):
        def local_neg_sum_radii(v):
            return -np.sum(v[2::3])
        # Reduce bounds for local search
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
        # Identify the most isolated circle
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        distances = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    distances[i] += np.sqrt(dx*dx + dy*dy)
        isolated_index = np.argmin(distances)
        # Perturb the most isolated circle
        v[3*isolated_index + 2] += 0.001  # Small radius increment
        v[3*isolated_index + 0] += 0.005  # Move circle slightly
        v[3*isolated_index + 1] += 0.005
        res_local = minimize(local_neg_sum_radii, v, method="SLSQP", bounds=local_bounds,
                            constraints=local_cons, options={"maxiter": 200, "ftol": 1e-9})
        return res_local.x if res_local.success else v

    v = local_optimization(v)
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
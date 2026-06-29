import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering with controlled spatial distribution
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    # Initial positions with geometric clustering
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols + np.random.uniform(-0.05, 0.05)
        y = (row + 0.5) / rows + np.random.uniform(-0.05, 0.05)
        xs[i] = x
        ys[i] = y
    # Alternate row staggering for asymmetry
    for i in range(n):
        row = i // cols
        if row % 2 == 1:
            xs[i] += 0.5 / cols
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
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

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Radical reconfiguration with randomized geometric clustering
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Generate new cluster positions with controlled spatial distribution
        new_xs = np.random.rand(n)
        new_ys = np.random.rand(n)
        for i in range(n):
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols + np.random.uniform(-0.05, 0.05)
            y = (row + 0.5) / rows + np.random.uniform(-0.05, 0.05)
            new_xs[i] = x
            new_ys[i] = y
        # Alternate row staggering for asymmetry
        for i in range(n):
            row = i // cols
            if row % 2 == 1:
                new_xs[i] += 0.5 / cols
        # Re-evaluate with new spatial arrangement
        perturbed_v = np.empty(3 * n)
        perturbed_v[0::3] = new_xs
        perturbed_v[1::3] = new_ys
        perturbed_v[2::3] = radii
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion to the most tightly packed cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate constraint tightness (inverse of minimum distance to boundary)
        min_dist_to_boundary = np.zeros(n)
        for i in range(n):
            x, y, r = centers[0][i], centers[1][i], radii[i]
            min_dist_to_boundary[i] = min(x - r, 1.0 - x - r, y - r, 1.0 - y - r)
        # Identify the most tightly packed cluster (min min distance)
        tightest_cluster_idx = np.argmin(min_dist_to_boundary)
        # Expand its radius slightly and adjust its position to maintain feasibility
        v[3*tightest_cluster_idx + 2] += 0.003
        v[3*tightest_cluster_idx] += 0.005
        v[3*tightest_cluster_idx+1] += 0.005
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
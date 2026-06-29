import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Optimize initial position initialization with hexagonal grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add fine-grained staggered displacement based on hexagonal packing
        x = x_center + np.random.uniform(-0.025, 0.025)
        y = y_center + np.random.uniform(-0.025, 0.025)
        if row % 2 == 1:
            x += 0.5 / cols * 0.7
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with vectorization
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                          "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})

    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Identify the three most constrained circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        constrained_indices = np.argsort(min_dists)[-3:]
        
        # Isolate and freeze the three most constrained circles
        frozen_centers = centers.copy()
        frozen_radii = radii.copy()
        for idx in constrained_indices:
            frozen_centers[idx] = centers[idx]
            frozen_radii[idx] = radii[idx]
        
        # Create a new vector with fixed constrained circles
        v_fixed = np.concatenate([frozen_centers.flatten(), frozen_radii])
        
        # Run constrained optimization with frozen circles
        res = minimize(neg_sum_radii, v_fixed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]

            # Calculate distance matrix again for current configuration
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dists[i, j] = np.sqrt(dx**2 + dy**2)
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            total_sum = np.sum(radii)

            # Calculate controlled expansion for least constrained circle
            target_total_sum = total_sum + 0.0075
            expansion_factor = (target_total_sum - total_sum) / (n - 1)

            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.2
            for i in range(n):
                if i != least_constrained_idx:
                    new_radii[i] += expansion_factor * 1.05

            # Update decision vector with expanded radii
            v_expanded = v.copy()
            v_expanded[2::3] = np.clip(new_radii, 1e-4, 0.5)

            # Final optimization with expanded radii
            res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
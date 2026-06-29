import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Randomized geometric hashing for initial placement
    np.random.seed(42)
    hash_grid = np.random.rand(n, 2) * 0.5
    xs = hash_grid[:, 0] + np.random.uniform(-0.05, 0.05, size=n)
    ys = hash_grid[:, 1] + np.random.uniform(-0.05, 0.05, size=n)
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)

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

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    def constraint_func(i, j, v):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2

    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(i, j, v)})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance matrix for all pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Compute minimal distances to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Smallest minimal distance (most constrained?)

        # Check if the least constrained is the one with biggest spacing
        least_constrained_idx = np.argmax(min_dists)

        # Try to expand this circle
        target_total_sum = np.sum(radii) + 0.01
        expansion = (target_total_sum - np.sum(radii)) / (n - 1)

        # Create adjusted radii with asymmetric expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion

        # Apply expansion with safety check
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # Reduce the expansion to maintain feasibility
                new_radii = np.clip(new_radii, 1e-4, None)
                new_radii = (new_radii - np.min(new_radii)) / (np.max(new_radii) - np.min(new_radii)) * (target_total_sum - np.sum(radii)) + np.sum(radii)

        v_new = v.copy()
        v_new[2::3] = new_radii

        # Final optimization with new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Generate more diverse spatial patterns with cluster-based and random perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows

        # Use both row and column-based clustering
        x = x_center + np.random.uniform(-0.09, 0.09) * (1 - (row % 2)) + np.random.randn() * 0.002
        y = y_center + np.random.uniform(-0.09, 0.09) * (1 - (col % 2)) + np.random.randn() * 0.002

        # Apply staggered grid only on even rows
        if row % 2 == 0:
            x += 0.5 / cols * np.random.uniform(-0.5, 0.5)
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

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12})

    # Asymmetric reconfiguration with enhanced stochastic perturbation
    if res.success:
        v = res.x
        spatial_hash = np.random.rand(n, 2) * 0.05
        # Apply aggressive spatial reconfiguration
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * 1.2
            perturbed_v[3*i+1] += spatial_hash[i, 1] * 1.2
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    # Targeted radius expansion with dynamic constrained circle selection and validation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)

        # Find least constrained circle (minimum minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        radius_least = radii[least_constrained_idx]
        min_distance = min_dists[least_constrained_idx]

        # Calculate expansion threshold based on max radius that doesn't violate constraints
        max_possible_radius = (min_distance - 1e-12) / 2
        expansion_amount = max(0, max_possible_radius - radius_least) / 1.5

        # Apply expansion to least constrained circle only
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_amount

        # Re-evaluate with new configuration and constraints
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

        # Validate and refine
        def validate_expanded(radii, centers):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    if np.sqrt(dx*dx + dy*dy) < radii[i] + radii[j] - 1e-12:
                        return False
            return True

        # Iterative refinement of the expanded configuration
        while True:
            if validate_expanded(new_radii, expanded_centers):
                break
            else:
                new_radii[least_constrained_idx] -= expansion_amount * 0.02
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

        # Use the refined configuration
        v = expanded_v
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12})

    # Final configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
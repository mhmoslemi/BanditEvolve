import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols

    # Geometric hexagonal grid with randomized perturbation
    grid = np.zeros((n, 2))
    for i in range(n):
        col = i % cols
        row = i // cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        
        # Hexagonal grid offset and stochastic perturbation
        if row % 2 == 1:  # alternate row offset
            x += 0.5 / cols
        # Add stochastic perturbations to avoid symmetry
        x += np.random.uniform(-0.04, 0.04)
        y += np.random.uniform(-0.04, 0.04)
        grid[i] = np.array([x, y])

    # Initial radius based on hexagonal packing
    side_length = 1.0 / cols
    r0 = side_length * 0.35 - 1e-3

    v0 = np.empty(3 * n)
    v0[0::3] = grid[:, 0]
    v0[1::3] = grid[:, 1]
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized and efficient overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j:
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                         - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with aggressive settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Identify circle with minimum distance to others (most constrained)
        min_dists = np.min(dists, axis=1)
        most_constrained_idx = np.argmin(min_dists)

        # Total current sum and target sum
        current_total = np.sum(radii)
        target_total = current_total + 0.01

        # Calculate expansion vector
        expansion = (target_total - current_total) / (n - 1)

        # Perturb circle positions to trigger layout change
        perturbation = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        perturbed_v[0::3] += perturbation[:, 0]
        perturbed_v[1::3] += perturbation[:, 1]

        # Re-evaluate configuration after perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})

        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])

            # Re-calculate distances for constraint validation
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)

            # Recalculate constrained circle
            min_dists = np.min(dists, axis=1)
            most_constrained_idx = np.argmin(min_dists)

            # Expand all circles equally based on spatial constraints
            expansion = (0.01 / (n - 1)) * (1.0 + 0.1 * np.random.rand())

            # Apply expansion to all circles except most constrained
            new_radii = radii.copy()
            for i in range(n):
                if i != most_constrained_idx:
                    new_radii[i] += expansion

            # Validate configuration after expansion
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < new_radii[i] + new_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    for i in range(n):
                        if i != most_constrained_idx:
                            new_radii[i] -= expansion * 0.1
                    # If all radii are at min, we exit to avoid infinite loop
                    if np.all((new_radii > 1e-5) & (new_radii < 0.5)):
                        break

            # Final optimization
            final_v = v.copy()
            final_v[2::3] = new_radii

            res = minimize(neg_sum_radii, final_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
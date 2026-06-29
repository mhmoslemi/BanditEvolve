import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Step 1: Initialize with hexagonal tiling and randomized spatial perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Perturb the base positions using a hexagonal offset pattern
        base_x += 0.05 * np.cos(2 * np.pi * i / n)
        base_y += 0.05 * np.sin(2 * np.pi * i / n)
        x = base_x + np.random.uniform(-0.08, 0.08)
        y = base_y + np.random.uniform(-0.08, 0.08)
        # Apply staggered pattern for alternating rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)

    # Step 2: Use small initial radius so that constraints are not overly tight
    r0 = 0.33 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Step 3: Define bounds for all variables
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Step 4: Define objective function to maximize the sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Step 5: Vectorized constraint for boundaries
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

    # Step 6: Vectorized constraint for pairwise circle overlap
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Step 7: Run the first optimization with increased iterations and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})

    # Step 8: After first iteration, perform spatial hashing and perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Step 9: Create geometric hash to simulate spatial diversity without over-constraining
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        perturbed_v[0::3] += spatial_hash[:, 0] * (radii / np.mean(radii))
        perturbed_v[1::3] += spatial_hash[:, 1] * (radii / np.mean(radii))

        # Step 10: Run second optimization with same constraints but different perturbations
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Step 11: After second iteration, identify circle with least distance pressure
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Calculate pairwise distances using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find minimum distance of each circle to others
        min_dists = np.min(dists, axis=1)
        # Find circle with least distance pressure
        least_constrained_idx = np.argmin(min_dists)

        # Targeted radius expansion on least constrained circle
        target_total_radius = np.sum(radii) + 0.006  # Aim for 0.6% increase in total
        expansion_factor = (target_total_radius - np.sum(radii)) / (n)  # distribute to all

        # Create expansion vector with more weight on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Spatial hashing-based expansion with directional emphasis
                spatial_perturbation = np.random.rand() * 0.002
                new_radii[i] += expansion_factor * (1.0 + spatial_perturbation)

        # Step 12: Apply expansion with vectorized validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                break
            else:
                # If expansion invalid, scale down by 95%
                new_radii = radii + (new_radii - radii) * 0.95

        # Step 13: Update optimized vector
        v_new = v.copy()
        v_new[2::3] = new_radii

        # Re-evaluate with expanded radii and reconfigured positions
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Step 14: Final output with clamping to ensure no negative radii
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
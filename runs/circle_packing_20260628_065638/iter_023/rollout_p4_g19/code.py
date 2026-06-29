import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing
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
    
    # Apply shake heuristic to smallest circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the smallest circles to shake
        smallest_indices = np.argsort(radii)[:6]  # Shake top 6 smallest circles
        # Apply small random perturbations to their positions
        for i in smallest_indices:
            # Apply more subtle spatial perturbation with adaptive scaling
            spatial_perturb = 0.03 * (1.0 - radii[i] / r0)  # Larger perturbation for smaller circles
            v[3*i] += np.random.uniform(-spatial_perturb, spatial_perturb)
            v[3*i+1] += np.random.uniform(-spatial_perturb, spatial_perturb)
            # Add tiny randomized radius perturbation
            v[3*i+2] += np.random.uniform(-0.001 * (1.0 - radii[i]/r0), 0.001 * (1.0 - radii[i]/r0))
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 250, "ftol": 1e-10})
    
    # Asymmetric reconfiguration: introduce stochasticity in circle placement
    if res.success:
        v = res.x
        # Create a randomized geometric hashing for reordering
        random_hash = np.random.rand(n, 2) * 0.07
        perturbed_v = v.copy()
        for i in range(n):
            # Apply adaptive perturbation based on radius size
            # Larger perturbation for smaller circles
            perturb_scale = 0.5 * (1.0 - v[3*i+2]/r0)
            perturbed_v[3*i] += random_hash[i, 0] * perturb_scale
            perturbed_v[3*i+1] += random_hash[i, 1] * perturb_scale
            perturbed_v[3*i+2] += random_hash[i, 0] * 0.0005 * perturb_scale  # Tiny radius perturbation
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 250, "ftol": 1e-11})

    # Targeted radius expansion on the most isolated circle
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
        # Calculate isolation metric with exponential weighting
        isolation = np.sum(np.exp(-dists / (np.mean(dists) + 1e-8)), axis=1)
        isolated_idx = np.argmin(isolation)
        
        # Calculate expansion factor to increase the most isolated circle's radius
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.007
        expansion_factor = (target_total_sum - total_sum) / (n - 1)  # Controlled expansion

        # Adjust radii to increase the most isolated circle's radius
        new_radii = radii.copy()
        # Over-expansion for isolated circle with adaptive scaling
        over_expansion = 1.4 * expansion_factor * (1.0 - radii[isolated_idx]/r0)
        new_radii[isolated_idx] += over_expansion
        # Controlled expansion for all other circles
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor * (1.0 - radii[i]/r0)  # Adaptive expansion based on current size
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
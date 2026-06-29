import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initial grid setup with more refined staggered pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid spacing adapted for more flexibility
        x_center = (col + 0.5 + 0.05 * np.sin(np.pi * row / 2)) / cols
        y_center = (row + 0.5 + 0.05 * np.cos(np.pi * row / 2)) / rows
        # Randomized offset with adaptive amplitude
        offset_amp = 0.07 * (1.0 - (row / (rows - 1)) ** 2)
        x = x_center + np.random.uniform(-offset_amp, offset_amp)
        y = y_center + np.random.uniform(-offset_amp, offset_amp)
        # Alternate row staggering
        if row % 2 == 1:
            x += 0.5 / cols * (0.8 + 0.2 * (1 - row / (rows - 1)))
        xs.append(x)
        ys.append(y)

    r0 = 0.34 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Optimized constraint construction with lambda capturing
    cons = []
    for i in range(n):
        # Left constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with vectorized and pre-compiled lambdas
    for i in range(n):
        for j in range(i + 1, n):
            # Pre-compile lambda to avoid closure capture issues
            idx_i, idx_j = i, j
            def constraint_func(v):
                dx = v[3*idx_i] - v[3*idx_j]
                dy = v[3*idx_i+1] - v[3*idx_j+1]
                return dx*dx + dy*dy - (v[3*idx_i+2] + v[3*idx_j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization phase with increased tolerance and iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-10, "eps": 1e-9})

    # Implement 'shake' heuristic for local minima escape
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Select 5 circles with the smallest radii for perturbation
        min_indices = np.argsort(radii)[:5]
        perturbation_radius = 0.003 * np.max(radii)
        noise = np.random.normal(0, 0.0015, (5, 2))

        # Apply localized spatial displacement to small circles
        shaken_v = v.copy()
        for idx in min_indices:
            # Shift centers while respecting boundaries
            dx = noise[np.argwhere(min_indices == idx)[0][0], 0] * (radii[idx] / np.mean(radii))
            dy = noise[np.argwhere(min_indices == idx)[0][0], 1] * (radii[idx] / np.mean(radii))
            shaken_v[3*idx] += dx
            shaken_v[3*idx+1] += dy
            # Ensure bounds are respected
            if shaken_v[3*idx] < 0 or shaken_v[3*idx] > 1:
                shaken_v[3*idx] = np.clip(shaken_v[3*idx], 0, 1)
            if shaken_v[3*idx+1] < 0 or shaken_v[3*idx+1] > 1:
                shaken_v[3*idx+1] = np.clip(shaken_v[3*idx+1], 0, 1)
            # Adjust radius to allow for new spacing
            shaken_v[3*idx+2] = max(1e-6, min(0.5, shaken_v[3*idx+2] * 1.05))
        
        res = minimize(neg_sum_radii, shaken_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    # Post-optimization refiner with targeted expansion and density control
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Calculate pairwise distances matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)

        # Find least constrained circle by maximizing minimum distance to others
        least_constrained_idx = np.argmax(min_dists)

        # Dynamic expansion based on density and potential for expansion
        current_total = np.sum(radii)
        target_growth = 0.007
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))

        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.18 * np.random.rand())  # Stochastic expansion
                new_radii[i] += expansion_i
        
        # Apply expansion while respecting constraints
        for _ in range(3):  # Limit to 3 attempts for efficiency
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

            # Validate expanded configuration
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
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final radius update
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
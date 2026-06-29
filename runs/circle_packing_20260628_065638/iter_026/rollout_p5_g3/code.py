import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Initialize positions with randomized geometric clustering and staggered grid (with better spacing for small circles)
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Enhanced randomness with clustering avoidance and better spacing
        x_offset = np.random.uniform(-0.06, 0.06)
        y_offset = np.random.uniform(-0.06, 0.06)
        
        # Shift alternate rows to prevent direct column alignment
        if row % 2 == 1:
            x_center += 0.5 / cols
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Add small jitter to avoid identical positions that may cause numerical instability
        x += np.random.uniform(-0.005, 0.005)
        y += np.random.uniform(-0.005, 0.005)
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.4 / cols - 1e-2
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
        # Left side constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right side constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom side constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top side constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerance and more iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-12, "gtol": 1e-12})

    # Asymmetric reconfiguration with stochastic spatial rehash
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial randomness with adaptive scaling for enhanced reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-13})
    
    # Targeted radius expansion on least constrained circle with adaptive heuristic
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the least constrained circle using minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        smallest_radius = radii[least_constrained_idx]
        
        # Calculate expansion factor using adaptive heuristic and total sum constraint
        target_growth = 0.0085
        current_total = np.sum(radii)
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Apply expansion to the least constrained circle with adaptive perturbation
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                # Use stochastic expansion with adaptive scaling based on circle size
                expansion_i = expansion_factor * np.random.uniform(0.8, 1.0) * (1 + (radii[i] / smallest_radius))
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation (local refinement)
        iterations = 0
        while iterations < 3:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between circles
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
                # If overlap detected, reduce expansion slightly and retry
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
        
        # Re-optimize with expanded radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
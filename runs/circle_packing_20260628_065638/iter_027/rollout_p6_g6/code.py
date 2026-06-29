import numpy as np

def run_packing():
    n = 26
    cols = 6  # Subtle shift to a slightly wider grid for better spread
    rows = (n + cols - 1) // cols
    
    # Initialize positions with hierarchical randomized grid + stochastic spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid position with slight adjustment for better spacing
        base_x = (col + 0.35) / cols
        base_y = (row + 0.35) / rows
        
        # Add geometric hashing for irregular spatial perturbation
        x_hash = 0.03 * np.random.randn()
        y_hash = 0.02 * np.random.randn()
        x = base_x + x_hash
        y = base_y + y_hash
        
        # Alternate row offset for staggered layout
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Better initial radius scaling based on rows and cols
    r0 = 0.35 / max(cols, rows) - 1e-3  # Adaptive scaling improves overall density
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Consistent 3*n bounds length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints using lambdas with captured i
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})

    # Vectorized overlap constraints using lambdas with captured (i, j)
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with adaptive settings
    initial_opt_options = {"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-9}
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options=initial_opt_options)

    # Stochastic asymmetric reconfiguration with adaptive geometric hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply adaptive geometric hashing with spatial correlation to enhance local reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.04  # Reduced magnitude for smoother transitions
        perturbed_v = v.copy()
        for i in range(n):
            # Use relative scaling to enhance stability
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate after perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    # Targeted radius expansion on globally unweighted least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute all pairwise distances vectorized
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with largest minimum distance (least constrained)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute expansion vector with adaptive radial growth
        current_total = np.sum(radii)
        target_growth = 0.0065  # Increased growth target
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Apply expansion with stochastic variation
        new_radii = radii.copy()
        expansion = expansion_factor * (1.0 + 0.1 * np.random.rand(n))
        new_radii += expansion
        
        # Apply targeted increase on least constrained circle
        new_radii[least_constrained_idx] *= 1.2  # Sizable increase for unweighted circle

        # Constraint validation with adaptive fallback
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
                # If overlap detected, reduce expansion progressively
                new_radii = radii + (new_radii - radii) * 0.92
                iterations += 1
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final re-evaluation with adaptive optimization
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 7  # Increase resolution to more precisely tile the space
    rows = (n + cols - 1) // cols
    
    # Initialize positions with hierarchical geometric hashing and adaptive perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with adaptive centering
        x_center = (col + 0.5) / cols * 1.05  # Slight expansion for spacing
        y_center = (row + 0.5) / rows * 1.05
        # Adaptive perturbation using radius-based scaling
        radius_scale = np.sqrt(np.arange(n) / n)  # radius-dependent position perturbation
        radius_perturb = 1.0 / (np.sqrt(n) + 1)
        x = x_center + np.random.uniform(-radius_perturb, radius_perturb) * radius_scale[i]
        y = y_center + np.random.uniform(-radius_perturb, radius_perturb) * radius_scale[i]
        # Alternate row staggering with exponential adjustment
        if row % 2 == 1:
            x += ((col + 0.5)/cols) * np.exp(-0.5 * row)  # exponential row spacing decay
        xs.append(x)
        ys.append(y)
    
    r0 = 0.15 / cols  # More flexible initial radius based on increased cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n length exactly

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective function is total radius to maximize

    # Optimized constraint generation: lambda closure with captured i
    cons = []
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: 1.0 - (x_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: 1.0 - (y_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # High-performance vectorized overlap constraints with geometric hashing
    # Use a precomputed pair-wise distance grid and use vectorized operations
    # To reduce computational overhead of nested loops
    # Precompute adjacency and neighbor lists based on geometric hashing
    # First, create a vectorized distance matrix
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                                (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations and tighter convergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-10, "disp": False})
    
    # Hybrid reconfiguration: spatial constraint perturbation with radial adaptive adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling for enhanced reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.04  # Reduced random perturbation
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb with radius-based scaling
            # Perturb x and y coordinates proportionally to radius to maintain spatial integrity
            if radii[i] > 1e-4:  # Avoid division by zero for minimal radii
                perturb_ratio = (1.0 - np.log(radii[i]/r0)) if radii[i] > r0 else np.sqrt(radii[i]/r0)
                perturbed_v[3*i] += spatial_hash[i, 0] * radii[i] * np.exp(-0.5 * (i/n))
                perturbed_v[3*i+1] += spatial_hash[i, 1] * radii[i] * np.exp(-0.5 * (i/n))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9, "disp": False})
    
    # Targeted radius expansion with geometric expansion on least constrained circle using adaptive expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix using broadcasting for performance
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute least constrained circles by maximum minimum distance to others
        # Also enforce a minimum margin for robustness
        min_dists = np.min(dists, axis=1)
        min_dist_with_margin = np.minimum(min_dists, 2 * np.min(radii))
        least_constrained_idx = np.argmax(min_dist_with_margin)
        
        # Expand radius while respecting geometric constraints using adaptive expansion
        # Use a weighted expansion that prioritizes circles with higher potential space
        current_total = np.sum(radii)
        max_growth_ratio = 1.1  # Limit total expansion to avoid instability
        # Use dynamic expansion per circle based on spatial availability
        expansion_factor = 0.0032 * (np.mean(min_dists) / np.sqrt(0.1))  # radius-dependent scaling
        expansion_vec = expansion_factor * (1.0 + np.random.rand(n) * 0.2)  # slight stochastic variation
        
        # Apply expansion in a controlled way while maintaining validity
        # Use binary search on expansion factor to maintain minimal gap
        # Compute an initial expansion and backtrack if invalid
        expanded_v = v.copy()
        original_radii = expanded_v[2::3].copy()
        expanded_v[2::3] = radii + expansion_vec  # Initial expansion
        expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
        
        # Check for validity with tight epsilon due to higher precision
        def is_valid(expanded_v):
            centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            radii = expanded_v[2::3]
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if (x - r < -1e-12 or x + r > 1 + 1e-12
                        or y - r < -1e-12 or y + r > 1 + 1e-12):
                    return False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < (radii[i] + radii[j]) - 1e-12:
                        return False
            return True

        # Use backtracking and adaptive iteration for expansion
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = radii + expansion_vec
            if is_valid(expanded_v):
                best_v = expanded_v
                break
            # If invalid and expansion is possible, decrease expansion vector
            # Scale and backtrack if expansion is too large
            # Compute how much to reduce the expansion factor to make it valid
            # Estimate by finding minimal expansion that doesn't cause collision
            min_expansion = 0
            max_expansion = 1e-3
            for step in np.logspace(-3, 0, 100):
                candidate = radii + step * expansion_vec
                candidate_v = v.copy()
                candidate_v[2::3] = candidate
                if is_valid(candidate_v):
                    min_expansion = step
                    break
            if min_expansion == 0:
                break
            expansion_vec = expansion_vec * (min_expansion / expansion_vec[-1])
        
        # Update to the best valid expansion
        v = best_v

    # Final optimization after expansion
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
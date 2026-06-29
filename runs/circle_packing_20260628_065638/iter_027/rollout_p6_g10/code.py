import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with clustered, randomized, and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce clustered randomness with directional bias
        x = x_center + np.random.uniform(-0.08, 0.12)
        y = y_center + np.random.uniform(-0.08, 0.12)
        # Create row-wise staggering with gradient scaling
        if row % 3 == 1:
            x += (0.5 / cols) * (1.0 + 0.2 * np.random.rand())
        elif row % 3 == 0:
            y += (0.5 / rows) * (1.0 + 0.2 * np.random.rand())
        xs.append(x)
        ys.append(y)
    
    r0 = 0.37 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure bounds length is 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Boundary constraints with closure capture
    cons = []
    for i in range(n):
        # Left
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Distance constraints with closure preservation
    for i in range(n):
        for j in range(i + 1, n):
            # Closure with explicit parameter passing and index binding
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased iterations and tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "maxfun": 5000})

    # Asymmetric reconfiguration with controlled spatial perturbation and adaptive expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Introduce directional spatial disorder with gradient perturbation
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Perturbation scale based on relative contribution to total radius
            perturbation_scale = (radii[i] / np.sum(radii)) * 0.5
            perturbed_v[3*i] += spatial_hash[i, 0] * perturbation_scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * perturbation_scale
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "maxfun": 3000})

    # Targeted radius expansion on least constrained circle with directional bias
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimum distance for each circle to ensure constraint tracking
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor with directional bias based on cluster position
        if least_constrained_idx < n // 2:  # Left half bias
            expansion_multiplier = 1.25
        else:  # Right half bias
            expansion_multiplier = 1.15
        
        # Calculate expansion based on growth potential
        current_total = np.sum(radii)
        expected_growth = 0.0105  # Incremental target
        expansion_factor = expected_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector with directional bias
        new_radii = radii.copy()
        # Expand targeted circle with multiplier
        new_radii[least_constrained_idx] += expansion_factor * expansion_multiplier
        # Expand other circles with stochastic but directional perturbation
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand()) * 0.95
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        # Use a more refined binary search for expansion adjustment
        adjusted_new_radii = new_radii.copy()
        for _ in range(3):
            expanded_v = v.copy()
            expanded_v[2::3] = adjusted_new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between circles
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < adjusted_new_radii[i] + adjusted_new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion by 0.7% per iteration
                adjusted_new_radii = radii + (adjusted_new_radii - radii) * 0.97
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = adjusted_new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "maxfun": 2500})

    # Final result extraction
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
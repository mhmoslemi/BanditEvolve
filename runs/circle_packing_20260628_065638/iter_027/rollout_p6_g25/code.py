import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with asymmetric geometric hashing and adaptive spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Use asymmetric geometric hashing for randomization
        hash_x = np.random.normal(0, 0.03) * (1.0 + 0.1 * np.random.rand())
        hash_y = np.random.normal(0, 0.03) * (1.0 + 0.1 * np.random.rand())
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x = x_center + hash_x
            y = y_center + hash_y
        else:
            x = x_center + hash_x
            y = y_center + hash_y
        # Apply boundary correction
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    # Initialize radii using adaptive spacing
    # Calculate baseline spacing and apply scaling with jitter for asymmetry
    baseline_radius = 0.35 / cols
    r0 = baseline_radius - np.random.uniform(0.001, 0.003)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Must match 3n elements

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda closures and i-indexing
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

    # Vectorized overlap constraints with lambda closure capturing i,j for each pair
    for i in range(n):
        for j in range(i + 1, n):
            # Ensure the constraint functions use proper captured i,j
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization pass with high precision and iteration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-12, "eps": 1e-12})

    # Asymmetric reconfiguration phase with randomized constraint relaxation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hashing with adaptive scaling based on radii distribution
        spatial_hash = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        
        # Apply asymmetric spatial perturbation with radius-based scaling
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii) + 0.01)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii) + 0.01)
        
        # Re-evaluate with constrained reconfiguration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-12})

        # Introduce targeted gradient perturbation for exploration
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            grad = np.gradient(radii)
            grad_perturbation = 0.001 * np.random.rand(n)
            # Apply directional gradient perturbation for asymmetry
            perturbed_v = v.copy()
            perturbed_v[2::3] += grad_perturbation
            perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
            
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-12})

    # Dynamic radius optimization with adaptive constraint relaxation and gradient nudging
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate minimum required spacing between any two circle centers
        min_spacing = np.mean(np.min(np.sqrt((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2), axis=2))
        target_spacing = min_spacing * 1.1  # Aim for 10% improvement
        # Calculate target radii for all circles given this spacing
        target_r = (target_spacing - 2 * np.min(radii)) / 2  # Based on packing density
        
        # Calculate expansion vector using adaptive gradient nudging for least constrained circles
        # Vectorized distance calculation for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor with adaptive heuristic
        current_total = np.sum(radii)
        expansion_factor = np.clip((target_spacing - 2 * radii[least_constrained_idx]) / (n - 1), 0.001, 0.01)
        
        # Apply expansion to least constrained and others with stochasticity
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())
        
        # Validate and refine expanded radii with adaptive constraint validation
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
                # If invalid, reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
        
        # Update decision vector with refined expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-12})

    # Final post-processing and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
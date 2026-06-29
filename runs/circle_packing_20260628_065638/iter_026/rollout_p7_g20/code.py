import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with optimized grid structure and randomized perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply randomized positional displacement with adaptive range
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Alternate row shift for staggered grid
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

    # Vectorized constraints with closure capture for boundary constraints
    cons = []
    for i in range(n):
        # Left edge
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right edge
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom edge
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top edge
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with enhanced numerical robustness
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2 - 1e-8  # Avoid floating point instability
                if dist_sq < min_dist_sq:
                    return min_dist_sq - dist_sq
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with advanced settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Asymmetric reconfiguration with targeted radius expansion
    if res.success:
        v = res.x
        # Calculate distances and identify least constrained circle
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.empty((n, n))
        
        # Vectorized pairwise distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distances for each circle to others
        min_dists = np.min(dists, axis=1)
        
        # Find the least constrained circle (max min distance)
        least_constrained_idx = np.argmax(min_dists)
        
        # Trigger reconfiguration with stochastic displacement
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate after perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted expansion for least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.006
        
        # Create expansion vector with careful gradient-aware adjustments
        expansion = np.zeros(n)
        expansion[least_constrained_idx] = 5.0  # Aggressive initial expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion[i] = 1.0 + 0.1 * np.random.rand()  # Stochastic expansion
        
        # Adaptive expansion control to prevent overlap
        max_expansion = np.zeros(n)
        max_expansion[least_constrained_idx] = 100.0
        
        # Create a gradient-based optimization function
        def expand_radius(v, centers, radii, expansion, max_expansion, cons, res, target_total_sum, total_sum, n):
            new_radii = v[2::3] + expansion
            new_radii = np.clip(new_radii, 1e-6, max_expansion)
            # Re-evaluate with new radii
            new_centers = np.column_stack([v[0::3], v[1::3]])
            # Validate expanded configuration
            for i in range(n):
                for j in range(i+1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        return v, False
            return v, True
        
        # Implement a gradient-based radius expansion
        expanded_v = v.copy()
        expanded_v[2::3] = radii
        
        # Apply expansion using iterative validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] += expansion * 0.01  # Small incremental expansion
            expanded_radii = np.clip(expanded_v[2::3], 1e-6, max_expansion)
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check for overlap
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                v = expanded_v
                radii = expanded_radii
                total_sum = np.sum(radii)
                break
        
        # Final expansion check
        final_total = np.sum(radii)
        if final_total < target_total_sum:
            # Final adjustment
            expansion_ratio = (target_total_sum - final_total) / (n - 1)
            for i in range(n):
                if i != least_constrained_idx:
                    radii[i] += expansion_ratio * (1 + 0.1 * np.random.rand())
            v[2::3] = radii
            v = v.clip(1e-6, 0.5)
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
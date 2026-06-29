import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Dynamic initialization with randomized geometric clustering, spatial hashing, and layered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid positions with staggered rows and randomized spatial hashing
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Add random spatial hashing for perturbation
        hash_offset = np.random.rand(2) * 0.04
        x = base_x + hash_offset[0]
        y = base_y + hash_offset[1]
        # Row-specific staggering
        if row % 3 == 1:
            x += 0.125 / cols
        elif row % 3 == 2:
            x -= 0.125 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with density-based scaling
    r0 = 0.45 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with tighter bounds and cleaner closure
    cons = []
    for i in range(n):
        # Left: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing and vectorization
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with improved convergence and precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-10})

    # Asymmetric reconfiguration phase with stochastic spatial hashing
    if res.success:
        v = res.x
        # Generate asymmetric spatial hashing for global layout reconfiguration
        hash_map = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0] * (1.0 + np.random.rand() * 0.3)
            perturbed_v[3*i+1] += hash_map[i, 1] * (1.0 + np.random.rand() * 0.3)
        # Re-evaluate the new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-10})
    
    # Targeted reconfiguration for least constrained circle with geometric reasoning
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial constraints for all circles
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        
        # Create a buffer for soft expansion
        buffer = 0.005
        target_total_sum = radii.sum() + 0.008
        expansion_factor = (target_total_sum - radii.sum()) / (n - 1)
        
        # Expand the least constrained circle with stochastic and soft constraints
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.25  # Over-expand slightly
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + np.random.rand() * 0.15)
        
        # Re-evaluate with new radii while maintaining constraints
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate the expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # If invalid, reduce expansion
                new_radii = radii.copy()
                new_radii[least_constrained_idx] += expansion_factor * 1.0
                for i in range(n):
                    if i != least_constrained_idx:
                        new_radii[i] += expansion_factor * (1.0 + np.random.rand() * 0.1)
        
        # Update and re-optimize
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-10})

    # Final optimization with convergence reinforcement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Final fine-tuning to ensure all constraints are respected
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-10:
                    # Adjust the smaller radius to maintain separation
                    if radii[i] < radii[j]:
                        radii[i] = max(radii[i], dist - radii[j] + 1e-10)
                    else:
                        radii[j] = max(radii[j], dist - radii[i] + 1e-10)
        
        # Update the decision vector
        v = np.concatenate([centers[:, 0], centers[:, 1], radii])
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
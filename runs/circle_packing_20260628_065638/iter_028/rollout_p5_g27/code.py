import numpy as np

def run_packing():
    n = 26
    cols = 3  # Reduce row count to allow more vertical space
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometry and enhanced spatial balance
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add spatial gradient for better spread
        x = x_center + np.random.uniform(-0.13, 0.13)
        y = y_center + np.random.uniform(-0.13, 0.13)
        
        # Row-dependent vertical staggering and horizontal offset 
        if row % 2 == 1:
            x += 0.5 / cols
        if np.random.rand() < 0.15:
            y += np.random.uniform(-0.15, 0.15)
        
        xs.append(x)
        ys.append(y)
    
    # Optimal initial radii derived from grid spacing and density
    r0 = 0.42 / rows - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Matches 3*n length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    # Vectorized bounds constraints
    for i in range(n):
        # Left bound
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})  # x + r <= 1
        # Right bound
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})        # x - r >= 0
        # Bottom bound
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]}) # y + r <= 1
        # Top bound
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})      # y - r >= 0

    # Efficient overlap constraints with gradient-based evaluation
    for i in range(n):
        for j in range(i+1, n):
            # Function with proper bound for numerical stability
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # 1st pass: initial optimization with high precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-12, "eps": 1e-6})
    
    # Generate geometric perturbation based on radii distribution
    if res.success:
        v = res.x
        # Compute radii and centers
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate perturbation map weighted by radius to enable controlled spatial shift
        perturbation_map = np.random.rand(n, 2) * 0.04
        weighted_perturbation = perturbation_map * (radii / np.mean(radii))
        
        # Apply perturbation with radius-proportional scaling
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += weighted_perturbation[i, 0]
            perturbed_v[3*i+1] += weighted_perturbation[i, 1]
        
        # 2nd pass: reconfiguration with higher precision
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-6})
    
    # Compute isolation metric using efficient vectorized approach
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Isolation score: inverse of proximity to other circles
        min_dists = np.min(dists, axis=1)
        isolation_score = 1.0 / (min_dists + 1e-8)
        least_constrained_idx = np.argmin(isolation_score)
        
        # Targeted expansion with radii proportionality
        current_total = np.sum(radii)
        target_growth = 0.0065
        max_possible = (1.0 - np.min(centers, axis=1) - radii) * 2
        expansion_factor = np.min(target_growth / max_possible) * 1.15
        
        # Calculate new radii with soft constraint enforcement
        new_radii = radii.copy()
        # Expand the least constrained circle with additional buffer
        new_radii[least_constrained_idx] = np.min([new_radii[least_constrained_idx] + expansion_factor,
                                                  0.7 - radii[least_constrained_idx]])
        
        # Distribute expansion to other circles with constraint-aware scaling
        for i in range(n):
            if i != least_constrained_idx:
                margin = (1.0 - centers[i, 0] - radii[i]) + (1.0 - centers[i, 1] - radii[i])
                if margin > 1e-6:
                    scale = np.min([1.0, expansion_factor * (margin / np.sum(margin))])
                    new_radii[i] = np.min([new_radii[i] + scale, 0.7 - radii[i]])
                else:
                    new_radii[i] = np.min([new_radii[i], 0.7 - radii[i]])
        
        # Apply new radii and optimize with tighter constraints
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-6})
    
    # Final validation and cleanup
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.7)  # Cap large radii for stability
    return centers, radii, float(radii.sum())
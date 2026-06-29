import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    # Use adaptive random seeds to introduce more diversity in initialization
    xs = []
    ys = []
    np.random.seed(int(np.random.rand() * 1000000))
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Adaptive random offset based on row and column proximity
        row_weight = np.sin(np.pi * row / rows) * 0.15
        col_weight = np.cos(np.pi * col / cols) * 0.15
        x = x_center + np.random.uniform(-row_weight, row_weight)
        y = y_center + np.random.uniform(-col_weight, col_weight)
        # Staggered grid with dynamic vertical offset to prevent row collapse
        if row % 2 == 1:
            x += (0.5 / cols) * (0.5 + np.sin(np.pi * row / rows))
        xs.append(x)
        ys.append(y)
    
    r0 = (0.42 / cols) * (1 - 1e-3) * np.ones(n) # adaptive initial radii
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # ensure bounds length is exactly 3n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries: use lambda with captured i
    cons = []
    for i in range(n):
        # Left boundary: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: 1 - (x_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: 1 - (y_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using lambda with captured i, j
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "maxcor": 100})
    
    # 1st-stage reconfiguration: forced geometric dissection of top two dynamically interacting
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance computation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the two most dynamically interacting (least distance to any other)
        interaction_score = np.sum(1 / (dists + 1e-10), axis=1) # avoid division by zero
        top_idx = np.argsort(interaction_score)[-2:]  # get two most interacting
        
        # Force dissection by introducing a new, large-scale constraint of reordering
        # Create two new clusters and enforce a hard separation constraint
        def hard_separation_constraint(v, i=0, j=1):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return np.clip(dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2, -1e5, None)
        
        # Create a new constraint for these two and perform secondary optimization
        new_cons = cons.copy()
        new_cons.append({"type": "ineq", "fun": hard_separation_constraint})
        
        # Perturbation to avoid local minima
        spatial_hash = np.random.rand(n, 2) * 0.1
        v_perturbed = v.copy()
        for i in range(n):
            v_perturbed[3*i] += spatial_hash[i, 0]
            v_perturbed[3*i+1] += spatial_hash[i, 1]
        
        # Secondary optimization focusing on structural reconfiguration
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 500, "ftol": 1e-12, "maxcor": 100})
    
    # 2nd-stage reconfiguration: target expansion on least constrained circle with multi-phase approach
    # After forced reconfiguration, perform a more robust expansion phase
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Recompute all distances with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2) + 1e-12  # avoid division by zero
        
        # Compute isolation score as reciprocal of min distance to other centers
        isolation_score = 1 / np.min(dists, axis=1)  # higher score indicates more isolated
        isolated_idx = np.argmax(isolation_score)
        
        # Compute total current sum and target expansion
        current_total = np.sum(radii)
        # Targeted growth based on global SOTA approach: 0.008 increase with controlled redistribution
        # Add small stochastic variation for more robustness
        target_total = current_total + np.random.uniform(0.005, 0.010)
        
        # Compute expansion distribution vector: expand isolated and spread to others
        # This is a two-phase expansion: isolate expansion + spread
        # Use a geometric progression to avoid sharp spikes and ensure stability
        expansion_factor = (target_total - current_total) / (n - 1)
        # Add 0.1% extra to isolated circle to stimulate growth
        expansion_factor_isolated = expansion_factor * 1.1
        expansion_factor_others = expansion_factor * 0.9
        
        # Apply expansion with conservative scaling
        expanded_radii = radii.copy()
        for j in range(n):
            if j == isolated_idx:
                expanded_radii[j] += expansion_factor_isolated
            else:
                expanded_radii[j] += expansion_factor_others
        
        # Check feasibility with expanded_radii before optimization
        expanded_centers = np.column_stack([v[0::3], v[1::3]])
        expanded_radii = expanded_radii.clip(1e-6, 0.45)  # clamp at reasonable upper bound
        
        # Prevalidation: check if expanded_radii meet all physical constraints
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Create new decision vector
            v_new = v.copy()
            v_new[2::3] = expanded_radii
            
            # Run final optimization with enhanced constraints and gradient tracking
            # Ensure constraints include both overlap and boundary conditions
            final_res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "maxcor": 100})
            res = final_res
        else:
            # If invalid, perform a constrained expansion by reducing the expansion factor
            # Use a more iterative expansion approach with binary search for feasible expansion
            # This is a simplified version of the full feasibility checking
            expansion_factor = 0.0
            while True:
                new_radii = radii.copy()
                for j in range(n):
                    if j == isolated_idx:
                        new_radii[j] += expansion_factor * 1.1
                    else:
                        new_radii[j] += expansion_factor * 0.9
                new_radii = new_radii.clip(1e-6, 0.45)
                expanded_centers = np.column_stack([v[0::3], v[1::3]])
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
                    expansion_factor -= 0.001
            if expansion_factor > 0:
                # Update radii and proceed to optimize
                v_new = v.copy()
                v_new[2::3] = new_radii
                final_res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "maxcor": 100})
                res = final_res
    
    # Final cleanup and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)  # clip to reasonable upper bound
    return centers, radii, float(radii.sum())
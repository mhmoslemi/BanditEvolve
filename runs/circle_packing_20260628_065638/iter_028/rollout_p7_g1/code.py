import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Use geometric tessellation with adaptive jitter and non-uniform sampling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Dynamic jitter based on row and radius density
        jitter = np.random.uniform(-0.06, 0.06)
        x = x_center + jitter
        jitter = np.random.uniform(-0.06, 0.06)
        y = y_center + jitter
        
        # Create staggered layout with row-dependent shift
        if row % 2 == 1:
            x += 0.5 / cols
        
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on spatial distribution and adaptive grid
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Enforce bounds consistent with decision vector of length 3*n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with stable capture of i
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized pairwise distance constraints with stable i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with high precision and stability
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Analyze top two dynamically interacting circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Identify two circles with highest mutual interaction (minimum distance)
        interaction = np.sum(dists, axis=1)
        top_idx = np.argsort(interaction)[-2:]  # Select top 2 most interacting
        
        # Specialized reconfiguration for these two circles with adaptive displacement
        # Create spatial displacement map with row-wise adjustment
        spatial_displacement = np.random.rand(n, 2) * 0.08
        displacement_factor = 1.5 * (radii[top_idx] / np.max(radii))  # Adaptive scaling
        perturbed_v = v.copy()
        
        # Apply directional displacement to high-interaction circles
        for idx in top_idx:
            dx = spatial_displacement[idx, 0] * displacement_factor[idx]
            dy = spatial_displacement[idx, 1] * displacement_factor[idx]
            perturbed_v[3*idx] += dx
            perturbed_v[3*idx+1] += dy
        
        # Re-optimization with localized structural changes
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 1e-11})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Targeted radius expansion of least constrained circle with adaptive constraints
        # Compute relative spacing matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        
        # Identify circle with largest minimal distance to others (most isolated)
        least_constrained_idx = np.argmax(min_dists)
        
        # Expand isolated circle while maintaining spatial integrity
        current_total = np.sum(radii)
        expansion_target = current_total + (0.006)  # Targeted expansion goal
        
        # Compute expansion potential with geometric constraint enforcement
        expansion_scalar = expansion_target / current_total
        # Calculate maximum allowable expansion while maintaining overlaps
        max_expandable = 0.04 * (1 - radii / np.max(radii))  # Soft limit based on density
        max_radius = np.min((radii + max_expandable, np.ones(n) * 0.5))
        
        # Apply expansion to least constrained with spatial enforcement
        # Create a buffer vector to handle expansion constraints
        expanded_radii = radii.copy()
        expanded_radii[least_constrained_idx] += 0.002 * expansion_scalar * (1.25)
        for i in range(n):
            if i != least_constrained_idx:
                # Apply proportional expansion to less constrained circles
                expansion_i = 0.001 * (expansion_scalar * (1.0 + 0.1 * np.random.rand()))
                expanded_radii[i] += expansion_i
        
        # Enforce spatial integrity of new radius configuration
        # Create a temporary buffer to validate the expansion
        v_expanded = v.copy()
        v_expanded[2::3] = expanded_radii
        centers_expanded = np.column_stack([v_expanded[0::3], v_expanded[1::3]])
        
        # Constraint validation via distance matrix
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                dx = centers_expanded[i, 0] - centers_expanded[j, 0]
                dy = centers_expanded[i, 1] - centers_expanded[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < (expanded_radii[i] + expanded_radii[j]) - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Update optimized vector with expanded radii
            res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 350, "ftol": 1e-10})
        else:
            # If expansion violates constraints, gradually reduce expansion
            while True:
                expanded_radii = radii + (expanded_radii - radii) * 0.95
                v_expanded = v.copy()
                v_expanded[2::3] = expanded_radii
                centers_expanded = np.column_stack([v_expanded[0::3], v_expanded[1::3]])
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        dx = centers_expanded[i, 0] - centers_expanded[j, 0]
                        dy = centers_expanded[i, 1] - centers_expanded[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < (expanded_radii[i] + expanded_radii[j]) - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
            res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 350, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
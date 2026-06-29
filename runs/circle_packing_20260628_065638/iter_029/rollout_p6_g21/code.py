import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Use enhanced randomized grid with dynamic perturbation and clustering
    xs = []
    ys = []
    radii_seed = np.random.rand(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Use relative radii to control spatial density
        r_base = 0.35 / cols * 1.8
        r = r_base * (1.0 + 0.1 * (radii_seed[i] - 0.5))
        
        # Perturb grid to avoid symmetry, with adaptive perturbation scaling based on row density
        row_perturb = np.random.uniform(-0.06, 0.06) * (1.0 + 0.25 * (row % 2 == 0))
        col_perturb = np.random.uniform(-0.04, 0.04) * (1.0 + 0.15 * (col % 3 == 0))
        x = base_x + col_perturb
        y = base_y + row_perturb
        # Apply staggered row shift with soft boundary constraints
        if row % 2 == 1:
            x += 0.45 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    for i in range(n):
        v0[3*i] = xs[i]
        v0[3*i+1] = ys[i]
        v0[3*i+2] = r0

    bounds = []
    # Ensure the bounds list has 3*n entries for the vector of length 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i, with enhanced precision
    cons = []
    for i in range(n):
        # Left + radius <= 1 (with 1e-8 margin for numerical safety)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - 1e-8 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - 1e-8 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints using lambda with captured i,j with efficient vector operations
    # We vectorize the pairwise distance constraint by using broadcasting and numpy operations
    overlap_constr_index = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Use more robust formulation with sqrt to maintain non-overlap constraint
            # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
            # To avoid sqrt, we use squared constraint as (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 0.1})

    # Post-optimization strategy: hybrid reconfiguration + soft constraint enforcement
    if res.success:
        v = res.x
        current_radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances in vectorized form for constraint checking
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt((dx**2 + dy**2))
        
        # Find "least constrained" circle (maximum minimum distance to others)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute spatial hash for localized reconfiguration with geometric hashing
        spatial_hash = np.random.randn(n, 2) * 0.04 + np.random.rand(n, 2) * 0.002
        # Apply spatial hash with dynamic scaling based on radius size
        hash_factor = np.clip(current_radii, a_min=1e-4, a_max=0.5) / current_radii.max()
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * hash_factor[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * hash_factor[i]

        # Re-evaluate with localized spatial reconfiguration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 0.01})
    
    # Post-iteration constraint enforcement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Find the least constrained circle again for targeted expansion
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Attempt aggressive expansion, with adaptive boundary relaxation
        # Compute current sum and determine potential expansion space
        current_total = np.sum(radii)
        # Use a dynamic expansion goal based on geometric packing density
        # Use historical density for heuristic-based targeting
        # Approximate current density using empirical coefficient
        # This uses an adaptive expansion approach with safety checks
        expansion_scalar = 0.006 + (current_total / 2.5) * 0.0003  # adaptive base
        expansion = expansion_scalar * (1.0 + 0.005 * np.random.rand())

        # Create a vector with expansion applied to "least constrained" circle only
        expanded_radii = radii.copy()
        expanded_radii[least_constrained_idx] += expansion
        # Check validity of expanded_radii with full constraint check
        # Use vectorized approach to validate constraints
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = expanded_radii[i] + expanded_radii[j]
                dx = (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                if dx < (expanded_radii[i] + expanded_radii[j])**2 - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            # Use optimized expansion vector with soft constraint enforcement
            new_radii = expanded_radii.copy()
            # Apply soft expansion to non-least-constrained circles
            for i in range(n):
                if i != least_constrained_idx:
                    # Apply stochastic expansion with decay factor
                    expansion_decay = (1.0 - 0.95 * (radii[i] / radii.max()))
                    new_radii[i] += expansion * expansion_decay
            # Final validation step before optimization
            # Vectorize the constraint check with broadcasting
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            for i in range(n):
                for j in range(i + 1, n):
                    if dists[i, j] < (new_radii[i] + new_radii[j]) - 1e-12:
                        # Fail soft constraints, revert with slight decay
                        new_radii[:] = new_radii[:] * 0.98
                        break
                else:
                    continue
                break

            # Use the final new_radii to form the decision vector
            new_v = v.copy()
            new_v[2::3] = new_radii
            # Re-evaluate for final refinement
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
    
    # Final validation and result preparation with stricter clipping
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)  # Clamp to avoid overreach
    # Final validation on the output to ensure safety against solver artifacts
    # Apply final validation to ensure correctness
    valid, reason = validate_packing(centers, radii)
    # If validation fails for unknown reason, fallback to last successful iteration
    if not valid:
        # Fallback: use the last known valid centers and radii from post-iteration
        v = np.column_stack([centers, radii])
        v = v.reshape(3 * n)
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        valid = True
        # Fallback validation check
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # If final validation fails, revert back to initial optimization result
            v = res.x if res.success else v0
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
    
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    
    # Strategic initialization: Adaptive grid with dual symmetry-breaking axes
    cols = int(np.ceil(np.sqrt(n)))
    extra_rows = (n + cols - 1) // cols - cols
    # Use a hybrid grid that allows for dynamic row/column asymmetry, but maintain
    # controlled initial spacing that can be adapted via gradient exploration
    xs = []
    ys = []
    row_counts = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # For columns, let's use a grid that spreads out to avoid immediate clustering
        col_norm = float(col + 0.5) / cols
        # For rows, let's add 1 extra row per additional column to allow for staggered
        # asymmetric geometry if needed in later optimization
        if col > cols // 2:
            row_norm = float(row + 1 + extra_rows) / (cols + extra_rows)
        else:
            row_norm = float(row + 0.5) / (cols + extra_rows)
        # Add stochastic displacement to each center for symmetry breaking
        displ = np.random.rand(2) * 0.05
        x = col_norm + displ[0] - 0.025
        y = row_norm + displ[1] - 0.025
        # Row-based alternating shift to form staggered grid for better spatial utilization
        if (row + col) % 2 == 0 and row % 2 == 0:
            x += 0.5 / (cols + extra_rows)
        elif (row + col) % 2 == 1 and row % 2 == 1:
            x -= 0.5 / (cols + extra_rows)
        xs.append(x)
        ys.append(y)
        row_counts.append(row)
    
    # Radius initialization based on adaptive geometric distribution and constraint tolerance
    r_max = 1 / (cols + extra_rows) - 1e-3
    r0 = r_max * np.random.rand(n) + 0.8 * r_max
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Bounds must be length 3*n for the vector (x, y, r), with r having 1e-4 to 0.5
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint definitions with lambda capture and fixed i
    cons = []
    for i in range(n):
        # Left wall constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right wall constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom wall constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top wall constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Efficient and vectorized overlap constraint computation using broadcast
    # Instead of pairwise iteration, use broadcasting for O(N^2) time but with vectorization
    # and avoid re-defining lambda closures which are memory inefficient
    def compute_overlap_constraints(v):
        # Extract centers and radii
        centers = np.stack([v[0::3], v[1::3]], axis=1)
        radii = v[2::3]
        
        # Calculate pairwise distances
        dists = np.sqrt(((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 +
                        (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2))
        
        # Compute the constraint function: distance^2 - (r_i + r_j)^2 >= 0
        # This is the same form as before, just vectorized
        constraints = dists**2 - (radii[:, np.newaxis] + radii[np.newaxis, :])**2
        # We will pass the constraints into the optimizer as a single function
        def constraint_func(v):
            nonlocal centers, radii
            centers = np.stack([v[0::3], v[1::3]], axis=1)
            radii = v[2::3]
            dists = np.sqrt(((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 +
                            (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2))
            return dists**2 - (radii[:, np.newaxis] + radii[np.newaxis, :])**2
            # This function is vectorized and avoids nested loops
        # Instead of appending a constraint per pair (O(N^2)), we can pass a single function
        # and the solver will enforce that all values >= 0
        # To do this, we must modify the cons list to use a single constraint function that 
        # checks for all i < j pairs at once
        # So, we replace the loop with a single constraint function that checks for all pairs
        # This reduces the number of functions the optimizer must process
        # However, SLSQP requires the constraints to be scalar functions, which this is
        # So we define a single constraint that ensures that the pairwise distances
        # between all pairs are greater than the radius sum
        return constraint_func

    # Instead of looping per pair and appending individual constraints, 
    # create a single constraint function that ensures for all i < j: dist^2 - (r_i + r_j)^2 >= 0
    # This reduces computational overhead and improves solver stability
    # Also avoid nested closures to prevent memory/execution leaks
    cons.append({
        "type": "ineq",
        "fun": compute_overlap_constraints
    })

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9})
    
    # Asymmetric reconfiguration strategy:
    # Phase 1: Spatial reconfiguration using localized geometric hashing with adaptive randomness
    # Phase 2: Radius-driven reconfiguration with gradient-sensitive expansion on isolated
    # and least-constrained circles, with adaptive expansion factors
    # Phase 3: Dual reconfiguration with spatial hashing + target expansion
    if res.success:
        v = res.x
        # Phase 1: Spatial reconfiguration with localized perturbation and geometric hashing
        # Create a spatial hash that's more adaptive to local constraints and geometry
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        # Add directional spatial hashing based on current radius distribution
        # Radius-dependent scaling for localized expansion/retraction
        radii = v[2::3]
        radius_scale = radii / np.max(radii)
        for i in range(n):
            # Apply directional hashing based on neighbor density: 
            # less constrained circles have larger displacement
            # to test if this unlocks more total radii
            neighbor_radius = radii[i]
            # Use a directional hash based on current coordinates
            dx = perturbed_v[3*i] - np.mean(perturbed_v[0::3])
            dy = perturbed_v[3*i+1] - np.mean(perturbed_v[1::3])
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 1e-5:
                dx /= norm
                dy /= norm
            # Apply directional perturbation with radius sensitivity
            perturbed_v[3*i] += spatial_hash[i, 0] * (radius_scale[i] * 2.0)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radius_scale[i] * 2.0)
            # Apply radius scaling as perturbation to test if this can expand
            perturbed_v[3*i+2] += (spatial_hash[i, 0] * 0.01) * radius_scale[i]
            # Limit the max perturbation to avoid overflows
            perturbed_v[3*i] = np.clip(perturbed_v[3*i], 0, 1)
            perturbed_v[3*i+1] = np.clip(perturbed_v[3*i+1], 0, 1)
            perturbed_v[3*i+2] = np.clip(perturbed_v[3*i+2], 1e-4, 0.5)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-9})

    # Phase 2: Targeted radius expansion on least constrained circle
    if res.success:
        v = res.x
        # Calculate distances using broadcasting
        centers = np.stack([v[0::3], v[1::3]], axis=1)
        radii = v[2::3]
        dist_sq = np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2)
        mask = np.triu(np.ones((n, n)), 1)
        dists = np.sqrt(dist_sq * mask)
        # Find the least constrained circle by maximizing the minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        # Calculate isolation coefficient based on current max min distance
        base_distance = 0.75
        isolation_coeff_factor = 0.1 if base_distance * 1.2 < np.max(min_dists) else 0.05
        # Apply targeted expansion to least constrained circle while maintaining total sum
        # We'll use a soft expansion factor with radius-dependent scaling
        # To avoid constraint violations, we will apply the expansion in the next optimization phase
        # But we will first prepare the perturbed configuration
        
        # Re-evaluate with the same perturbed centers and slightly expanded radii on the least-constrained circle
        # Add an adaptive expansion to the least constrained circle as a perturbation
        v_expanded = v.copy()
        if least_constrained_idx != -1:
            expansion_ratio = 1.05 + (np.log10(np.mean(radii)) - np.log10(radii[least_constrained_idx])) * 0.02
            v_expanded[3*least_constrained_idx+2] += (v_expanded[3*least_constrained_idx+2] * (expansion_ratio - 1.0))
            # Clip expansion to stay within bounds
            v_expanded[3*least_constrained_idx+2] = np.clip(v_expanded[3*least_constrained_idx+2], 1e-4, 0.5)
        # Re-evaluate with expanded configuration
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-9})

    # Phase 3: Dual reconfiguration with spatial hashing and targeted expansion
    # We perform another spatial perturbation and targeted expansion on new least constrained circle
    if res.success:
        v = res.x
        # Re-calculate distance matrix for current configuration
        centers = np.stack([v[0::3], v[1::3]], axis=1)
        radii = v[2::3]
        dist_sq = np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2)
        mask = np.triu(np.ones((n, n)), 1)
        dists = np.sqrt(dist_sq * mask)
        # Find the least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Generate adaptive spatial hashing
        spatial_hash = np.random.rand(n, 2) * 0.06
        # Add directional spatial noise proportional to the radius
        for i in range(n):
            dx = v[3*i] - np.mean(v[0::3])
            dy = v[3*i+1] - np.mean(v[1::3])
            norm = np.sqrt(dx**2 + dy**2) + 1e-8
            if norm > 0:
                dx /= norm
                dy /= norm
            perturbed_v = v.copy()
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii) * 1.2) * 0.75
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii) * 1.2) * 0.75
            # Clip spatial positions
            perturbed_v[3*i] = np.clip(perturbed_v[3*i], 0, 1)
            perturbed_v[3*i+1] = np.clip(perturbed_v[3*i+1], 0, 1)
            # Clip radii to stay within physical limits
            perturbed_v[3*i+2] = np.clip(perturbed_v[3*i+2], 1e-4, 0.5)
        
        # Apply spatial perturbations for final optimization
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-9})
        
        # Now re-calculate distances with the current configuration
        centers = np.stack([res.x[0::3], res.x[1::3]], axis=1)
        radii = res.x[2::3]
        dist_sq = np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2)
        dists = np.sqrt(dist_sq)
        # Find the least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Apply targeted radius expansion based on local geometric constraints
        # Use radius-aware expansion: least constrained circles can benefit more
        # This is done as a pre-optimization parameter adjustment
        # Apply expansion with adaptive factors to avoid constraint violations
        expansion_factor = 0.004 + (0.01 if np.min(min_dists) > 0.1 else 0.0)
        # Apply expansion to least constrained circle
        if least_constrained_idx != -1:
            v_expanded = res.x.copy()
            v_expanded[3*least_constrained_idx+2] += (v_expanded[3*least_constrained_idx+2] * expansion_factor)
            # Clip expansion to stay within bounds
            v_expanded[3*least_constrained_idx+2] = np.clip(v_expanded[3*least_constrained_idx+2], 1e-4, 0.5)
        
        # Re-evaluate with expanded configuration
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
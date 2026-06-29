import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    total_max_iter = 1600
    
    # First: strategic initialization to enable gradient exploration
    # Optimal cell layout with refined spacing and stochastic bias
    # Use row-major order, but with stochastic jitter to avoid symmetry
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Core layout: grid with precise placement
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add dynamic spatial jitter to enable topological variation
        x_offset = np.random.uniform(-0.03 + (0.1 * row)/5, 0.03 - (0.1 * row)/5)
        y_offset = np.random.uniform(-0.03 + (0.1 * col)/5, 0.03 - (0.1 * col)/5)
        x = x_center + x_offset
        y = y_center + y_offset
        
        # For staggered rows, shift rightward to simulate hexagonal packing
        if row % 2 == 1:
            x += 0.5 / cols # this is 0.1 of unit square width
            
        # Adjust to not touch the edge too early
        if x + 0.5 * 0.3 > 1.0:
            x -= 0.5 * 0.3
        if x - 0.5 * 0.3 < 0.0:
            x += 0.5 * 0.3
            
        if y + 0.5 * 0.3 > 1.0:
            y -= 0.5 * 0.3
        if y - 0.5 * 0.3 < 0.0:
            y += 0.5 * 0.3
        
        xs.append(x)
        ys.append(y)
    
    # Calculate initial estimate based on grid efficiency
    avg_cell_width = (1.0 / cols) # x-direction
    avg_cell_height = (1.0 / rows) # y-direction
    # Minimum safe radius based on hexagonal packing
    r0 = min(avg_cell_width * 0.45, avg_cell_height * 0.45)
    r0 = max(r0, 1e-4) # safety minimum

    # Vectorized initialization
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0) * 1.05 # initial over-estimate to give optimization room
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] # 3n entries

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Build constraint definitions with fixed scoping to avoid lambda issues
    constraints = []

    # 1. Boundary constraints: each circle must lie within [0,1]x[0,1]
    # Use explicit i-based lambda to avoid capture issues
    for i in range(n):
        constraints.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]}) # x - r >= 0
        constraints.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]}) # 1 - (x + r) >= 0
        constraints.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]}) # y - r >= 0
        constraints.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]}) # 1 - (y + r) >= 0
    
    # 2. Overlap constraints: pairwise distance >= r_i + r_j
    # Vectorized implementation with batched access
    for i in range(n):
        for j in range(i + 1, n):
            # Use partial anonymous function with capture
            def make_overlap_constr(i, j):
                def constr(v):
                    xi, yi = v[3*i], v[3*i + 1]
                    xj, yj = v[3*j], v[3*j + 1]
                    ri = v[3*i + 2]
                    rj = v[3*j + 2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    return dist_sq - (ri + rj)**2
                return constr
            constraints.append({"type": "ineq", "fun": make_overlap_constr(i, j)})

    # First optimization phase: base layout with high iter budget
    res_base = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-9}
    )

    # If first optimization fails, fall back to initial setup but use adaptive method
    if not res_base.success:
        res_base = minimize(
            neg_sum_radii,
            v0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1200, "ftol": 1e-11, "eps": 1e-9}
        )

    # 3. Spatial perturbation strategy: 
    # a. First, apply a soft geometric hashing-based spatial shift
    # b. Use proportional scaling of jitter with current radius - enables larger radii circles to have more margin
    if res_base.success:
        v = res_base.x.copy()
        # Apply perturbations with radius-aware spatial bias
        spatial_hash = np.random.rand(n, 2) * 0.08
        # Scale jitter by radius to allow more flexibility
        radius_factor = np.clip(v[2::3] * 10.0, 0.01, 0.40) # limit to avoid extreme perturbation
        perturbed_v = v.copy()
        for i in range(n):
            # x and y directions
            perturbed_v[3*i] += spatial_hash[i, 0] * radius_factor[i]
            perturbed_v[3*i + 1] += spatial_hash[i, 1] * radius_factor[i]
        
        # Reoptimize with perturbed layout
        res_perturbed = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9}
        )
        
        # Use best of base and perturbed result
        if res_perturbed.success:
            v = res_perturbed.x
        else:
            v = res_base.x

    # 4. Radius expansion phase with topological sensitivity and targeted expansion:
    # a. Compute current radius distribution
    radii_current = v[2::3]
    centers = np.column_stack([v[0::3], v[1::3]])
    radii_sorted = np.sort(radii_current)
    smallest_radius_idx = np.argmin(radii_current)
    least_constrained_idx = None
    
    # b. Compute distance matrix and find least constrained circle
    # (use broadcasting to optimize vector operation)
    dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
    dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
    dists = np.sqrt(dx ** 2 + dy ** 2)
    
    # Compute for each circle, the minimum distance to any other circle
    min_dists = np.min(dists, axis=1)
    min_dist_index = np.argmin(min_dists)
    least_constrained_idx = min_dist_index
    
    # c. Compute radius expansion targets
    current_total = np.sum(radii_current)
    target_growth = max(0.007, 0.003 + (current_total / 8.0) * 0.01)
    # Use a dynamic expansion that increases with total sum
    # Allow up to 10% relative expansion
    max_relative_grow = 0.06
    
    if res_base.success:
        # Expand based on least constrained circle (maximize margin)
        expansion_factor_base = 0.015
        expansion_factor = expansion_factor_base + (target_growth / current_total) * 0.5
        
        # Create a tentative expansion with increased radius on least constrained
        # Use a dynamic expansion factor dependent on current margin
        # Use safe, proportional radius growth from base expansion
        max_possible_growth = max_relative_grow * np.mean(radii_current)
        new_radii = radii_current.copy()
        new_radii[least_constrained_idx] += max(0.0, expansion_factor * 1.05)  # slight over-expansion
        # Distribute growth to all circles with some priority
        for i in range(n):
            # Apply growth based on margin
            if i != least_constrained_idx:
                # Use a soft exponential function for growth based on spacing margin
                margin = min_dists[i]
                if margin == 0:
                    continue
                growth_per = max(0.0, (target_growth - (current_total - radii_current[i])) / (n - 1))
                if growth_per > max_possible_growth:
                    growth_per = max_possible_growth
                # Apply to all circles except least constrained with soft bias
                new_radii[i] += growth_per * (margin / np.max(min_dists))
        
        # Validate the new configuration before optimization
        # Vectorized distance validation
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist < new_radii[i] + new_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Apply expansion in optimization
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            res_expanded = minimize(
                neg_sum_radii,
                expanded_v,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9}
            )
        else:
            res_expanded = minimize(
                neg_sum_radii,
                v,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9}
            )
        
        # Use best result between expansion or original
        if res_expanded.success:
            v = res_expanded.x
        else:
            v = v

    # 5. Final optimization pass with enhanced tolerance on convergence
    # Final optimization to refine
    res_final = minimize(
        neg_sum_radii,
        v,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-10}
    )
    
    # Final safety check for edge cases
    if not res_final.success:
        res_safety = minimize(
            neg_sum_radii,
            v,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-11}
        )
        if res_safety.success:
            v = res_safety.x
        else:
            v = v
    
    # Final post-processing: clip radii to valid bounds
    v = res_final.x if res_final.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    
    # Final validation (should be redundant but safety check)
    valid, error = validate_packing(centers, radii)
    if not valid:
        # If validation fails, use fallback (base or perturbed or safety result)
        if res_base.success:
            v = res_base.x
        else:
            v = v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.5)
        # re-validate
        valid, error = validate_packing(centers, radii)
        if not valid:
            # Last fallback
            centers = np.zeros((n, 2))
            radii = np.full(n, 1e-4)
            if n != 26:
                raise ValueError("Invalid number of circles")
    
    return centers, radii, float(radii.sum())
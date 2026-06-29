import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Enhanced spatial initialization with dynamic grid refinement and non-uniform density
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        spacing_x = 1 / cols + (0.05 * (col % 3)) * (1 if row % 3 == 0 else 0.5)
        spacing_y = 1 / rows + (0.05 * (row % 3)) * (1 if col % 3 == 0 else 0.5)
        x_center = col * spacing_x + 0.5 * spacing_x + np.random.uniform(-0.04, 0.04)
        y_center = row * spacing_y + 0.5 * spacing_y + np.random.uniform(-0.04, 0.04)
        xs.append(x_center)
        ys.append(y_center)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n-length, matches vector

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # We maximize the sum of radii via minimize

    # Define vectorized constraint functions with lambda closures (capture i)
    # Each circle has 4 boundary constraints, and each pair of circles has 1 overlap constraint
    # Total of 4*n boundary + n*(n-1)/2 overlap = 4*26 + (26*25)/2 = 104 + 325 = 429 constraints
    cons = []
    
    # Pre-calculate the constraints for efficient access later
    all_circle_indices = np.arange(n)
    for i in range(n):
        # Bound left (x - r >= 0) => x >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Bound right (x + r <= 1) => 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bound bottom (y - r >= 0) => y >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Bound top (y + r <= 1) => 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                dist_sq = dx*dx + dy*dy
                sum_radii = v[3*i + 2] + v[3*j + 2]
                return dist_sq - sum_radii * sum_radii  # >= 0 as constraint
            cons.append({"type": "ineq", "fun": constraint_func})

    # Optimization stages: initial, refinement, dynamic reconfig, constrained expansion, final
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-10, "eps": 1e-15})
    
    # Stage 1: Dynamic reconfiguration with asymmetric spatial perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        # Create dynamic perturbation based on radii distribution
        perturbation_factor = 0.08 * (radii[np.argmax(radii)] / radii.mean()) * (0.5 + np.random.rand())
        hash_map = np.random.rand(n, 2) * perturbation_factor
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0] * (radii[i] / radii.min())
            perturbed_v[3*i + 1] += hash_map[i, 1] * (radii[i] / radii.min())
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-15})
    
    # Stage 2: Identify and isolate top interacting pair (force geometric reconfiguration)
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized pairwise distance matrix (optimized for performance)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify top two interacting pairs
        interaction_scores = np.zeros(n * (n - 1) // 2)
        interaction_indices = np.zeros(n * (n - 1) // 2, dtype=int)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < (radii[i] + radii[j]) * 0.95:
                    interaction_scores[idx] = dists[i, j] - (radii[i] + radii[j])
                    interaction_indices[idx] = i * n + j
                    idx += 1
        
        # Find the two most interacting pairs
        if len(interaction_scores) > 0:
            # Sort by minimum distance to the interaction (negative for min)
            interaction_indices_sorted = interaction_indices[np.argsort(interaction_scores)]
            i1, j1 = divmod(interaction_indices_sorted[0], n)
            i2, j2 = divmod(interaction_indices_sorted[1], n)
            
            # Create a sub-configuration focusing on those two circles
            # Isolate their positions and apply forced reconfiguration
            # Create a new spatial hash based on their distance and radius disparity
            dx_pair = centers[i1, 0] - centers[j1, 0]
            dy_pair = centers[i1, 1] - centers[j1, 1]
            dist_pair = np.sqrt(dx_pair**2 + dy_pair**2)
            radius_diff = abs(radii[i1] - radii[j1])
            
            # Define spatial reconfiguration with target minimum distance
            target_min_dist = max((radii[i1] + radii[j1]) * 0.9, 0.05)
            # Apply forceful spatial displacement while keeping centers valid
            dx_new = dx_pair * 1.2 + np.random.uniform(-0.02, 0.02)
            dy_new = dy_pair * 1.2 + np.random.uniform(-0.02, 0.02)
            x1, y1 = centers[i1]
            x2, y2 = centers[j1]
            x1_new = x1 + dx_new
            y1_new = y1 + dy_new
            # Ensure no crossing boundaries
            x1_new = max(0.0, min(x1_new, 1.0))
            y1_new = max(0.0, min(y1_new, 1.0))
            x2_new = x2 - dx_new
            y2_new = y2 - dy_new
            x2_new = max(0.0, min(x2_new, 1.0))
            y2_new = max(0.0, min(y2_new, 1.0))
            
            # Update the decision vector with forced displacement
            new_centers = centers.copy()
            new_centers[i1, 0] = x1_new
            new_centers[i1, 1] = y1_new
            new_centers[j1, 0] = x2_new
            new_centers[j1, 1] = y2_new
            
            # Reconstruct v vector with updated centers
            new_v = v.copy()
            new_v[0::3][i1] = x1_new
            new_v[1::3][i1] = y1_new
            new_v[0::3][j1] = x2_new
            new_v[1::3][j1] = y2_new
            
            # Re-optimization focused on this subset with relaxed constraints
            # Apply stricter tolerance for this reconfiguration
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-15})
        
    # Stage 3: Controlled radius expansion on the least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute interaction distances to identify least constrained circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify min distances to all others
        min_dists = np.min(dists, axis=1)
        # Find the circle with the maximum minimum distance (least constrained)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute potential for expansion based on spatial constraints
        current_total = radii.sum()
        # Target expansion of 0.006 with prioritization of least constrained
        expansion_factor = 0.006 * (min_dists[least_constrained_idx] / min_dists.mean())
        expansion_amount = expansion_factor / (n - 1)
        
        # Apply expansion to the least constrained while enforcing non-overlap
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_amount * 1.2  # Small over-expansion
        
        # Apply stochastic expansion to other circles for diversified growth
        for i in range(n):
            if i != least_constrained_idx:
                stochastic_factor = 1.0 + 0.1 * np.random.rand()  # Introduce variability
                new_radii[i] += expansion_amount * stochastic_factor
        
        # Validate the new_radius configuration
        while True:
            # Recreate v with new radii
            new_v = v.copy()
            new_v[2::3] = new_radii
            
            # Validate the new configuration
            valid = True
            centers_new = np.column_stack([new_v[0::3], new_v[1::3]])
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers_new[i, 0] - centers_new[j, 0]
                    dy = centers_new[i, 1] - centers_new[j, 1]
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
                new_radii = radii + (new_radii - radii) * 0.99  # Reduce by 1%
        
        # Re-optimization to stabilize the configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-15})
    
    # Final refinement with tightened tolerances
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Enforce strict bounds checking
        for i in range(n):
            if v[3*i] - radii[i] < -1e-12:
                v[3*i] = max(min(v[3*i], 1.0), 0.0)
            if v[3*i + 1] - radii[i] < -1e-12:
                v[3*i + 1] = max(min(v[3*i + 1], 1.0), 0.0)
            v[3*i + 2] = np.clip(v[3*i + 2], 1e-6, 0.5)
        
        # Final optimization pass
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-15})
    
    # Final output preparation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
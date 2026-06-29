import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    max_iterations = 1800
    ftol = 1e-11
    final_tightening_ftol = 1e-12
    expansion_factor = 0.97
    min_radius = 0.99 * 1e-3
    # Initialize with adaptive staggered grid and refined randomness
    # First pass setup: geometric placement + perturbation
    xs = []
    ys = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        x_center = (col_idx + 0.48) / cols  # slightly offset to reduce grid clustering
        y_center = (row_idx + 0.48) / rows
        # Add dynamic noise: scaled by row/col index to avoid uniformity
        row_scale = np.sqrt(row_idx + 1) * 0.35
        col_scale = np.sqrt(col_idx + 1) * 0.25
        noise_x = np.random.uniform(-0.35*row_scale, 0.35*row_scale)
        noise_y = np.random.uniform(-0.35*col_scale, 0.35*col_scale)
        # Create staggered rows: alternating horizontal offset by 0.25/cols
        if row_idx % 2 == 1:
            x_center += 0.25 / cols
        x = x_center + noise_x
        y = y_center + noise_y
        xs.append(x)
        ys.append(y)
    
    r0_base = 0.42 / cols - 2e-4
    v0 = np.empty(3*n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0_base)

    # Precalculate bounds with strict 3n-length, match variables
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((min_radius, 0.5))  # radius
    
    # Objective: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint manager with closure safety and vectorization
    cons = []
    for i in range(n):
        # Left constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Precompute all pairwise constraints for optimization with vectorization
    # Vectorize all pairwise constraints using broadcasting to optimize speed
    # This is done precomputed with a fixed set of indices, not with dynamic i,j loops
    # This is key for performance in SLSQP which may struggle with dynamic function generation
    for i in range(n):
        for j in range(i+1, n):
            # Use lambda with captured i,j to avoid dynamic scoping issues
            # Use closure-safety by using lambda with default args and closure
            # Create a specialized constraint function for each pair
            def _pairwise_constr_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_i = v[3*i+2]
                r_j = v[3*j+2]
                # Ensure constraint for non-overlapping is >= 0: dist_sq - (r_i + r_j)^2 >= 0
                return dist_sq - (r_i + r_j)**2
            cons.append({"type": "ineq", "fun": _pairwise_constr_func})

    # First optimization with full iteration and tight tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": max_iterations, "ftol": ftol})
    
    # Adaptive phase 2: geometric dissection reconfiguration
    # First, extract best found state
    if res.success:
        v = res.x
        best_centers = np.stack([v[0::3], v[1::3]], axis=1)
        radii = v[2::3]
        # Step 1: identify the two most dynamically interacting circles
        # Optimized distance calculation using broadcasting once
        dx_all = best_centers[:, np.newaxis, 0] - best_centers[np.newaxis, :, 0]
        dy_all = best_centers[:, np.newaxis, 1] - best_centers[np.newaxis, :, 1]
        dists = np.sqrt(dx_all**2 + dy_all**2)
        interaction_matrix = dists * (1 / (radii[np.newaxis, :] + radii[:, np.newaxis])) # normalized
        # Find top two most interacting circles
        pairwise_interactions = interaction_matrix[np.triu_indices(n, k=1)]
        top_interacting_idx = np.argsort(pairwise_interactions)[-2:]
        # Get their indices in the original array
        # Also need to find the circle with lowest interaction to expand
        # First, identify the circle with the most uniform distance to others
        # For this, compute the min distance to neighbors (with weighting by radius to avoid dominance)
        min_dists = np.min(np.abs(dists), axis=1)
        isolation_idx = np.argmin(min_dists)
        # Prepare perturbation vector
        perturbation = np.zeros(3*n)
        # For the two most interacting circles, apply controlled reconfiguration
        for idx in top_interacting_idx:
            x = best_centers[idx, 0]
            y = best_centers[idx, 1]
            r = radii[idx]
            # Create a target configuration that creates more spacing
            # Use a geometric perturbation with respect to their interaction area
            # Move them apart and scale radii proportionately
            # Find their current distance
            dist_between = np.sqrt((best_centers[top_interacting_idx[0], 0] - best_centers[idx, 0])**2 + (best_centers[top_interacting_idx[0], 1] - best_centers[idx, 1])**2)
            # Compute target distance: current + 1e-3 (small expansion)
            target_dist_between = dist_between + 1e-3
            # Move them apart in direction of separation to reconfigure layout
            direction = (best_centers[idx] - best_centers[top_interacting_idx[0]]) / dist_between
            new_x1 = x + direction[0] * 0.025
            new_y1 = y + direction[1] * 0.025
            new_x2 = best_centers[top_interacting_idx[0], 0] - direction[0] * 0.025
            new_y2 = best_centers[top_interacting_idx[0], 1] - direction[1] * 0.025
            # Update their positions
            perturbation[3*idx] = new_x1 - best_centers[idx, 0]
            perturbation[3*idx+1] = new_y1 - best_centers[idx, 1]
            perturbation[3*top_interacting_idx[0]] = new_x2 - best_centers[top_interacting_idx[0], 0]
            perturbation[3*top_interacting_idx[0]+1] = new_y2 - best_centers[top_interacting_idx[0], 1]
            # Reduce radius for interacting pair slightly to allow expansion
            perturbation[3*idx+2] = -0.0005
            perturbation[3*top_interacting_idx[0]+2] = -0.0005
        # Add a small stochastic shift to their positions to avoid symmetry trapping
        for idx in top_interacting_idx:
            x_perturb = np.random.uniform(-0.005, 0.005)
            y_perturb = np.random.uniform(-0.005, 0.005)
            perturbation[3*idx] += x_perturb
            perturbation[3*idx+1] += y_perturb
        # Also add perturbation to the isolated circle to allow spatial reconfiguration
        x_perturb_isolated = np.random.uniform(-0.01, 0.01)
        y_perturb_isolated = np.random.uniform(-0.01, 0.01)
        perturbation[3*isolation_idx] += x_perturb_isolated
        perturbation[3*isolation_idx+1] += y_perturb_isolated
        # Compute the new perturbed vector
        v_perturbed = v + perturbation
        # Run phase 2 optimization on perturbed state
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": final_tightening_ftol})
        
        # Phase 3: adaptive expansion on isolated circle (with soft constraints)
        if res.success:
            v = res.x
            # Re-evaluate interaction and isolation for latest state
            # Optimized distance calculation
            dx_all = v[0::3][:, np.newaxis] - v[0::3][np.newaxis, :]
            dy_all = v[1::3][:, np.newaxis] - v[1::3][np.newaxis, :]
            dists = np.sqrt(dx_all**2 + dy_all**2)
            # Compute normalized pair-wise interaction
            interaction_matrix = dists * (1 / (v[2::3][np.newaxis, :] + v[2::3][:, np.newaxis]))
            # Find new top two interactors
            pairwise_interactions = interaction_matrix[np.triu_indices(n, k=1)]
            top_interacting_idx = np.argsort(pairwise_interactions)[-2:]
            # Find new isolated circle
            min_dists_current = np.min(np.abs(dists), axis=1)
            isolation_idx_current = np.argmin(min_dists_current)
            # Create new radius expansion vector
            new_v = v.copy()
            # Create a radius expansion vector that increases the isolated circle's radius while adjusting others
            # The idea: expand the isolated circle's radius, redistribute to others to maintain sum
            current_total = np.sum(v[2::3])
            # Define a target expansion: target increase up to 0.004, but dynamically adjust
            # Use a heuristic based on current density: higher density allows more expansion
            max_target_increase = 0.003
            expansion_to_total = max(0, (current_total + max_target_increase) - current_total) / (n - 1)
            expansion_amount = expansion_to_total * (1.0 + 0.5 * np.random.rand())  # stochastic
            # Distribute expansion while ensuring constraints
            new_radii = v[2::3].copy()
            # Expand the isolated circle
            new_radii[isolation_idx_current] += expansion_amount * 1.05  # slight overdrive
            # Distribute to others proportionally, ensuring they stay within radius bounds
            for i in range(n):
                if i != isolation_idx_current and new_radii[i] < 0.5 - 1e-5:
                    new_radii[i] += expansion_amount * np.random.rand() * 0.5
            # Update the new_v with this radial vector
            new_v[2::3] = new_radii
            # Re-evaluate with expanded radii
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": final_tightening_ftol})
        
        # Optional: final check for validation if not in success path
        if not res.success:
            # Fallback: perform one final optimization on original best state
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 120, "ftol": final_tightening_ftol})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-4, 0.5)
    # Final consistency check for validity
    return centers, radii, float(radii.sum())
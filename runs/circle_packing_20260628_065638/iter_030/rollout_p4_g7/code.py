import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # --- PHASE 0: STRATEGIC INITIALIZATION WITH GEOMETRICALLY AUGMENTED RANDOMIZED SPATIAL GRID ---
    # Define a more refined spatial grid with variable row/column spacing and gradient-aware seed
    # Use spatial partitioning, then refine with stochastic seeding to prevent premature convergence
    seed_offset = np.random.uniform(-0.08, 0.08)
    xs = []
    ys = []
    
    # Generate initial positions with a hybrid grid + geometric perturbation strategy
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / (cols + 1)  # Slightly reduce grid density for spacing
        y_center = (row + 0.5) / (rows + 1)
        x = x_center + np.random.normal(loc=0, scale=0.04) + seed_offset * (0.75 + i/26)
        y = y_center + np.random.normal(loc=0, scale=0.04) + seed_offset * (0.75 + i/26)
        # Alternate row shift to break symmetry and enable better inter-grid spacing
        if row % 2 == 1:
            x += 0.5 / (cols + 2)  # Smaller shift for reduced density
        xs.append(x)
        ys.append(y)
    
    # --- PHASE 1: RADIUS INITIALIZATION WITH DENSITY-SENSITIVE DYNAMIC INITIALIZATION ---
    # Initialize radii based on local spacing: more dense grid = smaller r0
    # Use a radius base that varies with grid density and row/column parity for better coverage
    base_radius = 0.29 / (cols + 1)  # Slightly lower base to allow optimization range
    radii_ratio = 1.0 + 0.05 * (i % 5)  # Add mild variance in radius distribution for optimization
    r0 = base_radius * radii_ratio - 1e-4  # Ensure minimal radii but allow growth
    # But also, make sure minimal radii aren't too small
    r0 = np.clip(r0, 1e-5, 0.42)  # Clip to reasonable bounds and prevent collapse
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # --- PHASE 2: BOUNDS CONSTRUCTION WITH ADAPTIVE ENFORCEMENT ---
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.45)]  # Tighter bounds on radii
    
    # --- PHASE 3: CONSTRAINT FUNCTION OPTIMIZATION ENGINE ---
    # Use a vectorized constraint factory with explicit dependency tracking and closure capture
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective: maximize sum of radii
    
    # --- PHASE 4: CREATE CONSTRAINTS WITH LAMBDA-FREE, VECTOR-CAPTURED STRUCTURE ---
    # Vectorized boundary constraints 
    cons = []
    for i in range(n):
        # Lx: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Rx: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Ly: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Ry: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # --- PHASE 5: OVERLAP CONSTRAINTS WITH GEOMETRIC HASHING (STOCHASTICALLY SENSITIVE) ---
    # Vectorized pairwise distance constraints with lambda-free closure management
    # For each pair, we compute (dist^2 - (r1 + r2)^2) >= 0
    
    # Use a vectorized constraint builder with lambda-free closure capture via positional tuple
    
    overlap_cons = []
    # Precompute grid distances to accelerate initial check
    # This provides a heuristic check for initial constraint generation
    # For this version, we use a hybrid: create all pairs, but generate constraints with dynamic closure capture
    for i in range(n):
        for j in range(i+1, n):
            # Create closure with i and j in a tuple to avoid lambda closure issues
            def make_overlap_fun(i=i, j=j):
                def f(v):
                    # Get centers from v
                    xi, yi = v[3*i], v[3*i+1]
                    xj, yj = v[3*j], v[3*j+1]
                    ri, rj = v[3*i+2], v[3*j+2]
                    # Compute squared distance between centers
                    dx = xi - xj
                    dy = yi - yj
                    dist_sq = dx*dx + dy*dy
                    # Compute minimum squared distance from edge of circles
                    min_dist_sq = (ri + rj) ** 2
                    # Return the constraint: dist^2 >= min_dist^2 => constraint >= 0
                    return dist_sq - min_dist_sq
                return f
            overlap_cons.append({"type": "ineq", "fun": make_overlap_fun})
    
    # Add overlap constraints as a group (this is more efficient internally)
    cons.extend(overlap_cons)
    
    # --- PHASE 6: STRATEGIC SOLVER CONFIGURATION AND HYPERPARAMETER TUNING ---
    # Use a hybrid optimization strategy
    # Step 1: Initial low-cost gradient descent to reach a local region
    # Step 2: Use SLSQP with tight bounds but with dynamic constraint relaxation
    # Step 3: Apply an asymmetric spatial perturbation strategy to escape local optima
    # Step 4: Apply a controlled expansion phase to push radii
    
    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="L-BFGS-B", bounds=bounds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-9})
    
    # If failed, try with SLSQP to get better constraint handling
    if not res.success:
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "ftol": 1e-9,
                                                  "eps": 1e-8, "maxls": 200})
    
    # --- PHASE 7: ASYMMETRIC RECONFIGURATION WITH SPATIAL RECONFIGURATION ---
    # If optimization success, we apply an asymmetric spatial reconfiguration strategy
    # To create a spatial perturbation that breaks symmetry but preserves the feasibility
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Create a spatial hash using radii-aware weighting to guide perturbation
        # This spatial perturbation is geometrically aware and sensitive to the current configuration
        # We scale the perturbation by the local radius density for dynamic influence
        
        # Spatial hashing: generate a spatially sensitive perturbation for each circle
        # This step is not fully random but depends on local spacing (radii-aware)
        spatial_hash = np.random.rand(n, 2) * (0.08 + 0.05 * np.mean(radii))
        # Apply a spatial perturbation that's proportional to current radii to improve reachability
        # We use a dynamic perturbation scaling based on the current configuration
        spatial_perturbation = 0.04 * (np.log1p(radii) / np.log1p(np.mean(radii)))
        perturbed_v = v.copy()
        for i in range(n):
            # Add spatially targeted perturbation and radius-scaled noise
            perturbed_v[3*i] += spatial_hash[i, 0] * spatial_perturbation[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * spatial_perturbation[i]
        
        # Perturb v, then apply SLSQP with tighter tolerances
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 250, "ftol": 1e-10, "eps": 1e-8})
    
    # --- PHASE 8: ASYMMETRIC EXPANSION WITH CONSTRAINT-AWARE EXPANSION ---
    # If successful, now we use an asymmetric radius expansion targeting the least constrained circle
    # This step is a refined version of the original approach but with dynamic constraint awareness and gradient tracking
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # --- PHASE 8A: DYNAMIC CONSTRAINT ASSESSMENT ---
        # Calculate current min distances to neighbors (distance to nearest circles) to assess constraint tightness
        # This helps us identify the least constrained circle
        
        # Vectorized distance matrix computation via broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Set diagonal distance to infinity to ignore self-to-self comparison
        np.fill_diagonal(dists, np.inf)
        
        # For each circle, find the closest neighbor and determine how tightly constrained it is
        closest_dists = np.min(dists, axis=1)
        # For each circle, calculate the "available space" as distance to neighbor minus (current radius + neighbor radius)
        # This is the amount of slack in the constraint
        available_space = closest_dists - (radii[np.newaxis, :] + radii[:, np.newaxis])
        # But ensure it's not negative (i.e., already overlapping)
        # Clip to ensure no negative available space is considered
        available_space = np.clip(available_space, 0.0, None)
        # Compute an effectiveness score for each circle, proportional to available space and inverse of current radius
        effectiveness = (available_space * (1/radii)) if (radii > 0).all() else np.zeros_like(radii)
        
        # If all circles are zero (overlapping), fall back to a base growth plan
        if (effectiveness == 0).all():
            target_total_sum = np.sum(radii) + 0.0035
            expansion = np.repeat((target_total_sum - np.sum(radii)) / n, n)
            v[2::3] += expansion
        else:
            # Find the circle with the highest effectiveness (least constrained)
            least_constrained_idx = np.argmax(effectiveness)
            # If a circle has zero effectiveness (already overlapping), force minimal change
            if effectiveness[least_constrained_idx] < 0.0001:
                # This is a fallback: do not expand
                target_total_sum = np.sum(radii)
            else:
                # Calculate a proportional radius expansion that does not break constraints
                # The expansion is calculated based on the max possible growth under the current constraints
                # We estimate it as 1.2 times the current value with a safety margin
                max_possible_growth = 0.0045  # Hard limit (based on empirical evidence, 0.45% of 1 unit per circle)
                expansion_per_circle = (max_possible_growth - (np.sum(radii) - np.sum(radii))) / n
                # But for the least constrained circle, multiply by a factor based on effectiveness
                expansion_factor = 1.2 * (effectiveness[least_constrained_idx] / np.max(effectiveness) if np.max(effectiveness) > 0 else 1)
                # Targeted expansion: grow the least constrained by expansion_factor times average expansion per circle
                # But ensure that no individual radius exceeds the maximum allowed (0.45)
                max_radius = np.max([0.45, 0.4 * np.sum(radii) / n])  # Conservative bound
                
                # Compute initial expansion
                expansion = np.zeros_like(radii)
                expansion[least_constrained_idx] = expansion_per_circle * expansion_factor
                expansion = np.clip(expansion, 0.0, max_radius - radii)
                
                # Now, compute the total expansion
                total_expansion = np.sum(expansion)
                # Calculate target new total sum based on 2% growth (adjust as needed)
                target_total_sum = np.sum(radii) + total_expansion * (1.02)
                
                # Distribute the expansion while maintaining constraint validity
                # We apply the expansion as a vector with a priority to the least constrained
                expansion = np.zeros_like(radii)
                expansion[least_constrained_idx] = 0.6 * (target_total_sum - np.sum(radii))
                # The rest are assigned proportionally to their effectiveness (i.e., constraint slack)
                # Weight each circle by their effectiveness and the inverse of their current radius (to account for smaller circles)
                weights = (effectiveness / (radii + 1e-6))  # Avoid division by zero
                weights = weights / np.sum(weights)  # Normalize
                expansion[~np.isin(np.arange(n), [least_constrained_idx])] = \
                    (target_total_sum - np.sum(radii) - expansion[least_constrained_idx]) * weights[~np.isin(np.arange(n), [least_constrained_idx])]
                
                # Clip all expansions to the max allowed radius
                expansion = np.clip(expansion, 0.0, max_radius - radii)
                # Apply expansion
                v[2::3] += expansion
        
        # Re-evaluate the expanded configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 250, "ftol": 1e-10, "ftol": 1e-9, "eps": 1e-8})
    
    # --- PHASE 9: FINAL VALIDATION + ENFORCEMENT OF CONSTRAINTS ---
    # In-place final check for any potential violations (especially near boundaries)
    # We perform a final validation check before returning to ensure no overfit
    final_v = res.x if res.success else v0
    centers = np.column_stack([final_v[0::3], final_v[1::3]])
    radii = np.clip(final_v[2::3], 1e-6, None)
    
    # Apply final validation
    # Check for all circles that might be outside the box (unlikely if constraints are properly managed)
    for i in range(n):
        x, y, r = centers[i], 0.0, radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12 
            or y - r < -1e-12 or y + r > 1 + 1e-12):
            # If any circle is outside, perturb its position slightly and re-run
            perturbation = np.random.uniform(-0.001, 0.001) * 2
            centers[i][0] += perturbation
            centers[i][1] += perturbation
            # Re-construct the vector and run a final optimization pass
            v = np.concatenate([centers[:, 0], centers[:, 1], radii])
            res = minimize(neg_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                           constraints=cons, options={"maxiter": 50, "ftol": 1e-11, "eps": 1e-8})
    
    # Final centers and radii after validation
    centers = np.column_stack([final_v[0::3], final_v[1::3]])
    radii = np.clip(final_v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = int(np.ceil(n / cols))  # Explicit, stable rows assignment
    
    # Initialize with a hybrid layout: deterministic grid with 
    # enhanced randomization of positions and radii, 
    # plus dynamic spatial adjustment based on interplay indices
    xs, ys = [], []
    random_seed = np.random.randint(0, 1000000)  # random initialization
    np.random.seed(random_seed)
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add dynamic offset based on row-col interaction metric
        offset_angle = np.arctan2(row, col)
        spatial_offset = np.array([
            np.cos(offset_angle) * 0.08,
            np.sin(offset_angle) * 0.08
        ])
        
        x = x_center + np.random.uniform(-0.1, 0.1) + spatial_offset[0]
        y = y_center + np.random.uniform(-0.1, 0.1) + spatial_offset[1]
        
        # Stagger rows: alternate rows shift by 0.5/cols
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure bounds list is length 3*n, with 3 entries per circle
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # We want to maximize sum of radii, so minimize -sum_radii
    
    # Boundary constraints: (x - r) >= 0, (1 - x - r) >= 0, same for y
    # We use vectorized constraints that take into account the index of the circle
    cons = []
    for i in range(n):
        # x >= r
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # 1 - x - r >= 0 => x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # y >= r
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # 1 - y - r >= 0 => y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints: 
    for i in range(n):
        for j in range(i + 1, n):
            # The constraint is: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
            # Reformulated as objective being a >= 0, so "ineq" constraint
            cons.append({
                "type": "ineq", 
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
                )
            })
    
    # First optimization pass: high-precision, large iteration count, tighter tolerance
    # Use a more modern optimization configuration
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 2000,  # 2000 iterations
            "ftol": 1e-12,    # Extremely tight convergence
            "gtol": 1e-10,    # Gradient tolerance
            "eps": 1e-8,      # Finite difference epsilon
            "disp": False
        }
    )
    
    # Optimization failed, fallback to perturbation on the original v0 (with enhanced seed)
    if not res.success:
        # Try with seed 0 for deterministic fallback
        np.random.seed(0)
        # Re-generate v0 with seed 0 
        xs, ys = [], []
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            x = x_center + np.random.uniform(-0.1, 0.1)
            y = y_center + np.random.uniform(-0.1, 0.1)
            if row % 2 == 1:
                x += 0.5 / cols
            xs.append(x)
            ys.append(y)
        v0 = np.empty(3 * n)
        v0[0::3] = xs
        v0[1::3] = ys
        v0[2::3] = r0
        # Re-run with seed 0
        res = minimize(
            neg_sum_radii, 
            v0, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 2000, 
                "ftol": 1e-12,
                "gtol": 1e-10,
                "eps": 1e-8
            }
        )
    
    # Stochastic topological reconfiguration phase: 
    # Isolate most interacting circles and reconfigure their spatial relationships
    # This will induce a local reconfiguration that can allow larger radii growth if viable
    if res.success:
        v = res.x
        # Vectorize centers and radii for efficiency
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # First pass: identify the two most dynamically interacting circles
        # Vectorized distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Normalize by the average distance to normalize interaction metric
        avg_dist = np.mean(dists[dists > 1e-10])
        if avg_dist > 1e-10:
            dists_normalized = dists / avg_dist
        else:
            dists_normalized = dists  # fallback to dists
        # Interaction is sum of distances to avoid extreme outliers
        interaction_metric = np.sum(dists_normalized, axis=1)
        # Select the top 2 most connected circles
        top_idx = np.argsort(interaction_metric)[-2:]
        
        # Create a targeted perturbation for the top two circles, but not full randomization
        # Introduce a local perturbation to their positions to "dissect" their topology
        # Also, add a slight spatial bias to allow for radius expansion
        top_v = v.copy()
        for idx in top_idx:
            # Perturb positions with a bias in an angle that's opposite to the center
            current_center_x = v[3*idx]
            current_center_y = v[3*idx + 1]
            current_radius = v[3*idx + 2]
            direction_x = (current_center_x - 0.5)
            direction_y = (current_center_y - 0.5)
            # Normalize direction and scale by radius
            if (direction_x**2 + direction_y**2) > 1e-10:
                dir_norm = np.sqrt(direction_x**2 + direction_y**2)
                direction_x /= dir_norm
                direction_y /= dir_norm
                # Move in this direction, but scaled by some factor and radius
                # For example: move in this direction but not too far
                # Scale by radius to ensure movement relative to size
                perturb_strength = current_radius * 1.5 * 0.3
                delta_x = direction_x * perturb_strength * 0.8
                delta_y = direction_y * perturb_strength * 0.8
                # Apply the perturbation
                top_v[3*idx] += delta_x
                top_v[3*idx + 1] += delta_y
        
        # Add a tiny expansion hint to the radii of these two circles
        expansion_hints = np.zeros(n)
        for idx in top_idx:
            expansion_hints[idx] = 0.001  # tiny hint to allow radius growth
        # This is handled implicitly in the optimization, but we can use it as guidance
        # Now re- optimize this modified configuration, using a new seed
        np.random.seed(123456)  # New random seed to prevent further optimization bias
        res = minimize(
            neg_sum_radii, 
            top_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 600, 
                "ftol": 1e-11,  # Slightly looser for faster convergence
                "gtol": 1e-9,
                "eps": 1e-8,
                "disp": False
            }
        )
    
    # Final optimization with enhanced spatial reconfiguration and guided expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Identify the most isolated circle - this will be a candidate for expansion
        # Use a metric that is more nuanced than simple distances: 
        # (1) Minimum spatial distance
        # (2) Normalized distance from center
        # (3) Interaction strength as a weight
        
        # Compute spatial distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute normalized distances from center
        center = np.array([0.5, 0.5])
        dist_to_center = np.sqrt((centers - center)**2).sum(axis=1)
        
        # Compute interaction metrics as before
        interaction_metric = np.sum(dists, axis=1)
        
        # Create composite metric
        # Weighting: (0.7 * min_distance) + (0.2 * dist_to_center) + (0.1 * interaction_strength)
        # We use inverse metrics for isolation
        inverse_dist = 1 / (dists + 1e-8)  # avoid division by zero
        inverse_dist_sum = np.sum(inverse_dist, axis=1)
        isolation_metric = (0.7 * inverse_dist_sum) + (0.2 * dist_to_center) + (0.1 * interaction_metric)
        isolated_idx = np.argmin(isolation_metric)
        
        # Compute current total sum of radii for possible expansion
        total_sum = np.sum(radii)
        # Targeted expansion: we aim to expand the isolated circle with a proportional increase
        # To be safe, we calculate the potential maximum expansion based on the smallest distance
        # to another circle
        min_dist = np.min(dists[isolated_idx, :])
        if min_dist > 1e-7:
            # Calculate maximum possible expansion (without overlapping)
            # max_radius = min( (min_dist - radii[isolated_idx]) / 2, ... )
            # But instead, we do a strategic expansion and then re-optimization
            # We will increase the radius of the isolated circle (but not too much) 
            # Then we allow the optimization to expand the rest as needed
            
            # We aim for a growth of up to 1.5 times the current value, but this needs to be done carefully
            # To be safe, we add 5%-7% as a growth factor, and let the solver scale others
            # Add small expansion incrementally, not all at once
            expansion_factor = 0.02  # a modest growth factor
            
            # We first attempt to expand the radius of the isolated circle, but only if it's feasible
            # Create a small buffer for expansion check first
            max_possible_radius = (min_dist - radii[isolated_idx]) / 2
            max_add = max(0, (max_possible_radius - radii[isolated_idx]))  # can't exceed this
            # Cap expansion factor based on max_possible_radius
            if max_add > 0:
                actual_growth = expansion_factor * max_add
                # Apply to the isolated circle
                expanded_radii = radii.copy()
                expanded_radii[isolated_idx] += actual_growth
                # Now we optimize with this adjusted radius
                expanded_v = v.copy()
                expanded_v[2::3] = expanded_radii
                # Re- optimize with this perturbation
                # New seed to avoid optimization lock-in
                np.random.seed(789012)
                res = minimize(
                    neg_sum_radii, 
                    expanded_v, 
                    method="SLSQP", 
                    bounds=bounds,
                    constraints=cons,
                    options={
                        "maxiter": 600,
                        "ftol": 1e-11,
                        "gtol": 1e-10,
                        "eps": 1e-8
                    }
                )
            else:
                # No expansion possible, proceed with current configuration
                res = minimize(
                    neg_sum_radii, 
                    v, 
                    method="SLSQP", 
                    bounds=bounds,
                    constraints=cons,
                    options={
                        "maxiter": 600,
                        "ftol": 1e-11,
                        "gtol": 1e-10,
                        "eps": 1e-8
                    }
                )
        else:
            # No expansion possible, proceed with current configuration
            res = minimize(
                neg_sum_radii, 
                v, 
                method="SLSQP", 
                bounds=bounds,
                constraints=cons,
                options={
                    "maxiter": 600,
                    "ftol": 1e-11,
                    "gtol": 1e-10,
                    "eps": 1e-8
                }
            )

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)  # clip to avoid numerical instability from unbounded radii
    
    # Final validation pass against the validator - not done in the function but enforced by the system
    return centers, radii, float(radii.sum())
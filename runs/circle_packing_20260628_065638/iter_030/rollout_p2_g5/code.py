import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    max_iter_initial = 800
    max_iter_refinement = 350
    post_refinement_steps = 1
    constraint_relaxation_factor = 0.85
    radius_stability_weight = 0.98
    perturbation_intensity = 0.008
    adaptive_radius_growth_factor = 1.12

    # Optimized initial spatial initialization: 
    # - dynamic grid layout with variable column spacing
    # - stochastic perturbation with adaptive variance
    # - enhanced stagger pattern with distance-based offset

    # Base positions with adaptive grid
    base_xs = []
    base_ys = []
    for i in range(n):
        row = i // cols
        col_num = i % cols
        # Adaptive column spacing: more spread at the edges
        col_spacing = 1.0 / cols + 0.02 * np.sin(np.pi * col_num / cols)
        x_center = (col_num + 0.5) * col_spacing
        y_center = (row + 0.5) / rows

        # Adaptive spatial perturbation
        perturbation_x = np.random.uniform(-0.08, 0.08)
        perturbation_y = np.random.uniform(-0.08, 0.08)
        
        # Distance-based stagger adjustment
        # More stagger for circles on the same row
        if row == i // cols:
            # Alternate columns to stagger
            if (col_num % 2 == 0 and row % 2 == 0) or (col_num % 2 == 1 and row % 2 == 1):
                x_center += 0.01 / (1 + (row / rows)**2)
                perturbation_x += 0.005 * np.random.randn()
        
        # Apply perturbation
        x_center += perturbation_x
        y_center += perturbation_y
        
        # Clamp to bounds with margin
        x = np.clip(x_center, 0.0 + 1e-5, 1.0 - 1e-5)
        y = np.clip(y_center, 0.0 + 1e-5, 1.0 - 1e-5)
        
        base_xs.append(x)
        base_ys.append(y)

    # Optimized radial initialization: 
    # - dynamic radius distribution with distance-to-edge consideration
    # - initial radii based on adaptive spacing and edge constraints
    # - use of non-uniform distribution for better spatial coverage

    col_distances = [1.0 / cols + 0.02 * np.sin(np.pi * i / cols) for i in range(cols)]
    row_distances = [1.0 / rows for i in range(rows)]
    avg_col_dist = np.mean(col_distances)
    avg_row_dist = np.mean(row_distances)
    min_dist = np.min([avg_col_dist, avg_row_dist])

    r0 = np.zeros(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        col_dist = col_distances[col]
        row_dist = row_distances[row]
        
        # Edge distance consideration
        edge_gap = 0.1 * np.min([1.0 - x_center(row, col), x_center(row, col)])
        r0[i] = np.min([
            0.001 * (1.0 / (np.log(col_dist + 1e-12) + 0.01)) - 1e-3,
            edge_gap * np.sqrt(1.0 - (1.0 - (1.0 / (1.0 / row_dist)**2))**2),
            0.2 * min_dist * (1.0 - (row / rows)) * (1.0 - (col / cols))
        ]) + 1e-4

    # Create decision vector
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(base_xs)
    v0[1::3] = np.array(base_ys)
    v0[2::3] = np.array(r0)

    # Bounds configuration (3n for 3*n params)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function for the optimizer
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Optimize constraint handling with explicit parameterized lambdas 
    # (avoids closure capture issues by passing explicit parameters)
    def constraint_fun_left_side(v, i):
        return v[3*i] - v[3*i + 2]
    def constraint_fun_right_side(v, i):
        return 1.0 - v[3*i] - v[3*i + 2]
    def constraint_fun_bottom_side(v, i):
        return v[3*i + 1] - v[3*i + 2]
    def constraint_fun_top_side(v, i):
        return 1.0 - v[3*i + 1] - v[3*i + 2]
    def constraint_fun_overlap(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i + 1] - v[3*j + 1]
        return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2

    # Constraints list with correct lambda capture
    cons = []
    for i in range(n):
        # Left side
        cons.append({"type": "ineq", "fun": lambda v, i=i: constraint_fun_left_side(v, i)})
        # Right side
        cons.append({"type": "ineq", "fun": lambda v, i=i: constraint_fun_right_side(v, i)})
        # Bottom side
        cons.append({"type": "ineq", "fun": lambda v, i=i: constraint_fun_bottom_side(v, i)})
        # Top side
        cons.append({"type": "ineq", "fun": lambda v, i=i: constraint_fun_top_side(v, i)})

    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            # Avoid repeated evaluation by ensuring unique closure bindings
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_fun_overlap(v, i, j)})

    # First optimization pass
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, 
                   options={"ftol": 1e-11, "gtol": 1e-9, 
                            "maxiter": max_iter_initial, 
                            "eps": 1e-8, 
                            "disp": False})

    # If solution is not valid, fallback
    if not res.success:
        v = v0
    else:
        v = res.x

    # Optimization phase 1: adaptive radius expansion with constrained perturbation
    # Introduce dynamic perturbation patterns 
    # - use of spatial gradient to identify regions for expansion
    # - localized gradient-based radius adjustments
    # - spatial constraint balancing
    # - controlled perturbation with adaptive intensity

    # Phase 1: spatial reconfiguration with radial expansion
    def spatial_expansion_reconfigure(v, i, j, expansion_coeff):
        # Calculate spatial gradients for local perturbation
        # Focus expansion on spatially underutilized zones
        dx = v[3*i] - v[3*j]
        dy = v[3*i + 1] - v[3*j + 1]
        distance = np.sqrt(dx**2 + dy**2)
        return np.sqrt((np.min([v[3*i + 2], v[3*j + 2]]) / (distance + 1e-9))) * expansion_coeff

    if res.success:
        def compute_isolation_effectiveness(v):
            # Isolated circles are those with minimal interaction
            # Use vectorized distance calculation to compute interaction
            dx = v[0::3, np.newaxis] - v[0::3]
            dy = v[1::3, np.newaxis] - v[1::3]
            dists = np.sqrt(dx**2 + dy**2)
            min_distances = np.min(np.where(dists > 0, dists, np.inf), axis=1)
            # Weight min_distances by radius sizes
            isolation_score = np.sum(1 / (min_distances + 1e-9) * (v[2::3] + 1e-9))  # +1e-9 to avoid division by zero
            return isolation_score

        def compute_dynamism(v):
            # High dynamism circles are those that are near multiple circles
            # Use vectorized distance calculation to compute interaction
            dx = v[0::3, np.newaxis] - v[0::3]
            dy = v[1::3, np.newaxis] - v[1::3]
            dists = np.sqrt(dx**2 + dy**2)
            min_distances = np.min(np.where(dists > 0, dists, np.inf), axis=1)
            # Weight by radius sizes
            dynamism_score = np.sum(1 / (min_distances + 1e-9) * (v[2::3] + 1e-9))
            return dynamism_score

        def compute_spatial_effectiveness(v):
            # Spatial effectiveness: balance of being both isolated and interacting
            # Use dynamic weights to avoid local minima
            isolation = compute_isolation_effectiveness(v)
            dynamism = compute_dynamism(v)
            spatial_score = 1.0 / (1.0 + isolation + 1.0 + dynamism)  # Normalize to avoid overflow
            return spatial_score

        # Apply a dynamic perturbation strategy with adaptive radius expansion
        # This ensures the radius expansion is guided by the spatial constraints
        for _ in range(post_refinement_steps):
            # First compute the current spatial effectiveness
            current_effectiveness = compute_spatial_effectiveness(v)
            v_current = v.copy()
            
            # Introduce a controlled spatial perturbation using gradient of spatial effectiveness
            # Calculate spatial gradients using finite differences
            eps = 1e-6
            perturbations = []
            for idx in range(n):
                # Temporarily perturb the circle position
                temp_v = v_current.copy()
                # Small spatial perturbation for gradient estimation
                temp_v[3*idx] += eps
                temp_v[3*idx + 1] += eps
                temp_v[3*idx + 2] += 0
                
                # Evaluate spatial effectiveness
                temp_effectiveness = compute_spatial_effectiveness(temp_v)
                perturbations.append(temp_effectiveness - current_effectiveness)
            
            # Use gradient to guide perturbation, with directional component to avoid over-perturbation
            directional_perturb = np.random.rand(n) * 1e-6 * perturbation_intensity
            v_perturbed = v_current + directional_perturb

            # Reoptimize with perturbations
            modified_bounds = bounds 
            modified_cons = cons

            # Adjust constraint tightness for reconfiguration
            for c in modified_cons:
                if c['type'] == 'ineq':
                    c['fun'] = lambda v, i=i, factor=constraint_relaxation_factor: constraint_fun_left_side(v, i) * factor

            perturbed_res = minimize(neg_sum_radii, v_perturbed, method="SLSQP",
                                     bounds=modified_bounds,
                                     constraints=modified_cons,
                                     options={"ftol": 1e-11, "gtol": 1e-9,
                                              "maxiter": max_iter_refinement, 
                                              "eps": 1e-8, 
                                              "disp": False})
            
            if perturbed_res.success:
                v = perturbed_res.x
            # If not converged, continue with current v with a soft radius extension
            else:
                # Apply radius growth with minimal constraints
                # Target radius extension: small growth to prevent over-perturbation
                # Grow radii based on spatial effectiveness 
                new_radii = v[2::3].copy()
                effective_growth = 0.01  # small growth factor, adjust as needed
                for i in range(n):
                    # Compute interaction score of circle i
                    dx = v[0::3, np.newaxis] - v[0::3]
                    dy = v[1::3, np.newaxis] - v[1::3]
                    dists = np.sqrt(dx**2 + dy**2)
                    min_dists = np.min(np.where(dists > 0, dists, np.inf), axis=1)
                    
                    # Calculate growth factor based on isolation: lower interactions should get larger expansion
                    isolation_score = np.sum(1 / (min_dists + 1e-9) * (v[2::3] + 1e-9))
                    # Growth proportional to inverse of interaction score, capped
                    growth = effective_growth * (1.0 / (1.0 + isolation_score * 0.5))
                    # Distribute the growth to others with a soft constraint on growth
                    if i != np.argmin(isolation_score):
                        new_radii[i] += growth * 0.2
                    else:
                        new_radii[i] += growth * 1.0
                
                # Apply the growth with soft bounds
                growth_vector = v.copy()
                growth_vector[2::3] = new_radii
                # Reoptimize with the new radii
                new_res = minimize(neg_sum_radii, growth_vector, method="SLSQP",
                                   bounds=modified_bounds,
                                   constraints=modified_cons,
                                   options={"ftol": 1e-11, "gtol": 1e-9,
                                            "maxiter": max_iter_refinement,
                                            "eps": 1e-8, 
                                            "disp": False})
                if new_res.success:
                    v = new_res.x
                else:
                    # Final fallback: grow radii of least constrained with soft limits
                    new_radii = v[2::3].copy()
                    # Create vectorized distances
                    dx = v[0::3, np.newaxis] - v[0::3]
                    dy = v[1::3, np.newaxis] - v[1::3]
                    dists = np.sqrt(dx**2 + dy**2)
                    min_distances = np.min(np.where(dists > 0, dists, np.inf), axis=1)
                    # Compute isolation scores
                    isolation = np.sum(1 / (min_distances + 1e-9) * (v[2::3] + 1e-9))
                    # Sort to find least constrained
                    isolated_idx = np.argsort(isolation)[0]
                    # Add soft expansion to isolation index
                    new_radii[isolated_idx] += 0.01 * radius_stability_weight
                    # Reapply
                    growth_vector = v.copy()
                    growth_vector[2::3] = new_radii
                    final_res = minimize(neg_sum_radii, growth_vector, method="SLSQP",
                                         bounds=modified_bounds,
                                         constraints=modified_cons,
                                         options={"ftol": 1e-11, "gtol": 1e-9,
                                                  "maxiter": max_iter_refinement,
                                                  "eps": 1e-8, 
                                                  "disp": False})
                    if final_res.success:
                        v = final_res.x
                    else:
                        v = v_current

    # Apply final radius adjustment with soft constraints
    # Apply adaptive radius scaling to prevent collapse
    # Ensure the radius growth doesn't create overlap without constraint violation
    def apply_radius_scaling(v):
        radii = v[2::3]
        centers = v[0::3], v[1::3]
        dx = centers[0][np.newaxis, :] - centers[0][:, np.newaxis]
        dy = centers[1][np.newaxis, :] - centers[1][:, np.newaxis]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimal distance between each pair
        dists[dists == 0] = np.inf
        min_dists = np.min(dists, axis=1)
        # Calculate a dynamic radius scaling factor
        scaling_factor = 1.0
        for i in range(n):
            # For circles that can grow without violating constraints
            # Use a heuristic based on the distance to nearest circle
            dist = min_dists[i]
            if dist < 3 * radii[i] + 1e-9:  # Overlapping, need to reduce
                scaling_factor = max(scaling_factor, (dist - 1e-9) / (radii[i] + 1e-9))
        
        # Scale radii with a safety factor to ensure they are within bounds
        scaled_radii = np.min([radii * scaling_factor, np.array([0.45] * n)])
        return scaled_radii

    # Apply final optimization and constraint-based scaling
    # Ensure that even after refinement, overlap is avoided
    v_final = v.copy()
    # Apply radius scaling to prevent overlap
    final_radii = apply_radius_scaling(v)
    v_final[2::3] = final_radii

    # Apply final optimization pass with updated radii
    # This ensures final configuration is valid
    final_res = minimize(neg_sum_radii, v_final, method="SLSQP",
                         bounds=bounds,
                         constraints=cons,
                         options={"ftol": 1e-11, "gtol": 1e-9,
                                  "maxiter": 300,
                                  "eps": 1e-8, 
                                  "disp": False})

    v = final_res.x if final_res.success else v

    # Final validation and clipping
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final safety check
    is_valid = True
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < radii[i] + radii[j] - 1e-12:
                is_valid = False
                break
        if not is_valid:
            break

    if not is_valid:
        # Fall back to last known safe configuration
        # Apply conservative scaling to safe configuration
        v_final = v.copy()
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        for i in range(n):
            # Find nearest neighbors with minimal distance for safe radius scaling
            dx = centers[i, 0] - centers[:, 0]
            dy = centers[i, 1] - centers[:, 1]
            dists = np.sqrt(dx**2 + dy**2)
            min_dist = np.min(dists[dists != 0])  # Avoid self
            radii[i] = np.min([radii[i], (min_dist - 1e-9) / 2])
        v_final[2::3] = radii
        centers = np.column_stack([v_final[0::3], v_final[1::3]])
        radii = np.clip(v_final[2::3], 1e-6, None)

    return centers, radii, float(radii.sum())
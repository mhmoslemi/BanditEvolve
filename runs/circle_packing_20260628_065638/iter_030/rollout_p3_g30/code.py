import numpy as np

def run_packing():
    n = 26
    col_count = 5
    row_count = 3  # Explicit grid with 5x3 layout for better control over spacing and constraints
    # Grid-based initial placement with optimized staggering and random jitter
    col_space = 1.0 / col_count
    row_space = 1.0 / row_count
    # Define initial grid points with jitter, alternating rows with offset
    # We'll compute positions in a more structured way to allow for precise control and reduce collisions from grid misalignment
    x_centers = np.zeros(n)
    y_centers = np.zeros(n)
    jitters_x = np.random.uniform(-0.03, 0.03, size=n)
    jitters_y = np.random.uniform(-0.03, 0.03, size=n)

    for i in range(n):
        col = i % col_count
        row = i // col_count
        
        # Column-based x-position: symmetric around center of column space
        x_center = col * col_space + col_space * 0.5
        # Row-based y-position: symmetric around center of row space 
        y_center = row * row_space + row_space * 0.5
        
        # Add jitter to break symmetry and reduce initial clustering
        x_centers[i] = x_center + jitters_x[i]
        y_centers[i] = y_center + jitters_y[i]
        
        # Alternate row offset to achieve staggered layout
        if row % 2 == 1:
            x_centers[i] += col_space * 0.3  # Smaller stagger to avoid overlapping with row boundaries
        
        # Also ensure not to go out of bounds due to jitter
        max_offset = np.sqrt(0.5 * (col_space + row_space)**2)
        # Adjust if initial jitter causes crossing of the boundaries
        x_centers[i] = np.clip(x_centers[i], 0.0 + 1e-6, 1.0 - 1e-6)
        y_centers[i] = np.clip(y_centers[i], 0.0 + 1e-6, 1.0 - 1e-6)
    
    # Initial radius estimation is based on grid spacing and edge space
    # Optimize the initial radius based on grid spacing - 0.5*grid spacing as safe buffer
    # Using a slightly more aggressive initial value than before, since we use controlled grid alignment
    initial_radius = (1.0 / (col_count + row_count)) * 0.85 - 1e-3  # More aggressive than parent approach
    r0 = np.full(n, initial_radius)

    # Decision vector is of length 3*n
    v0 = np.zeros(3*n)
    v0[0::3] = x_centers
    v0[1::3] = y_centers
    v0[2::3] = r0

    # Bounds must be exactly 3n elements: x, y, r for each circle
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective to maximize sum of radii (convert to minimization by negation)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Build constraints using vectorised lambda functions with capture that are safe for scipy.optimize
    # Constraints: 
    # - x >= r (left margin)
    # - x + r <= 1 (right margin)
    # - y >= r (bottom margin)
    # - y + r <= 1 (top margin)
    # - distance between centers >= r1 + r2 (overlap constraints)

    # Precompute all constraints
    cons = []

    for i in range(n):
        # Left margin constraint (x >= r)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right margin constraint (x + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom margin constraint (y >= r)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top margin constraint (y + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        
        # Also, we'll add "soft" constraints by adding distance constraints at reduced weighting
        # This allows for more gradual convergence and avoids sharp gradients causing oscillation
        for j in range(i+1, n):
            # Use anonymous function with captured i and j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return (dx**2 + dy**2) - (v[3*i+2] + v[3*j+2])**2
            # Add the constraint with weight to allow for softer enforcement
            cons.append({
                "type": "ineq",
                "fun": constraint_func,
                "jac": lambda v, i=i, j=j: (
                    # Jacobian for dx^2 + dy^2 - (r_i + r_j)^2
                    2*(v[3*i] - v[3*j]), 
                    2*(v[3*i+1] - v[3*j+1]), 
                    -2*(v[3*i+2] + v[3*j+2]),
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    # for the rest of the parameters where i and j don't affect the constraint
                ) + tuple([0]*(3*(n) - 3*2))  # Pad with 0s for non-related parameters
            })
    
    # Initial optimization with tighter tolerances and more iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP",
                   bounds=bounds,
                   constraints=cons,
                   options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-12})

    # If initial optimization succeeds, perform asymmetric reconfiguration using a 
    # combination of spatial hashing and adaptive reconfiguration with multi-stage refinement
    
    if res.success:
        best_v = res.x
        # We will perform multiple refinement stages: 
        # 1. Spatial hashing with directional perturbation (based on current configuration) to explore new areas
        # 2. Targeted expansion with spatial analysis using matrix inversion
        # 3. Hybrid optimization using multiple phases to ensure stability

        # --- Phase 1: Spatial Hashing with Directional Perturbation ---
        # Perturb by a small amount (directionally based on position, radius, and spatial relationships)
        # Use directional vectors based on grid coordinates and distance magnitudes
        # This avoids random noise but preserves some randomness for escaping local minima
        spatial_hash_coeff = 0.03  # Scaling factor for spatial perturbation
        
        # Generate perturbation vectors based on gradient direction (approximate)
        # Here's a heuristic: if in high density areas, perturb more; if isolated, perturb less
        # For simplicity, use relative radii and grid coordinates for spatial direction
        perturb = np.zeros_like(best_v)
        for i in range(n):
            x, y, r = best_v[3*i], best_v[3*i+1], best_v[3*i+2]
            # Create a directional vector based on proximity to grid edges
            dx_dir = 1.0 - x - r  # Proximity to right edge
            dy_dir = 1.0 - y - r  # Proximity to top edge
            perturb[3*i] += dx_dir * spatial_hash_coeff * r  # Scale by radius
            perturb[3*i+1] += dy_dir * spatial_hash_coeff * r
            
            # Add random jitter with direction sensitivity
            perturb[3*i] += (np.random.rand() - 0.5) * spatial_hash_coeff * np.sqrt(1.0 - (x - 0.5)**2)
            perturb[3*i+1] += (np.random.rand() - 0.5) * spatial_hash_coeff * np.sqrt(1.0 - (y - 0.5)**2)

        perturbed_v = best_v + perturb
        # Clip to bounds to prevent escaping
        for i in range(n):
            perturbed_v[3*i] = np.clip(perturbed_v[3*i], 0.0, 1.0)
            perturbed_v[3*i+1] = np.clip(perturbed_v[3*i+1], 0.0, 1.0)
            perturbed_v[3*i+2] = np.clip(perturbed_v[3*i+2], 1e-4, 0.5)
        
        # Reoptimize with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-12})

        # If res is still successful, we perform multi-stage optimization

        if res.success:
            current_v = res.x
            # --- Phase 2: Targeted expansion with spatial analysis and gradient-based reconfiguration ---
            # Re-evaluate distances and find the most constrained (or least constrained) circle
            # Calculate distance matrix using vectorized operations
            centers = current_v[0::3], current_v[1::3]
            centers = np.column_stack((centers[0], centers[1]))
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    distances[i, j] = np.sqrt(dx**2 + dy**2)
            
            # Find the most isolated circle (minimum distance to others)
            min_dist = np.min(distances, axis=1)
            least_isolated_idx = np.argmin(min_dist)

            # Alternatively, find the most connected (most constrained)
            max_dist = np.max(distances, axis=1)
            most_connected_idx = np.argmax(max_dist)

            # Create a mask for expansion to target the least isolated circle
            # We'll use a multi-phase approach to increase its radius while maintaining total sum
            # The expansion will be gradual and adaptive
            # Use a hybrid approach: expand the isolated circle + redistribute extra to others
            
            new_v = current_v.copy()
            radii = current_v[2::3]
            new_radii = radii.copy()

            # We will perform a targeted expansion that is 0.3% more than initial
            expansion_target_percent = 0.003  # Expand by 0.3%
            expansion_value = radii[least_isolated_idx] * expansion_target_percent
            new_radii[least_isolated_idx] += expansion_value
            
            # Create an expanded version that also increases some nearby circles slightly
            # This allows for more stable expansion and avoids over-concentration of radius in one spot
            # Distribute the expansion to others with a priority towards nearby circles
            # To ensure the constraint remains satisfied, we do this incrementally via reoptimization
            # Instead of direct assignment, we will generate the new_radii vector and perform local optimization
            # This is safer as direct radius assignment leads to gradient discontinuities

            # Create a new parameter vector: keep centers, expand radii
            new_v = current_v.copy()
            new_v[2::3] = new_radii

            # Reoptimize with the new parameter vector
            res = minimize(neg_sum_radii, new_v, method="SLSQP",
                           bounds=bounds,
                           constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-12})

            # If optimization succeeds, perform additional expansion via gradient refinement
            # This phase focuses on maximizing potential, while ensuring stability
            if res.success:
                final_v = res.x
                final_radii = final_v[2::3]
                final_centers = np.column_stack((final_v[0::3], final_v[1::3]))

                # Check if we can perform a final reconfiguration
                # This phase focuses on spatial reconfiguration using direction vector gradients
                # We will attempt to slightly reorient the least constrained circle towards more open areas
                # This is done with a directional perturbation based on current configuration

                # Compute direction vector based on distance to edge and proximity to other circles
                x, y, r = final_centers[least_isolated_idx]
                dx_direction = 1.0 - x - r
                dy_direction = 1.0 - y - r
                # Add a small directional perturbation
                dx_perturb = dx_direction * 0.005 * r
                dy_perturb = dy_direction * 0.005 * r
                final_centers[least_isolated_idx, 0] += dx_perturb
                final_centers[least_isolated_idx, 1] += dy_perturb

                # Clip again and reconstruct the full vector
                final_v = np.zeros_like(final_v)
                final_v[0::3] = final_centers[:, 0]
                final_v[1::3] = final_centers[:, 1]
                final_v[2::3] = final_radii.copy()
                # Clip final parameters
                for i in range(n):
                    final_v[3*i] = np.clip(final_v[3*i], 0.0, 1.0)
                    final_v[3*i+1] = np.clip(final_v[3*i+1], 0.0, 1.0)
                    final_v[3*i+2] = np.clip(final_v[3*i+2], 1e-4, 0.5)

                # Reoptimize with fine-tuned directional perturbation
                res = minimize(neg_sum_radii, final_v, method="SLSQP",
                               bounds=bounds,
                               constraints=cons,
                               options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-12})

        # Final fallback: if any stage fails, we use the best_v from previous successful stages
        if not res.success:
            final_v = best_v
        else:
            final_v = res.x
    else:
        final_v = v0

    # Final post-optimization clipping and validation
    centers = np.column_stack((final_v[0::3], final_v[1::3]))
    radii = np.clip(final_v[2::3], 1e-6, 0.5)
    # Enforce non-negative and within bounds
    radii[radii < 1e-6] = 1e-6
    centers[centers < 0.0] = 0.0
    centers[centers > 1.0] = 1.0

    # Final optimization step in case earlier steps failed due to constraints or gradients
    res = minimize(neg_sum_radii, final_v, method="SLSQP",
                   bounds=bounds,
                   constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-12})

    # Final output
    if res.success:
        v = res.x
    else:
        v = final_v
    
    centers = np.column_stack((v[0::3], v[1::3]))
    radii = np.clip(v[2::3], 1e-6, 0.5)
    return centers, radii, float(radii.sum())
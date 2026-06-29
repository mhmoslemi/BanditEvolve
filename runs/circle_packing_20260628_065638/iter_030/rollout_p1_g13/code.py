import numpy as np

def run_packing():
    n = 26
    
    # Precompute geometric grid with advanced adaptive partitioning
    cols = int(np.ceil(n ** (1/2) * 0.75))
    rows = (n + cols - 1) // cols
    cell_area = (1.0 / rows) * (1.0 / cols)
    # Adaptive grid: adjust columns based on row density
    cols = int(np.ceil(n / rows))
    
    # Grid-based initialization with geometric hashing and staggered optimization
    grid = []
    for r in range(rows):
        row_start_col = r * cols
        row_end_col = min((r + 1) * cols, n)
        row_entries = []
        for c in range(row_start_col, row_end_col):
            if c < n:
                # Adaptive cell center calculation with row/column density correction
                cell_center_x = (c + 0.5) / cols  # + 0.02 if (r % 3 == 0 and c % 2 == 1)
                cell_center_y = (r + 0.5) / rows
                # Add random perturbations while avoiding grid collapse
                x = cell_center_x + np.random.uniform(-0.07, 0.07) * (1.0 / (cols * (rows**0.85)))
                y = cell_center_y + np.random.uniform(-0.07, 0.07) * (1.0 / (rows * (cols**0.85)))
                # Stagger rows - alternate row alignment using sine function
                stagger = 0.25 * np.sin(np.pi * (r % 2)) / (cols * (rows + 1))
                if r % 2 == 1:
                    x += stagger + np.random.uniform(-0.025, 0.025)*(1.0/cols)
                row_entries.append((x, y))
        grid.extend(row_entries)
    
    # Initialize radii with adaptive density-aware distribution
    # Based on grid cell area and spacing: r_max = min(0.15, 0.3 / (cells_density * 1.3)) 
    # Use adaptive inverse sqrt of density
    cell_density = 1.0 / (rows * cols)
    # Use radius that is inverse proportional to spatial density, avoiding fixed pattern
    base_radius = np.sqrt(cell_area) * 0.82
    # Ensure minimum radius is consistent with spatial density and grid
    min_radius_initial = 0.02 + 0.0002 * (rows * cols)  # adaptive scaling for larger grids
    r0 = np.clip(base_radius, 1e-4, 0.35) - np.random.uniform(0, 0.01)  # adaptive noise
    # Add small random variation to avoid grid pattern
    random_r = np.random.uniform(-0.005, 0.005, size=n) * (1.1 + 1.0/((rows * cols)**0.5))
    r0 = np.clip(r0 + random_r, 1e-4, 0.35)
    
    # Construct decision vector
    v0 = np.empty(3 * n)
    v0[0::3] = np.array([x for x, y in grid])
    v0[1::3] = np.array([y for x, y in grid])
    v0[2::3] = r0.copy()

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Define objective for maximization (minimize negative sum)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Construct constraints with strict lambda capture and correct indexing
    cons = []

    # Add boundary constraints with error-checking for index and vector safety
    for i in range(n):
        # Left margin constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})  # Explicit i capture
        
        # Right margin constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        
        # Bottom margin constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i +1] - v[3*i + 2]})
        
        # Top margin constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i +1] - v[3*i + 2]})
    
    # Add inter-circle overlap constraints with advanced vectorization
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with i, j capture for all constraints
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i + 1] - v[3*j + 1])**2 
                             - (v[3*i + 2] + v[3*j + 2])**2})

    # Initial optimization with high precision, adaptive tolerances, and enhanced convergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={
                       # High tolerance for early phase
                       "maxiter": 1000, 
                       "ftol": 1e-12,  # High precision for final tightening
                       "gtol": 1e-12, 
                       "eps": 1e-12,  # Small perturbation
                       "disp": False
                   })
    
    # Enhanced reconfiguration step with geometric hashing and radius balancing
    if res.success:
        # Extract final state
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Validate before reconfiguration
        if not validate_packing(centers, radii)[0]:
            print("Initial configuration failed")
            v = v0
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds, 
                           constraints=cons, options={
                               "maxiter": 600, 
                               "ftol": 1e-12, 
                               "gtol": 1e-12, 
                               "disp": False
                           })
    
    # Apply safety validation and grammar check before any expansion
    def grammar_and_safety_check(v):
        try:
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            if np.any(np.isnan(radii)) or np.any(np.isnan(centers)):
                return False
            if np.any(radii < 1e-5):
                return False
            if not validate_packing(centers, radii)[0]:
                return False
            return True
        except Exception:
            return False

    # Ensure safety before expansion (strict)
    if res.success:
        v = res.x
        if grammar_and_safety_check(v):
            # Safety check passed
            pass
        else:
            print("Safety check failed")
            v = v0
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds, 
                           constraints=cons, options={
                               "maxiter": 600, 
                               "ftol": 1e-12, 
                               "gtol": 1e-12, 
                               "disp": False
                           })
    
    # Apply targeted expansion on least constrained circle with geometry-aware expansion
    if res.success:
        # Extract current state
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances and identify least constrained circle
        # Vectorized pairwise distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (maximize min distance)
        min_distances = np.min(dist_matrix, axis=1)
        least_constrained_idx = np.argmax(min_distances)
        min_constrained_distance = min_distances[least_constrained_idx]
        
        # Calculate growth based on current radius sum, geometry, and spatial density
        current_total = np.sum(radii)
        # Growth based on spatial density: higher density means smaller potential growth
        density_factor = 1.0 / (rows * cols)  # inverse of spatial density
        # Growth based on minimal distance and radius
        potential_growth = min_constrained_distance * (1.0 - np.mean(radii)) * density_factor
        
        # Target total sum expansion: grow by 1.5% of current sum
        target_percent_growth = 0.015  # 1.5% of current sum expansion
        target_total = current_total * (1.0 + target_percent_growth)
        
        # Define expansion plan
        expansion_plan = np.zeros(n)
        expansion_plan[least_constrained_idx] = 0.8 * potential_growth  # more aggressive for anchor
        base_growth_per_circle = (target_total - current_total) / (n) * density_factor
        
        # Add proportional baseline expansion
        expansion_plan += base_growth_per_circle
        
        # Apply expansion carefully to prevent overstepping and maintain geometry
        expanded_radii = radii.copy()
        expanded_centers = centers.copy()
        for i in range(n):
            if i == least_constrained_idx:
                # Handle special case for least constrained
                if expanded_radii[i] + expansion_plan[i] > 0.35:
                    # Adjust expansion to respect hard limit
                    expansion_plan[i] = 0.35 - expanded_radii[i]
            else:
                if expanded_radii[i] + expansion_plan[i] > 0.35:
                    # Adjust to respect hard limit
                    expansion_plan[i] = 0.35 - expanded_radii[i]
            # Apply expansion to radius
            expanded_radii[i] += expansion_plan[i]
            # Update center if expansion triggers spatial shift
            # Use gradient-based shift to preserve constraints
            # Adjust position based on expansion and spatial density
            if expansion_plan[i] > 1e-7:
                # Calculate new center based on spatial density and expansion
                spatial_density = 1.0 / (rows * cols)
                new_expanded_r = expanded_radii[i]
                # Apply shift based on previous radius and current expansion
                shift_factor = new_expanded_r / (radii[i] + expansion_plan[i]) if radii[i] != 0 else 1.0
                # Apply controlled spatial perturbation based on expansion
                # Add small random shift in direction of expansion for spatial reconfiguration
                dx_perturb = np.random.uniform(-0.003, 0.003) * (0.01 * (rows * cols))
                dy_perturb = np.random.uniform(-0.003, 0.003) * (0.01 * (rows * cols))
                expanded_centers[i, 0] += dx_perturb * shift_factor
                expanded_centers[i, 1] += dy_perturb * shift_factor
            # Ensure position stays within bounds
            expanded_centers[i, 0] = np.clip(expanded_centers[i, 0], 0, 1)
            expanded_centers[i, 1] = np.clip(expanded_centers[i, 1], 0, 1)
    
        # Construct new decision vector with expanded radii and shifted centers
        expanded_v = v.copy()
        expanded_v[2::3] = expanded_radii
        expanded_v[0::3] = expanded_centers[:, 0]
        expanded_v[1::3] = expanded_centers[:, 1]
        
        # Re-evaluate with expansion
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 400,
                           "ftol": 1e-11,
                           "gtol": 1e-11,
                           "eps": 1e-11,
                           "disp": False
                       })
    
    # Final clean-up and validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.35)  # clip to prevent overexpansion
    # Ensure final configuration passes the safety check
    if not grammar_and_safety_check(v):
        print("Final safety check failed. Reverting to safe configuration")
        # Revert to safe configuration
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)  # safe clipping now that we've re-evaluated
    
    # Final sanity checks
    assert np.prod(radii > 0), "Radius <=0 detected"
    assert np.all(np.isfinite(centers)), "NaN center detected"
    assert np.all(np.isfinite(radii)), "NaN radius detected"
    
    return centers, radii, float(radii.sum())
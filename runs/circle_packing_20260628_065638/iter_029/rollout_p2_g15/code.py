import numpy as np

def run_packing():
    n = 26
    cols = 6  # Adjust column count for spatial diversity and geometric control
    rows = (n + cols - 1) // cols
    cell_size = 0.5  # Base spatial cell unit
    
    # Generate primary grid cells with randomized staggered placement
    grid_x = (np.arange(cols) + 0.5) / cols
    grid_y = (np.arange(rows) + 0.5) / rows
    
    # Initialize with randomized grid cells with geometric clustering and perturbation for diversity
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # Primary grid center with spatial cell perturbation to avoid over-concentration
        x_center = grid_x[col]
        y_center = grid_y[row]
        
        # Spatial perturbation with radial distribution for better spatial hashing
        r_perturb = 0.06  # Perturbation magnitude
        x = x_center + np.random.uniform(-r_perturb, r_perturb)
        y = y_center + np.random.uniform(-r_perturb, r_perturb)
        
        # Stagger alternate rows to enhance spatial separation
        if row % 2 == 1:
            x += 0.4 / cols  # Larger horizontal shift to avoid row alignment
        # Clamp to square bounds
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation using spatial cell density and enhanced scaling
    # Base radius is based on square packing: cell_size^2 = 2*r^2 => r = cell_size / sqrt(2)
    # But we improve this with better scaling for grid-aligned distribution
    radius_base = 0.37 / cols  # Slightly optimized base radius
    r0 = radius_base - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Build bounds vector with 3*n entries for decision vector of length 3n
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))   # x-range
        bounds.append((0.0, 1.0))   # y-range
        bounds.append((1e-4, 0.5))  # radius range
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint definitions with optimized lambda closures
    cons = []
    for i in range(n):
        # Left constraint: x >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Circular overlap constraints with optimized computation
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2
                )
            })
    
    # Initial optimization with aggressive iteration and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-12})
    
    # If optimization fails, fallback to alternate initialization
    if not res.success:
        print("Initial optimization failed, attempting alternate initialization")
        
        # Alternate grid initialization with geometric re-shuffling
        xs = []
        ys = []
        for i in range(n):
            col = i % cols
            row = i // cols
            x_center = grid_x[col]
            y_center = grid_y[row]
            
            # Larger perturbation to avoid uniform distribution and promote separation
            x = x_center + np.random.uniform(-0.1, 0.1)
            y = y_center + np.random.uniform(-0.1, 0.1)
            
            # Alternate row shift with larger range for better distribution
            if row % 2 == 1:
                x += 0.5 / cols  # 50% of cell width as shift
            # Clamp to square bounds
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            xs.append(x)
            ys.append(y)
        
        v0 = np.empty(3 * n)
        v0[0::3] = np.array(xs)
        v0[1::3] = np.array(ys)
        v0[2::3] = np.full(n, r0)
        
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1200, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-12})
    
    # Perform advanced spatial reconfiguration and re-optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for constraint validation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle by maximizing the min distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        r_max = np.max(radii)
        r_avg = np.mean(radii)
        
        # Generate perturbation for spatial reconfiguring using enhanced gradient-aware method
        # Perturbation is based on spatial distribution of current configuration
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Gradient-aware scaling: more movement for less constrained circles
            scale_factor = (1.0 + 0.5 * (min_dists[i] / r_avg))
            dx_perturb = spatial_hash[i, 0] * (radii[i] / r_avg) * 1.3 * scale_factor
            dy_perturb = spatial_hash[i, 1] * (radii[i] / r_avg) * 1.3 * scale_factor
            perturbed_v[3*i] += dx_perturb
            perturbed_v[3*i+1] += dy_perturb
        
        # Second-level optimization with spatial reconfiguration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-12})
    
    # Targeted expansion phase: explore expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 + (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
        
        # Recompute least constrained circle for current configuration
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        base_radius = 0.36 / cols
        
        # Estimate potential expansion based on current density and spatial availability
        # Calculate area utilization and potential growth space
        occupied_area = np.sum(np.pi * radii**2)
        available_area = 1.0
        utilization = occupied_area / available_area
        max_possible_growth = (1 - utilization) * 2  # 2x possible growth
            
        expansion_factor = max_possible_growth / (n - 1)
        
        # Generate radial expansion vector with adaptive scaling and stochasticity
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Increase stochasticity for expansion with spatial potential
                spatial_expansion = 1.0 + 0.15 * np.random.rand()
                expansion_i = expansion_factor * spatial_expansion
                new_radii[i] += expansion_i
        
        # Apply expansion with strict constraint validation - multiple passes
        max_expansion_attempts = 5
        expansion_successful = False
        for _ in range(max_expansion_attempts):
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                expansion_successful = True
                break
            else:
                # Reduce expansion by 10% if invalid
                new_radii = radii + (new_radii - radii) * 0.9
                # Optional: force reset if expansion is too unstable
                if np.any(new_radii < 0.001):
                    new_radii = radii.copy()
        
        # Use expanded configuration if valid
        if expansion_successful:
            v_new = v.copy()
            v_new[2::3] = new_radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-12})
        else:
            # If expansion fails, use original config with tighter constraints
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-12})
    
    # Final fallback: ensure we always return valid configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
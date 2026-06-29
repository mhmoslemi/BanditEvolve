import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Base grid parameters with spatial hashing and adaptive initialization
    grid_x = (np.arange(cols) + 0.5) / cols
    grid_y = (np.arange(rows) + 0.5) / rows
    
    # Randomized and adaptive initialization with enhanced spatial hashing and cluster separation
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Base position: grid-based
        x_base = grid_x[col]
        y_base = grid_y[row]
        
        # Random perturbation with adaptive amplitude based on grid density and proximity
        max_offset = 0.05 + 0.003 * (cols / rows)  # Larger spacing for tighter grids
        dx = np.random.uniform(-max_offset, max_offset) * (cols / n)
        dy = np.random.uniform(-max_offset, max_offset) * (rows / n)
        
        # Stagger rows to avoid alignment
        if row % 2 == 1:
            x_base += 0.5 / cols
        # Apply perturbation
        x = x_base + dx
        y = y_base + dy
        
        # Spatial hashing to ensure spatial diversity (prevent grid bias)
        spatial_hash = np.array([col, row]) / np.array([cols, rows]).astype(np.float64)
        spatial_hash *= 0.003 * cols  # Small-scale perturbation to avoid alignment
        
        # Final position with hashing perturbation
        x += spatial_hash[0]
        y += spatial_hash[1]
        
        # Clamp within the unit square
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation with adaptive scaling
    # Base radius is adjusted based on grid density, avoiding uniform grid spacing bias
    radius_base = 0.34 / cols + (1.0 / (n + 1)) * (0.25 / cols)
    r0 = np.full(n, radius_base - 1e-3)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Ensure bounds are exactly 3n entries, matching the length of the vector
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))    # x
        bounds.append((0.0, 1.0))    # y
        bounds.append((1e-4, 0.5))   # radius
    
    # Objective: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints: 4 per circle (left/right, bottom/top), and all-pairs distances
    cons = []
    
    # Vectorized boundary constraint functions: 
    # Left (x - r >= 0), Right (x + r <= 1), Bottom (y - r >= 0), Top (y + r <= 1)
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized circle-circle overlap constraints: distance^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i+1, n):
            # Ensure capture of i and j in closure
            # Use higher order lambda to bind parameters
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + 
                    (v[3*i+1] - v[3*j+1])**2 - 
                    (v[3*i+2] + v[3*j+2])**2
                )
            })
    
    # First pass optimization with tighter settings
    res = minimize(neg_sum_radii, v0, method='SLSQP', 
                   bounds=bounds, constraints=cons, 
                   options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    # Optimization reconfiguration: spatial grid tiling with explicit cluster separation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Cluster-aware grid reconfiguration with explicit spatial hashing
        # Use a tiling approach on the unit square with multiple grids
        grid_params = [0.4, 0.55, 0.7]  # different spacings to capture various scale patterns
        new_centers = []
        new_radii = []
        
        # For each grid spacing, generate candidate cluster positions
        for spacing in grid_params:
            cols_curr = int(np.ceil(n ** 0.5 * spacing))
            rows_curr = (n + cols_curr - 1) // cols_curr
            
            # Generate a grid with staggered rows and spatial hashing
            grid_x_curr = np.linspace(0.0, 1.0, cols_curr + 1)[1:-1]  # avoid edges
            grid_y_curr = np.linspace(0.0, 1.0, rows_curr + 1)[1:-1]
            grid_x_curr = (grid_x_curr + 0.5) / cols_curr
            grid_y_curr = (grid_y_curr + 0.5) / rows_curr
            
            # Generate initial positions for this grid
            temp_x = []
            temp_y = []
            for idx in range(n):
                col = idx % cols_curr
                row = idx // cols_curr
                x = grid_x_curr[col] + np.random.uniform(-0.03, 0.03)
                y = grid_y_curr[row] + np.random.uniform(-0.03, 0.03)
                
                # Staggered rows
                if row % 2 == 1:
                    x += spacing / (2 * cols_curr)
                
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                temp_x.append(x)
                temp_y.append(y)
            
            # Add to cluster list
            new_centers.append(np.column_stack([temp_x, temp_y]))
            new_radii.append(np.full(n, spacing / cols_curr - 1e-3))  # radius based on spacing
        
        # Now, use a combination of grid-based cluster options to form an enhanced configuration
        # Select from all grid configurations to form a hybrid grid
        # Pick the one with maximal radius sum for initial reconfiguration
        best_config = None
        best_sum = -np.inf
        
        for config_idx in range(len(new_centers)):
            cx = new_centers[config_idx]
            cr = new_radii[config_idx]
            # Ensure validity of this cluster configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = cx[i, 0] - cx[j, 0]
                    dy = cx[i, 1] - cx[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < cr[i] + cr[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid and np.sum(cr) > best_sum:
                best_config = config_idx
                best_sum = np.sum(cr)
        
        if best_config is not None:
            # Re-configure to this grid-based cluster
            cx = new_centers[best_config]
            cr = new_radii[best_config]
            
            # Create a perturbed vector from this configuration
            v_new = np.empty(3 * n)
            v_new[0::3] = cx[:, 0]
            v_new[1::3] = cx[:, 1]
            v_new[2::3] = cr
        
        # Re-evaluate with this configuration
        res = minimize(neg_sum_radii, v_new if best_config is not None else v, 
                       method='SLSQP', bounds=bounds, constraints=cons, 
                       options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})
    
    # Spatial expansion targeting: 
    # Find least constrained (i.e., farthest from other circles) and boost radius
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix efficiently
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute min distance per circle
        min_dists = np.min(dists, axis=1)
        
        # Find the least constrained circle: one with the largest minimal distance
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate radius expansion strategy based on minimal distance to others
        # Use gradient-adjusted expansion: grow radius in proportion to min distance
        current_total = np.sum(radii)
        # We aim to increase the total sum by at least 7% or find a better balance
        expansion_target = current_total * 1.02  # Slightly better than 2.634292
        expansion_per_circle = (expansion_target - current_total) / n
        
        # Create an expansion vector that targets the least constrained circle
        new_radii = radii.copy()
        # First, boost the least constrained circle with a multiplier based on its minimum distance
        # Scale by a soft factor based on the relative min distance
        max_min_dist = np.max(min_dists)
        min_min_dist = np.min(min_dists)
        dist_factor = (min_dists[least_constrained_idx] - min_min_dist) / (max_min_dist - min_min_dist)
        radius_increase = dist_factor * (expansion_per_circle * 1.2)  # Over-expanding slightly
        new_radii[least_constrained_idx] += radius_increase
        
        # Stochastically expand other circles with some randomness for diversity
        for i in range(n):
            if i != least_constrained_idx:
                # Use some local spatial influence (e.g., neighbors' expansion)
                neighbor_distances = dists[i, :]
                neighbor_radii = radii[neighbor_distances < 2*radii[i]]
                if len(neighbor_radii) > 0:
                    local_growth = expansion_per_circle * (1.0 + 0.05 * np.random.rand())
                    new_radii[i] += local_growth
                else:
                    # If isolated, expand equally
                    new_radii[i] += expansion_per_circle
        
        # Apply expansion with local constraint validation
        # Iterate until the configuration is valid
        iterations = 0
        while iterations < 5:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            
            # Validate expanded configuration
            valid = True
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
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
                break
            else:
                # Reduce expansion by 4% to ensure we do not overshoot
                new_radii = radii + (new_radii - radii) * 0.96
                iterations += 1
        
        # Apply expansion and optimize with the refined configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})
    
    # Final result with fallback and spatial refinement
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    
    # Final validation step to ensure we have a valid packed configuration
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < radii[i] + radii[j] - 1e-12:
                raise RuntimeError("Final validation failed with packing constraint")
    
    return centers, radii, float(radii.sum())
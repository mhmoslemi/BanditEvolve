import numpy as np

def run_packing():
    # --- CONFIGURATIVE CONSTANTS ---
    n = 26
    grid_cols = 5
    grid_rows = (n + grid_cols - 1) // grid_cols
    seed_offset = 0.055  # Increased from 0.05 to 0.055 for more diverse cluster breakage
    expansion_factor = 0.0042  # Slightly below the parent's 0.006 to avoid overexpanding in the initial phase
    
    # Initialize geometric clustering with improved stochasticity and spatial balance
    xs = []
    ys = []
    base_centers = np.zeros((n, 2))
    
    for i in range(n):
        row = i // grid_cols
        col = i % grid_cols
        # Base grid layout with offset for spatial balance
        x_center = (col + 0.5) / grid_cols
        y_center = (row + 0.5) / grid_rows
        
        # Add a small, randomized shift to avoid uniform grid clusters
        # Introduce a dynamic offset that depends on spatial location to create more organic spacing
        x_offset = np.random.uniform(-seed_offset, seed_offset) * (1 + 0.1 * row)
        y_offset = np.random.uniform(-seed_offset, seed_offset) * (1 + 0.1 * col)
        # Alternate row shuffling to simulate natural packing
        if row % 2 == 1 and col < 3:  # Add controlled staggering in left part of grid
            x_center += 0.1 / grid_cols + np.random.uniform(-0.025, 0.025)
        # Add a small spatial correlation to prevent extreme clustering
        x_center += (np.random.rand() - 0.5) * 0.03 * (1 / (0.6 + row * 0.2))
        y_center += (np.random.rand() - 0.5) * 0.03 * (1 / (0.6 + col * 0.2))
        
        base_centers[i, 0] = x_center + x_offset
        base_centers[i, 1] = y_center + y_offset
    
    # Initial radius calculation with improved distribution and adaptive scaling
    avg_cell_size = 1.0 / grid_cols  # Approximate base cell size in the grid
    # Radians of initial radius: we use 0.6 * avg_cell_size - 1e-3 to enable better growth
    base_radius = 0.6 * avg_cell_size - 1e-3
    # Add spatial variation in base radii: larger circles in high-staggered positions to allow better distribution
    base_radius += np.random.rand(n) * 0.08 - 0.04  # Small variation to allow growth space
    base_radius = np.clip(base_radius, 1e-3, 0.4)  # Preventing extremely small or large initial values
    
    # Form initial vector
    v0 = np.empty(3 * n, dtype=np.float64)
    v0[0::3] = base_centers[:, 0]
    v0[1::3] = base_centers[:, 1]
    v0[2::3] = base_radius.copy()
    
    # --- BOUNDS CONFIGURATION ---
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-3, 0.4)]  # Reduced max radius to avoid overgrowth in first optimization
    
    # --- CONSTRAINTS DEFINITION (Vectorized for performance) ---
    cons = []
    
    for i in range(n):
        # x >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # x + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # y + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using lambda with captured indices to optimize
    # Note: Use a separate loop to avoid closure issues during constraint creation
    for i in range(n):
        for j in range(i + 1, n):
            # Avoid creating lambda closures that refer to variables that change during loop
            # So, we use an intermediate function to wrap it for safe constraint creation
            # Using a helper to generate the constraint functions safely
            
            def create_overlap_constraint(i, j):
                def constraint(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                return constraint
            
            cons.append({"type": "ineq", "fun": create_overlap_constraint(i, j)})
    
    # --- OPTIMIZATION TACTICAL PHASES ---
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-10})
    
    # --- PHASE 1: Asymmetric Perturbation with Spatial-aware Optimization ---
    # If success, perform spatial-aware perturbation with radial dependency
    # Perturbation strategy: use spatial gradient of the configuration to guide perturbation
    # This reduces the likelihood of perturbing into invalid spaces
    if res.success:
        v = res.x
        radii = v[2::3]
        # Use gradient from the last successful optimization to guide the next iterations
        # Compute spatial gradients for the last iteration
        centers = np.column_stack([v[0::3], v[1::3]])
        dx_grid = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
        dy_grid = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
        dist_grid = np.sqrt(dx_grid**2 + dy_grid**2)
        
        # Compute spatial sensitivity for each circle
        # Sensitivity = (sum of inverse distances) to others, weighted by radii
        spatial_sensitivity = np.zeros(n)
        for i in range(n):
            # Calculate local influence of this circle
            influence = 0.0
            for j in range(n):
                if i != j:
                    dist = dist_grid[i, j]
                    if dist > 1e-6:
                        influence += 1 / (dist * (radii[i] + radii[j]))
            spatial_sensitivity[i] = influence
        
        # Use spatial sensitivity to shape the next perturbation
        seed_map = np.random.rand(n, 2)
        perturbation_factor = 0.035 * (radii / np.mean(radii)) * spatial_sensitivity / np.max(spatial_sensitivity)
        # Perturb centers with a direction proportional to spatial sensitivity
        # This encourages expansion on less densely packed areas
        perturbed_v = v.copy()
        for i in range(n):
            dir_x = seed_map[i, 0] * 0.8 - 0.4  # Randomized direction
            dir_y = seed_map[i, 1] * 0.8 - 0.4
            perturbed_v[3*i] += dir_x * perturbation_factor[i]
            perturbed_v[3*i+1] += dir_y * perturbation_factor[i]
        
        # Optimized step with tighter controls
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})
    
    # --- PHASE 2: Iterative Expansion via Constraint-Driven Growth (Surgical Expansion) ---
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2) + 1e-12 # Small epsilon to avoid nan in 0 distance
        
        # Calculate minimum distance to others for each circle
        min_distances = np.min(dists, axis=1)
        growth_potentials = np.zeros(n)
        for i in range(n):
            # Growth potential is inverse of minimum distance, scaled by current radius
            # We penalize circles that are too close to others
            if min_distances[i] > 1e-6:
                growth_potentials[i] = 1 / min_distances[i] * (radii[i]) * (1 + 0.1 * np.random.rand())
            else:
                growth_potentials[i] = 0.0
        
        # Select most constrained circles (least growing potential)
        # Sort indices by growth potential in ascending order (least constrained)
        sorted_indices = np.argsort(growth_potentials)
        least_constrained_idx = sorted_indices[0]
        second_least = sorted_indices[1]
        third_least = sorted_indices[2]
        
        # Create expanded radius vector: focus expansion on least constrained
        # Apply expansion factor per constraint, with some randomness in expansion for diversity
        expansion_per_circle = expansion_factor + (np.random.rand(n) - 0.5) * 0.0005
            
        # But, give more expansion to the least constrained ones
        expansion_per_circle[least_constrained_idx] += 0.0004
        expansion_per_circle[second_least] += 0.0003
        expansion_per_circle[third_least] += 0.0002
        
        # Clip radii to ensure they do not exceed 0.5 (unit square max)
        new_radii = np.clip(radii + expansion_per_circle, 1e-3, 0.5)
        
        # Re-evaluate with potential expansion, keeping constraints in mind
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        # Validate new configuration and re-apply constraints iteratively
        # We perform a series of refinement loops with decreasing expansion until valid
        # This is more robust than a single expansion pass
        
        # First, apply expansion and check validity
        temp_v = expanded_v
        temp_centers = np.column_stack([temp_v[0::3], temp_v[1::3]])
        
        valid_flag = True
        for i in range(n):
            for j in range(i+1, n):
                dx = temp_centers[i, 0] - temp_centers[j, 0]
                dy = temp_centers[i, 1] - temp_centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < (temp_v[3*i+2] + temp_v[3*j+2]) - 1e-12:
                    valid_flag = False
                    break
            if not valid_flag:
                break
        
        if valid_flag:
            res = minimize(neg_sum_radii, temp_v, method="SLSQP", bounds=bounds, 
                           constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
        else:
            # Need to iteratively adjust expansion to find a valid configuration
            # Start with smaller expansion steps
            for step in range(4, 0, -1):
                # Scale down the proposed expansion
                scale_factor = 0.2 + 0.15 * step
                temp_v = v.copy()
                temp_v[2::3] = radii + expansion_per_circle * scale_factor
                temp_centers = np.column_stack([temp_v[0::3], temp_v[1::3]])
                
                valid_flag = True
                for i in range(n):
                    for j in range(i+1, n):
                        dx = temp_centers[i, 0] - temp_centers[j, 0]
                        dy = temp_centers[i, 1] - temp_centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < (temp_v[3*i+2] + temp_v[3*j+2]) - 1e-12:
                            valid_flag = False
                            break
                    if not valid_flag:
                        break
                
                if valid_flag:
                    res = minimize(neg_sum_radii, temp_v, method="SLSQP", bounds=bounds, 
                                   constraints=cons, options={"maxiter": 150, "ftol": 1e-11})
                    break
                else:
                    # Try a more conservative step
                    scale_factor = 0.05 * step
                    temp_v = v.copy()
                    temp_v[2::3] = radii + expansion_per_circle * scale_factor
                    temp_centers = np.column_stack([temp_v[0::3], temp_v[1::3]])
                    
                    valid_flag = True
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = temp_centers[i, 0] - temp_centers[j, 0]
                            dy = temp_centers[i, 1] - temp_centers[j, 1]
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < (temp_v[3*i+2] + temp_v[3*j+2]) - 1e-12:
                                valid_flag = False
                                break
                        if not valid_flag:
                            break
                    
                    if valid_flag:
                        res = minimize(neg_sum_radii, temp_v, method="SLSQP", bounds=bounds, 
                                       constraints=cons, options={"maxiter": 150, "ftol": 1e-11})
                    else:
                        # Final fallback: use previous result
                        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds, 
                                       constraints=cons, options={"maxiter": 150, "ftol": 1e-11})
        
    # Final verification and clipping
    if res.success:
        v = res.x
    else:
        # Fallback to initial optimization result if not successful
        v = res.x
    
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Ensure minimal radii and prevent overgrowth
    
    return centers, radii, float(radii.sum())
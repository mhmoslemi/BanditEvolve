import numpy as np

def run_packing():
    n = 26
    
    # Initialize positions with randomized spatial hashing and dynamic geometric grid
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    base_width = 1.0 / cols * 0.95
    base_height = 1.0 / rows * 0.95
    
    # Spatial hashing using Voronoi partitioning with adaptive jitter
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid point
        base_x = float(col) * base_width + base_width / 2
        base_y = float(row) * base_height + base_height / 2
        # Add spatial jitter using geometric hashing to prevent grid alignment
        hash_idx = np.random.randint(0, 1000)
        jitter = np.array([0.0, 0.0])
        if hash_idx % 13 == 0:
            jitter[0] = 0.05 * np.random.choice([-1, 1])
        elif hash_idx % 7 == 0:
            jitter[1] = 0.05 * np.random.choice([-1, 1])
        elif hash_idx % 29 == 0:
            jitter[0] = 0.08 * np.random.choice([-1, 1])
        
        # Shift alternate rows to create staggered grid
        x = base_x + jitter[0]
        y = base_y + jitter[1]
        if row % 2 == 1 and np.random.rand() < 0.5:
            x += base_width / 2.5
        # Spatial hashing with non-uniform random shift
        if hash_idx % 17 == 0:
            x = np.random.uniform(base_x - 0.02, base_x + 0.02)
        xs.append(x)
        ys.append(y)
    
    # Compute initial radii with geometric progression and edge bias
    total_width_used = np.max(xs) - np.min(xs)
    total_height_used = np.max(ys) - np.min(ys)
    area_used = total_width_used * total_height_used
    base_area = base_width * base_height
    radius_scale = np.sqrt((area_used / base_area) * (1.0 - 0.01 * n))
    r0 = radius_scale * 0.3 / (cols * rows)
    r0 = np.clip(r0, 0.002, 0.3)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for i in range(n):
        # Use tighter bounds on coordinates and radii based on geometry of the system
        bounds += [(0.0, 1.0 - 2 * r0), (0.0, 1.0 - 2 * r0), (1e-4, 0.48)]  # Slightly smaller bounds for stability

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with lambda with closure variables
    cons = []
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: 1.0 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: 1.0 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing and early filtering
    # Only calculate constraints for non-adjacent points that are likely to be in conflict
    for i in range(n):
        for j in range(i + 1, n):
            # Use geometric hashing to filter out non-interfering pairs
            dx = v0[3*i] - v0[3*j]
            dy = v0[3*i+1] - v0[3*j+1]
            distance_sqr = dx**2 + dy**2
            min_dist = v0[3*i+2] + v0[3*j+2]
            if distance_sqr < (min_dist)**2 * 1.8: 
                # Only add constraint if initial distance is close to sum of radii
                lambda_func = lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
                cons.append({"type": "ineq", "fun": lambda_func})
                # Add a soft constraint that penalizes very tight packing to prevent numerical issues
                cons.append({"type": "ineq", 
                             "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2 * 1.2})
    
    # Initial optimization with tight tolerances, warm start, and multiple stages
    base_options = {
        "maxiter": 600,
        "ftol": 1e-10,
        "gtol": 1e-10,
        "eps": 1e-9,
        "disp": False,
        "return_all": False
    }
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={**base_options, "maxiter": 300}
    )
    
    if res.success:
        # Re-start with more aggressive geometric perturbation and targeted reordering
        v = res.x
        
        # Re-evaluate all points using vectorized distance matrix
        dx = v[0::3] - v[0::3, np.newaxis]
        dy = v[1::3] - v[1::3, np.newaxis]
        dists_sqr = (dx**2 + dy**2)
        min_dists = np.min(dists_sqr, axis=1)
        idx_array = np.argsort(min_dists)
        isolated_idx = idx_array[-1]  # Most isolated
        nearest_indices = idx_array[0:5]  # Nearest
        neighbors = np.unique(np.concatenate([nearest_indices, np.arange(n)]))
        
        # Generate new positions with hybrid reordering and spatial hashing
        new_v = v.copy()
        new_centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply multi-stage reconfiguration
        for perturbation_scale in [0.02, 0.03, 0.015]:
            random_perturbation = np.random.rand(n, 2) * perturbation_scale
            new_centers = new_centers + random_perturbation
            # Enforce boundaries
            for i in range(n):
                new_centers[i, 0] = np.clip(new_centers[i, 0], 0.0, 1.0)
                new_centers[i, 1] = np.clip(new_centers[i, 1], 0.0, 1.0)
            new_v[0::3] = new_centers[:, 0]
            new_v[1::3] = new_centers[:, 1]
            
            # Re-run optimization with new positions
            res = minimize(
                neg_sum_radii,
                new_v,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={**base_options, "maxiter": 200, "ftol": 1e-10}
            )
            if res.success:
                v = res.x
                break
        
        # Post-optimization re-ordering and reconfiguration with spatial hashing
        if res.success:
            dx = v[0::3] - v[0::3, np.newaxis]
            dy = v[1::3] - v[1::3, np.newaxis]
            dists_sqr = dx**2 + dy**2
            # Use spatial hashing to find most isolated circles
            min_dists = np.min(dists_sqr, axis=1)
            sorted_idx = np.argsort(min_dists)
            isolated_idx = sorted_idx[-1]  # Most isolated
            
            # Spatial hashing to find high-impact areas
            hash_grid = np.zeros((5, 5))  # 5x5 spatial grid
            hash_idx = np.arange(n)
            # Assign hash grid based on coordinates
            grid_x = np.floor(v[0::3] * 5).astype(int)
            grid_y = np.floor(v[1::3] * 5).astype(int)
            # Ensure all indices in 0-4
            grid_x = np.clip(grid_x, 0, 4)
            grid_y = np.clip(grid_y, 0, 4)
            grid_idx = (grid_y * 5 + grid_x).astype(int)
            # Group indices by grid cell to find high-impact areas
            grid_idx_to_indices = {}
            for idx, g in zip(range(n), grid_idx):
                if g not in grid_idx_to_indices:
                    grid_idx_to_indices[g] = []
                grid_idx_to_indices[g].append(idx)
            grid_dists = np.zeros(25)
            # Evaluate distances in each grid cell
            for cell_idx in grid_idx_to_indices:
                if cell_idx in grid_idx_to_indices:
                    cell_indices = grid_idx_to_indices[cell_idx]
                    for i in cell_indices:
                        for j in cell_indices:
                            if i < j:
                                dx = v[3*i] - v[3*j]
                                dy = v[3*i+1] - v[3*j+1]
                                dist_sqr = dx**2 + dy**2
                                grid_dists[cell_idx] = max(grid_dists[cell_idx], dist_sqr)
            # Find highest-impact cell (most distance between neighbors)
            impact_idx = np.argmax(grid_dists)
            # Get affected indices
            impact_indices = grid_idx_to_indices.get(impact_idx, [])
            
            # Targeted expansion based on reordering
            new_radii = v[2::3]
            if len(impact_indices) > 0 and isolated_idx in impact_indices:
                # Expand isolated circle with geometric constraint
                expansion_factor = 0.006 / (n - 1) * 1.5
                new_radii[isolated_idx] += expansion_factor
            if len(impact_indices) > 4:
                # Expand all in the high-impact area by 10%
                expansion = (0.01 * (n - 1)) * (1.0 / (n - 1))
                new_radii[impact_indices] += expansion
            
            # Apply expansion in a safe way with constraint validation
            while True:
                # Construct new vector
                new_v = v.copy()
                new_v[2::3] = new_radii
                new_centers = np.column_stack([new_v[0::3], new_v[1::3]])
                # Validate new radii against constraints
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = new_centers[i, 0] - new_centers[j, 0]
                        dy = new_centers[i, 1] - new_centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < new_radii[i] + new_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    # Reduce expand factor if invalid
                    expansion_factor = max(expansion_factor * 0.95, 0.001)
                    new_radii = v[2::3] + (new_radii - v[2::3]) * 0.98
        
        # Final optimization
        if res.success:
            res = minimize(
                neg_sum_radii,
                new_v,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={**base_options, "maxiter": 200, "ftol": 1e-12}
            )
    
    # Final validation with fallback to original solution if needed
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final constraint validation for edge cases
    if not validate_packing(centers, radii)[0]:
        # If validation fails, revert to fallback solution
        # Use a simpler initialization with spatial hashing but more direct
        default_v = np.empty(3 * n)
        default_v[0::3] = np.linspace(0.0, 1.0, n)
        default_v[1::3] = np.linspace(0.0, 1.0, n)
        default_v[2::3] = 0.01 * np.ones(n)
        centers_def = np.column_stack([default_v[0::3], default_v[1::3]])
        radii_def = np.clip(default_v[2::3], 1e-6, None)
        centers, radii, _ = validate_packing(centers_def, radii_def)
        v = np.concatenate([centers.flatten(), radii])
        v = np.clip(v, 1e-6, 0.5)
        
    return centers, radii, float(radii.sum())
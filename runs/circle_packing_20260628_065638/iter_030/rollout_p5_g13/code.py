import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Step 1: Initialize centers with a hexagonal grid + perturbations + soft spatial hashing + initial radius calculation
    # Spatial perturbation is more aggressive for rows with larger inter-cell spacing
    xs = []
    ys = []
    # Spatial hashing is not random but structured with adaptive bias to reduce symmetry
    # Initial radii are calculated via circle packing formula with a dynamic scaling factor
    initial_r = 0.35 / cols - 1e-3  # base radii, will be refined
    row_weight = 1.2  # higher weight for upper rows to allow more expansion vertically
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        # Base center: hexagonal grid, but with adaptive spacing
        # x_center is col-based with a fixed offset
        x_base = (col_idx + 0.5) / cols
        # y_center with staggered rows
        # rows are stacked: row 0, row 1 shifted, row 2 back, etc.
        # Use row_idx parity to determine vertical shift
        y_base = (row_idx + 0.5) / rows
        # Add stagger for second row (row_idx % 2 == 1)
        if row_idx % 2 == 1:
            y_base += 0.25 / rows
        # Spatial hashing based on row and column to avoid perfect grid patterns
        hash_row = row_idx % 3  # low variance
        hash_col = col_idx % 2  # low variance
        # Perturb with a bias toward row spacing and adaptive amplitude
        if hash_row == 0:  # perturb rows closer to the top to allow for greater radius scaling
            x_offset = np.random.uniform(-0.08, 0.08) * (1.0 + 0.25 * (1.0 - row_idx / rows))
            y_offset = np.random.uniform(-0.10, 0.10) * (1.0 + 0.25 * (1.0 - row_idx / rows))
        else:  # perturb more in central and lower rows
            x_offset = np.random.uniform(-0.05, 0.05)
            y_offset = np.random.uniform(-0.05, 0.05)
        x_centered = x_base + x_offset
        y_centered = y_base + y_offset
        
        xs.append(x_centered)
        ys.append(y_centered)
    
    # Step 2: Optimize with better bounds and initial constraints, adding constraints for edge effects
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, initial_r) * np.sqrt(1.0 + 0.1 * np.random.rand())  # slight stochastic radius scaling for diversity

    # Bounds structure ensures 3 * n parameters, consistent with vector size
    bounds = []
    for _ in range(n):
        # Centers
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # radii have lower bound

    # Optimizer objective: maximize sum of radii by minimizing negative sum (convenience for SLSQP)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # minimize negative sum = maximize radii

    # Step 3: Constraints are now structured explicitly - boundary constraints, inter-circle constraints, and soft constraints
    cons = []
    for i in range(n):
        # Bound constraints for centers
        # x center: x_min = radius[i], x_max = 1 - radius[i]
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # y center: same logic
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
        # Soft constraint: add a "pressure" to maintain inter-circle distance to allow for better local minima
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1e-4 - v[3*i + 2] * (1.0 / (np.sqrt(5.0) ))})  # small pressure to keep radii from exploding
        
    # Step4: Generate overlap constraints with advanced geometric hashing and constraint prioritization
    # For efficiency, we group circles with close spatial proximity with a geometric hash
    # Use a grid hashing function to precompute nearby circles for overlap constraints (instead of O(n^2) constraints)
    # The spatial hashing is based on 2D grid and will reduce the number of required constraints
    # This enables us to add all pairwise constraints, but only for neighboring circles via grid hashing
    from collections import defaultdict
    spatial_hash_gridsize = (4, 4)  # grid size per spatial hashing
    hash_index = lambda x, y, gridsize: (int(x * gridsize[0]), int(y * gridsize[1]))
    # Create spatial hash map
    spatial_hash_map = defaultdict(list)
    for i in range(n):
        x = xs[i]
        y = ys[i]
        hash_key = hash_index(x, y, spatial_hash_gridsize)
        spatial_hash_map[hash_key].append(i)
    
    # Now process all circles in the hash map and generate overlap constraints
    # This reduces constraint count while maintaining coverage
    overlapping_pairs = set()  # to avoid double counting
    for hash_key in spatial_hash_map:
        circle_indices = spatial_hash_map[hash_key]
        for i in circle_indices:
            for j in circle_indices:
                if i < j:  # ensure unique pairs
                    overlapping_pairs.add((i, j))
    
    # Generate the constraint functions
    # Use a lambda per constraint, but with closure
    for i, j in overlapping_pairs:
        center_i_idx = 3 * i
        center_j_idx = 3 * j
        radii_i_idx = center_i_idx + 2
        radii_j_idx = center_j_idx + 2
        def constraint_func(v, i=i, j=j):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist_sq = dx * dx + dy * dy
            min_dist_sq = (v[3*i+2] + v[3*j+2]) ** 2
            # constraint is dist_sq >= min_dist_sq to prevent overlapping
            return dist_sq - min_dist_sq  # positive for feasible
        cons.append({"type": "ineq", "fun": constraint_func})

    # Additional constraint: ensure no circle is too large, preventing total radius from exploding
    # This avoids some edge cases where all radii might increase unboundedly
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 2.0 - v[3*i + 2]})  # prevent radii exceed 2
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 2] - 1e-3})  # prevent radii from being too small
    
    # Step5: Initial optimization with tighter tolerances and more aggressive optimization
    # We use a combination of methods with adaptive scaling for better results
    # First optimize with high iteration count and moderate tolerance
    # Then, if not successful, reconfigure positions in a way that increases inter-circle spacing
    # Finally, do a final optimization with high precision to refine

    # First optimization with 5000 iterations with tight tolerance
    res1 = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-12})
    
    if res1.success:
        # If success, perform spatial hashing and perturbation (but with fixed constraints)
        # This is a targeted perturbation of centers that can allow for better radii expansion without violating constraints
        v = res1.x
        # Spatial reconfiguration: generate adaptive hashes to slightly shift centers in directions that increase usable space
        # Spatial hashing here is deterministic, to reduce randomness that might lead to suboptimal local minima
        # This step is done only once after the primary optimization
        perturbation_gridsize = (6, 6)
        spatial_hash_grid = np.zeros((perturbation_gridsize[0], perturbation_gridsize[1]))
        for i in range(n):
            x = v[3*i]
            y = v[3*i+1]
            grid_x = int(x * perturbation_gridsize[0])
            grid_y = int(y * perturbation_gridsize[1])
            grid_x %= perturbation_gridsize[0]
            grid_y %= perturbation_gridsize[1]
            # For a grid, apply a small offset to nearby circles
            if grid_x > 0:
                spatial_hash_grid[grid_x - 1, grid_y] += 0.04 * (1.0 + 0.1 * np.random.rand())  # small displacement
            if grid_x < perturbation_gridsize[0] - 1:
                spatial_hash_grid[grid_x + 1, grid_y] += 0.04 * (1.0 + 0.1 * np.random.rand())
            if grid_y > 0:
                spatial_hash_grid[grid_x, grid_y - 1] += 0.04 * (1.0 + 0.1 * np.random.rand())
            if grid_y < perturbation_gridsize[1] - 1:
                spatial_hash_grid[grid_x, grid_y + 1] += 0.04 * (1.0 + 0.1 * np.random.rand())
        # Apply this hash as a reconfiguration delta
        perturbed_v = v.copy()
        for i in range(n):
            x = v[3*i]
            y = v[3*i+1]
            grid_x = int(x * perturbation_gridsize[0])
            grid_y = int(y * perturbation_gridsize[1])
            grid_x %= perturbation_gridsize[0]
            grid_y %= perturbation_gridsize[1]
            # Apply per-cell perturbation
            perturbation_x = spatial_hash_grid[grid_x, grid_y] * (1.0 + 0.1 * np.random.rand())
            perturbation_y = spatial_hash_grid[grid_x, grid_y] * (1.0 + 0.1 * np.random.rand())
            perturbed_v[3*i] += perturbation_x
            perturbed_v[3*i+1] += perturbation_y
        
        # Now do a second pass with this new configuration to refine
        res2 = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-12})
        
        # If second optimization is successful, perform a final optimization with high-precision tuning
        if res2.success:
            # Final optimization
            res2final = minimize(neg_sum_radii, res2.x, method="SLSQP", bounds=bounds,
                                constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-12})
            res = res2final
        else:
            res = res1
    else:
        res = res1

    # Final adjustment: after optimization, perform a check on smallest radius and apply a targeted expansion
    # First, if any circle has a smaller radius than a threshold, apply expansion
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    
    # Identify the smallest circle (not zero, which is handled by the clip)
    valid_radii = radii[radii > 1e-6]
    if len(valid_radii) > 0:
        smallest_radius_idx = np.argmin(radii[radii > 1e-6])
        smallest_radius = radii[smallest_radius_idx]
        # Calculate the minimal expansion factor while keeping total radius within bounds
        # Assume the smallest circle can expand by up to 30% if other circles are at their max
        expansion_max_ratio = 0.35
        max_allowed_expansion = min(expansion_max_ratio, 2.0 - smallest_radius)  # prevent radii from exceeding 2.0
        expansion_ratio = max_allowed_expansion * (4.0 / 3.0)  # apply slight over-expansion
        expansion = smallest_radius * expansion_ratio

        # Apply this expansion with constraint checking
        new_v = v.copy()
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion
        
        # Check if expansion keeps the constraint valid
        expanded_centers = np.column_stack([new_v[0::3], new_v[1::3]])
        expansion_valid = True
        # For performance, we check overlaps only with neighboring circles
        # (this is an optimization to speed up the check)
        for i in range(n):
            for j in range(i + 1, n):
                if i == smallest_radius_idx or j == smallest_radius_idx:
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < (new_radii[i] + new_radii[j]) - 1e-12:
                        expansion_valid = False
                        break
            if not expansion_valid:
                break
        if expansion_valid:
            # Apply the expansion if it's valid
            new_v[2::3] = new_radii
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 150, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
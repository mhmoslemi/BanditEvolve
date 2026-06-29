import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with adaptive geometric clustering and dynamic layout
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with variable spacing for adaptability
        cell_w = 1.0 / cols
        cell_h = 1.0 / rows
        x_center = col * cell_w + 0.5 * cell_w
        y_center = row * cell_h + 0.5 * cell_h
        
        # Adaptive offset based on grid spacing
        x_offset = np.random.uniform(-0.02, 0.02)
        y_offset = np.random.uniform(-0.02, 0.02)
        
        # Staggered grid to minimize inter-circle crowding
        if row % 2 == 1:
            x_center += 0.5 * cell_w
            x_offset += np.random.uniform(-0.03, 0.03)
        
        x = x_center + x_offset
        y = y_center + y_offset
        
        xs.append(x)
        ys.append(y)
    
    # Set initial radii with adaptive scale depending on grid cell size
    cell_area = (1.0 / cols) * (1.0 / rows)
    avg_radius_scalar = np.sqrt(cell_area) * 0.75
    r0 = avg_radius_scalar - (1e-3 * np.sqrt(cell_area))  # Ensure small radius buffer
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure 3*n entries for the vector of length 3*n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.55)]  # Use smaller min radius buffer
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup with lambda captures
    cons = []
    for i in range(n):
        # Left + radius <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with vectorized handling
    # Instead of per pair evaluation, use a spatial hash to reduce computational density
    # This is an adaptive constraint method to limit evaluations on most-interacting pairs
    # First, create a spatial hash grid
    spatial_hash_size = 5
    hash_grid = np.zeros((spatial_hash_size, spatial_hash_size, n))
    hash_cell_size = 1.0 / spatial_hash_size
    for i in range(n):
        x = v0[3*i]
        y = v0[3*i+1]
        hash_x = int(np.floor(x / hash_cell_size))
        hash_y = int(np.floor(y / hash_cell_size))
        hash_grid[hash_y, hash_x, i] = 1
    
    # Now, build constraints only for neighboring hash cells
    hash_constraints = []
    for h_x in range(spatial_hash_size):
        for h_y in range(spatial_hash_size):
            for i in range(n):
                if hash_grid[h_y, h_x, i] == 0:
                    continue
                # Expand in all 8 directions and create constraints only if neighbors exist
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_hash_x = h_x + dx
                        neighbor_hash_y = h_y + dy
                        if neighbor_hash_x < 0 or neighbor_hash_x >= spatial_hash_size or neighbor_hash_y < 0 or neighbor_hash_y >= spatial_hash_size:
                            continue
                        for j in range(n):
                            if hash_grid[neighbor_hash_y, neighbor_hash_x, j] == 0:
                                continue
                            # Only add constraints between i and j if they are not the same and not already added
                            if i == j:
                                continue
                            # Check if we've already created a constraint between this pair
                            if (i, j) in hash_constraints or (j, i) in hash_constraints:
                                continue
                            # Add constraint for this pair
                            hash_constraints.append((i, j))
    
    # Now build constraints for all hash-constrained pairs
    for i, j in hash_constraints:
        # Create constraint for i and j
        def constraint_func(v, i=i, j=j):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
        
        cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-9})
    
    # Dynamic post-optimization with adaptive symmetry breaking and radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # First: symmetry breaking via adaptive spatial hashing
        hash_grid = np.zeros((spatial_hash_size, spatial_hash_size, n))
        hash_cell_size = 1.0 / spatial_hash_size
        for i in range(n):
            x = v[3*i]
            y = v[3*i+1]
            hash_x = int(np.floor(x / hash_cell_size))
            hash_y = int(np.floor(y / hash_cell_size))
            hash_grid[hash_y, hash_x, i] = 1
        
        # Apply symmetric perturbation across spatially clustered circles
        sym_perturbation = np.random.uniform(-0.01, 0.01, size=(n, 2))
        v_perturbed = v.copy()
        for i in range(n):
            hash_x = int(np.floor(v[3*i] / hash_cell_size))
            hash_y = int(np.floor(v[3*i+1] / hash_cell_size))
            if hash_grid[hash_y, hash_x, i] > 0:
                v_perturbed[3*i] += sym_perturbation[i, 0] * (radii[i] / np.mean(radii))
                v_perturbed[3*i+1] += sym_perturbation[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with perturbed vector
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-9})
    
    # Step 2: Adaptive global expansion via constrained optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute vectorized pairwise distances and min distances per circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate possible expansion based on current total and spatial distribution
        current_total = np.sum(radii)
        max_safe_growth = 0.006  # Safety margin for radius expansion
        avg_radius = current_total / n
        expansion_factor = max_safe_growth / n
        
        # Create expansion vector with targeted expansion
        expansion = np.zeros(n)
        expansion[least_constrained_idx] = expansion_factor * 1.2  # Slight over-expansion for aggressive growth
        for i in range(n):
            if i != least_constrained_idx:
                expansion[i] = expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
        
        # Apply expansion with safety check
        while True:
            new_radii = radii + expansion
            # Ensure new radii do not exceed max allowed
            new_radii = np.clip(new_radii, 1e-6, 0.55)
            v_expanded = v.copy()
            v_expanded[2::3] = new_radii
            expanded_centers = np.column_stack([v_expanded[0::3], v_expanded[1::3]])
            
            # Check for overlaps with tolerance
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                expansion *= 0.95  # Reduce expansion slightly
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    # Final refinement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Final local expansion check using adaptive radius constraints
        for i in range(n):
            # Add an adaptive radius increase constraint only if its radius is at least 5% below average
            if radii[i] < np.mean(radii) * 0.95:
                # Create a constraint to prevent radius from exceeding average by 10%
                avg_radius = np.mean(radii)
                max_radius_i = avg_radius * 1.1
                max_radius_i = min(max_radius_i, 0.55)
                cons.append({"type": "ineq", "fun": lambda v, i=i: max_radius_i - v[3*i+2]})
        
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
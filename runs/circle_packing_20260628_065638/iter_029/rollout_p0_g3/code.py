import numpy as np

def run_packing():
    n = 26
    
    # Adaptive grid configuration using a hybrid geometric-quantum placement strategy
    cols_x = 5
    rows_y = 5
    col_indices = np.random.permutation(n)
    
    # Initialize with quantum-inspired spatial perturbation and hexagonal tiling
    xs = []
    ys = []
    for i in range(n):
        col = col_indices[i] % cols_x
        row = col_indices[i] // cols_x
        # Base grid with hexagonal spacing
        x_center = (col + 0.5) / cols_x
        y_center = (row + 0.5) / rows_y
        # Quantum displacement to break symmetry and allow non-linear expansion
        random_displacement = np.random.uniform(-0.04, 0.04, 2)
        quantum_offset = np.random.rand(2) * np.abs(random_displacement) * 0.3
        x = x_center + random_displacement[0] + quantum_offset[0]
        y = y_center + random_displacement[1] + quantum_offset[1]
        # Alternate row vertical compression to simulate hexagonal tiling
        if row % 2 == 1:
            y += (1 / rows_y) * 0.1  # Slight vertical adjustment
        xs.append(x)
        ys.append(y)
    
    # Initial radii estimation using non-linear expansion with local and global constraints
    # Using geometric spacing and adaptive radius scaling based on proximity
    r0 = 0.5 / (cols_x + rows_y) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint system with dual-level enforcement - 
    # 1. Boundary constraints with strict tolerance
    # 2. Overlap constraints with soft constraint relaxation via penalty terms
    
    # Boundary constraints with strict enforcement (inequalities)
    cons = []
    for i in range(n):
        # Left side constraint with 1e-12 tolerance
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2] + 1e-12})
        # Right side constraint with 1e-12 tolerance
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] + 1e-12})
        # Bottom side constraint with 1e-12 tolerance
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2] + 1e-12})
        # Top side constraint with 1e-12 tolerance
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] + 1e-12})

    # Overlap constraints with adaptive soft constraint relaxation
    # Each overlap constraint has:
    # - A hard constraint (distance between centers ≥ sum of radii)
    # - A soft penalty term with adaptive scaling to explore edge cases
    for i in range(n):
        for j in range(i + 1, n):
            # Adaptive scaling factor based on distance and spatial constraint pressure
            def calc_overlap_func(i, j):
                def func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    radii_sum = v[3*i+2] + v[3*j+2]
                    # Soft penalty with adaptive scaling factor based on proximity
                    # Use log scale to give more flexibility to marginally constrained circles
                    penalty = np.log(max(1e-6, dist - radii_sum + 1e-12)) * 0.03
                    return dist - radii_sum + penalty
                return func
            cons.append({"type": "ineq", "fun": calc_overlap_func(i, j)})

    # First optimization with increased max iterations and enhanced tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-9})
    
    # Asymmetric spatial reconfiguration with guided randomization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate guided spatial hash based on spatial gradient
        # Use radius and spatial constraints to inform perturbation magnitude
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            # Weight spatial perturbation by distance to walls and neighbor distances
            edge_distance = np.min([v[3*i] - 1e-5, 1 - v[3*i] - 1e-5,
                                   v[3*i+1] - 1e-5, 1 - v[3*i+1] - 1e-5])
            neighbor_distances = np.zeros(n)
            for j in range(n):
                if j != i:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    neighbor_distances[j] = dist - (radii[i] + radii[j]) + 1e-12
            constraint_pressure = np.sum(np.abs(neighbor_distances)) + edge_distance
            if constraint_pressure > 0:
                scale_factor = 0.05 * (1 + np.log(constraint_pressure))
            else:
                scale_factor = 0.05
            perturbed_v[3*i] += spatial_hash[i, 0] * scale_factor * radii[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale_factor * radii[i]
        
        # Re-evaluate with new spatial configuration and hybrid constraint enforcement
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    # Targeted expansion phase with influence-driven gradient analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dist_mat = np.zeros((n, n))
        
        # Vectorized distance computation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist_mat = np.sqrt(dx**2 + dy**2)
        
        # Compute influence map based on distance and spatial constraints
        # Influence weight = 1/(distance + 1e-8) for circles far from walls
        dist_to_walls = np.min(np.stack([
            centers[:, 0] - 1e-5, 1 - centers[:, 0] - 1e-5,
            centers[:, 1] - 1e-5, 1 - centers[:, 1] - 1e-5
        ]), axis=0)
        influence_weights = 1 / (dist_mat + 1e-8)
        influence_weights *= np.exp(-20 * (distance_to_wall_min / 0.02))
        
        # Normalize influence to create an expansion priority vector
        influence_sum = np.sum(influence_weights, axis=1)
        influence_normalized = influence_weights / (influence_sum[:, np.newaxis] + 1e-8)
        
        # Identify least constrained circle - find circle with highest influence spread
        least_constrained_idx = np.argmin(influence_normalized.min(axis=1))
        
        # Calculate growth based on current total sum and expansion potential
        current_total = np.sum(radii)
        # Estimate expansion potential based on spatial constraint margins
        expansion_potential = np.min([np.min(centers[:, 0] - 1e-5 - radii),
                                    np.min(1 - centers[:, 0] - 1e-5 - radii),
                                    np.min(centers[:, 1] - 1e-5 - radii),
                                    np.min(1 - centers[:, 1] - 1e-5 - radii)]) 
        expansion_factor = max(0.0005, 0.02 * expansion_potential / 0.0005)
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        # Base expansion for least constrained circle
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        # Add stochastic expansion to other circles based on influence
        for i in range(n):
            if i != least_constrained_idx:
                # Higher influence ⇒ more expansion
                expansion_i = expansion_factor * (1.0 + 1.2 * influence_normalized[i, np.arange(n)] / np.max(influence_normalized[i, np.arange(n)]))
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation using vectorized checks
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration using vectorized pairwise checks
            valid = True
            # Compute pairwise distances and check overlap constraint
            dx = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            for i in range(n):
                for j in range(i + 1, n):
                    if dists[i, j] < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion slightly and try again
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration using hybrid strategy
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
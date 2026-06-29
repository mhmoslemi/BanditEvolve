import numpy as np

def run_packing():
    n = 26
    cols = 6  # increase grid width for finer spatial control at lower density
    rows = (n + cols - 1) // cols  # dynamic rows
    
    # Initialize with geometric clustering and dynamic cell size adaptation
    xs = []
    ys = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        
        # Calculate cell size based on remaining space
        remaining_width = (1.0 - (col_idx + 1) / cols)
        remaining_height = (1.0 - (row_idx + 1) / rows)
        cell_size = np.sqrt(remaining_width * remaining_height)
        if cell_size < 0.1:
            cell_size = 0.1
        
        # Base center point with geometric bias toward top-left
        x_center_base = (col_idx + 0.5) / cols + (np.random.uniform(-0.03, 0.03) * cell_size)
        y_center_base = (row_idx + 0.5) / rows + (np.random.uniform(-0.03, 0.03) * cell_size)
        
        # Add randomized distortion with adaptive magnitude
        dx = np.random.uniform(-0.04 * cell_size, 0.04 * cell_size)
        dy = np.random.uniform(-0.04 * cell_size, 0.04 * cell_size)
        x_center = x_center_base + dx
        y_center = y_center_base + dy
        
        # Implement staggered row distortion for complex packing
        if row_idx % 2 == 0:
            x_center += np.random.uniform(-0.015 * cell_size, 0.015 * cell_size)
        xs.append(x_center)
        ys.append(y_center)
    
    # Initial radii with cell-based scaling and spatial variation factor
    cell_based_radii = 0.25 / np.sqrt(np.mean([(col_idx + 0.5 / cols)**2 + (row_idx + 0.5 / rows)**2 for row_idx, col_idx in zip([i // cols for i in range(n)], [i % cols for i in range(n)])]))
    r0 = cell_based_radii - np.random.uniform(0.02, 0.06)  # spatially variable initial values
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.clip(np.full(n, r0), 1e-4, 0.45)  # upper bound slightly tightened

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]  # strict upper bound on radii (tighter than original)

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with proper lambda closure and i tracking
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
    
    # Vectorized overlap constraint with geometric hashing and adaptive penalty scaling
    # Use vectorization to avoid nested loops in constraints
    # We compute pairwise distances once at setup using vector math
    # Generate distance matrix with square root and precompute min distances
    centers = np.array([xs, ys]).T
    dists_full = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    # Create constraints that enforce distance >= (r_i + r_j) for each pair
    for i in range(n):
        for j in range(i+1, n):
            # Use vectorized version with lambda capturing i and j
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: dists_full[i][j] - v[3*i+2] - v[3*j+2]})  # Note: dists_full is global; this is an optimization bottleneck
    
    # Initialize with advanced perturbation based on spatial and radii dynamics
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-9})
    
    # Forced geometric dissection - isolate interaction dynamics and reconfigure critical pairs
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        
        # Identify the two circles with greatest interaction (max pairwise distance to all other circles)
        interaction_weight = np.sum(dists, axis=1)
        top_idx = np.argsort(interaction_weight)[-2:]  # most interactive pair
        
        # Apply dynamic spatial perturbations to this pair while keeping others static
        perturbation_scale = 0.02 * np.sqrt(np.max(radii))
        for idx in top_idx:
            # Randomly perturb positions with adaptive magnitude
            v[3*idx] += np.random.uniform(-perturbation_scale, perturbation_scale)
            v[3*idx+1] += np.random.uniform(-perturbation_scale, perturbation_scale)
            # Adjust radii while keeping their interaction constraints
            v[3*idx+2] += np.random.uniform(-0.003, 0.003)
            v[3*idx+2] = np.clip(v[3*idx+2], 1e-4, 0.45)
        
        # Enforce new configurations via re-optimization with adaptive constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    
    # Novel adjacency constraint: force a dynamic topological rearrangement
    # Introduce a "critical connection" between the two least isolated circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        
        # Calculate isolation by average min distance, not sum
        avg_min_distance = np.mean(np.min(dists, axis=1))
        isolated_idx = np.argsort(np.mean(dists, axis=0))[:2]  # least isolated pair
        critical_idx1, critical_idx2 = isolated_idx
        
        # Introduce adjacency constraint between them by enforcing minimum distance
        # We add a constraint to maintain at least 0.02 units apart
        def adjacency_func(v, i=critical_idx1, j=critical_idx2):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return np.sqrt(dx**2 + dy**2) - (radii[i] + radii[j]) + 0.02
        
        cons.append({"type": "ineq", "fun": lambda v: adjacency_func(v, critical_idx1, critical_idx2)})
        
        # Re-optimization with this new constraint to force topological reordering
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-9})

    # Constrained radius expansion on least constrained circle with dynamic scaling
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        
        # Calculate isolation by average min distance, not sum
        avg_min_distance = np.mean(np.min(dists, axis=1))
        # Find the most isolated circle
        isolations = np.min(dists, axis=1)
        isolated_idx = np.argmin(isolations)
        
        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        growth_factor = 0.008  # target expansion
        # Use adaptive expansion factor (higher for more isolated nodes)
        expansion_factor = growth_factor * (1 + 0.2 * (np.min(isolations) / avg_min_distance))
        
        # Create an expansion vector to distribute additional radius
        expanded_radii = radii.copy()
        expanded_radii[isolated_idx] = radii[isolated_idx] + expansion_factor  # directly expand this circle
        # Distribute remaining expansion to others based on their isolation
        total_expanded = np.sum(expanded_radii) - current_total
        for i in range(n):
            if i != isolated_idx:
                expansion_i = expansion_factor * (np.min(dists[i]) / avg_min_distance) * np.sqrt(radii[i])
                expanded_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(expanded_radii, 1e-4, 0.45)  # maintain upper bound
                
            # Evaluate for validity with early breaking
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_v[3*i] - expanded_v[3*j]
                    dy = expanded_v[3*i+1] - expanded_v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < expanded_radii[i] + expanded_radii[j] - 2e-8:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Gradual reduction of expansion
                expansion_factor *= 0.95
                # Recalculate expansion based on new factor
                total_expanded = np.sum(expanded_radii) - current_total
                for i in range(n):
                    if i != isolated_idx:
                        expansion_i = expansion_factor * (np.min(dists[i]) / avg_min_distance) * np.sqrt(radii[i])
                        expanded_radii[i] -= expansion_i/2
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = np.clip(expanded_radii, 1e-4, 0.45)
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-9})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
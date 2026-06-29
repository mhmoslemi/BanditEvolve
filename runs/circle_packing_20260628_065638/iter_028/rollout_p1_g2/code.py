import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.05, 0.05) * (1.2 if row % 2 == 1 else 1.0)
        y = y_center + np.random.uniform(-0.05, 0.05) * (1.2 if row % 2 == 1 else 1.0)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Improve initial radius estimation based on grid density and spatial hashing
    grid_density = (np.sqrt(n) / (cols * rows)) ** 2
    r0 = (1.0 / (cols + rows)) * (0.35 + (0.15 * grid_density)) - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with closure avoidance using lambda with i
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with closure avoidance using lambda with i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Apply radical non-local reconfiguration using dynamic spatial hashing and directional expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate dynamic spatial hashing for grid reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.04 * (1.0 + 0.2 * np.random.rand(n))
        spatial_hash = np.clip(spatial_hash, -0.03, 0.03)
        
        # Generate dynamic adjacency hashing using geometric neighborhood
        adjacency_hash = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0 and dist < 0.15:
                    adjacency_hash[i] += 0.1 * np.array([dx, dy]) * (1.0 / dist)
                    adjacency_hash[j] += 0.1 * np.array([-dx, -dy]) * (1.0 / dist)
        
        # Perturb positions in grid-reconfiguration mode
        perturbed_v = v.copy()
        for i in range(n):
            # Apply spatial hash with direction based on grid cell neighbor relationships
            row = i // cols
            col = i % cols
            base_x = (col + 0.5) / cols
            base_y = (row + 0.5) / rows
            perturbed_v[3*i] += spatial_hash[i, 0] * (1.0 + 0.2 * np.random.rand()) * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1.0 + 0.2 * np.random.rand()) * (radii[i] / np.mean(radii))
            
            # Apply directional expansion based on adjacency hash and neighbor relationships
            if i < n - 2:
                perturbed_v[3*i+2] += adjacency_hash[i, 0] * 0.004
            
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Apply targeted spatial hashing and radial expansion to least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting with memory optimization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)  # 1176 elements (26*26)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate growth budget based on current radius distribution and spatial hashing
        current_total = np.sum(radii)
        target_growth = 0.01  # 1% increase total
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Generate directional growth hashing for enhanced expansion
        direction_hash = np.random.rand(n, 2) * 0.05
        direction_hash = np.clip(direction_hash, -0.025, 0.025)
        
        # Apply directional expansion with dynamic adjustment based on spatial context
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.25  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Directional expansion based on spatial hashing and neighborhood density
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                expansion_factor_i = expansion_factor * (1.0 + 0.1 * direction_hash[i, 0])
                if adj_weight < 0.15:
                    expansion_factor_i *= 1.5  # Boost for nearby circles
                else:
                    expansion_factor_i *= 0.98  # Reduce for distant circles
                
                direction_component = np.random.rand() * 0.02 * direction_hash[i, 1]
                if np.random.rand() < 0.1:
                    direction_component *= 0.5
                new_radii[i] += expansion_factor_i + direction_component
        
        # Apply expansion with constraint validation
        while True:
            # Apply direct expansion in a constrained way
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly for all circles
                new_radii = radii + (new_radii - radii) * 0.95
            
        # Update decision vector with reconfigured positions and expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
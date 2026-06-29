import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with non-uniform, hexagonal tiling with adaptive offset and spatial hashing
    xs = []
    ys = []
    r0 = 0.35 / cols - 1e-3
    
    # Adaptive spacing based on circle diameter and neighbor proximity
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Geometric hashing to avoid symmetric clustering: use hash of circle index
        hash_x = (i * 937) % cols
        hash_y = (i * 713) % rows
        x_offset = (hash_x - cols / 2) * 0.06
        y_offset = (hash_y - rows / 2) * 0.06
        
        # Dynamic offset based on grid position and spacing density
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Alternate row staggering: more dynamic than fixed offset
        if row % 2 == 1:
            x += 0.5 / cols * np.sin(row * 0.5)
        
        xs.append(x)
        ys.append(y)
    
    # Initialize decision vector from coordinates and radius
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    # Ensure bounds list has 3*n items matching the vector's length
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # x, y, r
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized non-overlap constraints using closure capture and lambda
    # Note: this is a critical refactor over the parent's constraint handling
    
    # Spatial hashing for constraint reconfiguration
    cons = []
    # Define all per-circle constraints (boundaries) with spatial hashing
    for i in range(n):
        # Use hash-based offset to avoid symmetry
        spatial_hash = (i * 1013) % 1000
        # Constraint 1: x - r >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i, s=spatial_hash: v[3*i] - (v[3*i+2] + (s * 0.001))})
        # Constraint 2: 1 - x - r >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i, s=spatial_hash: (1.0 - v[3*i] - (v[3*i+2] + (s * 0.001)))})
        # Constraint 3: y - r >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i, s=spatial_hash: v[3*i+1] - (v[3*i+2] + (s * 0.001))})
        # Constraint 4: 1 - y - r >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i, s=spatial_hash: (1.0 - v[3*i+1] - (v[3*i+2] + (s * 0.001)))})
    
    # Vectorized pairwise constraints with optimized computation
    for i in range(n):
        for j in range(i+1, n):
            # Use geometric hashing for pairwise constraints
            spatial_hash = (i + j) * 137 % 1000
            
            # Create lambda with captured index and hash
            def make_overlap_func(i, j, hash):
                def constraint_func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    distance_sq = dx*dx + dy*dy
                    # Add geometric hashing penalty to prevent symmetry
                    distance_penalty = (np.sin(hash * 0.01) ** 2) * (v[3*i+2] + v[3*j+2])
                    return distance_sq - (v[3*i+2] + v[3*j+2])**2 - distance_penalty
                return constraint_func
            
            cons.append({
                "type": "ineq",
                "fun": make_overlap_func(i, j, spatial_hash)
            })
    
    # Multi-step optimization with hierarchical refinement
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-11})
    
    # Spatial hashing reconfiguration phase with gradient-enhanced perturbation
    if res.success:
        v = res.x
        # Generate spatial hash for all circles with adaptive scaling
        spatial_hashes = (np.arange(n) * 179) % 1000
        perturbed_v = v.copy()
        
        # Perturb positions using spatial hash with adaptive radius-based scaling
        for i in range(n):
            x_perturb = (np.sin(spatial_hashes[i] * 0.003) * v[2+3*i] * 0.3)
            y_perturb = (np.cos(spatial_hashes[i] * 0.003) * v[2+3*i] * 0.3)
            
            perturbed_v[3*i] += x_perturb
            perturbed_v[3*i+1] += y_perturb
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})
    
    # Introduce directional expansion from least constrained circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute minimum distance constraint for each circle
        # Vectorized distance matrix using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Exclude self-distance and track min distances per circle
        min_distances = np.min(dists, axis=1)
        min_distances[np.arange(n)] = np.inf  # Skip self
        min_distances = np.min(min_distances, axis=1)
        
        # Determine least constrained circle based on maximal min distance
        least_constrained_idx = np.argmax(min_distances)
        least_constrained_radius = radii[least_constrained_idx]
        average_radius = np.mean(radii)
        
        # Calculate expansion based on both current sum and potential
        current_total = np.sum(radii)
        # Targeted growth: expand by a fixed fraction of average radius
        target_growth = 0.007
        max_possible_growth = (1.0 - np.min(centers, axis=1) - radii) * np.sqrt(2) * 0.8
        max_possible_growth = np.min(max_possible_growth)
        
        # Calculate expansion per circle
        expansion_per_circle = target_growth * average_radius
        expansion_factor = expansion_per_circle / average_radius
        
        # Use directional expansion to maximize total sum
        # Assign higher expansion to the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = np.clip(
            radii[least_constrained_idx] + expansion_factor * 1.2,
            1e-4, 0.5
        )
        
        # Calculate expansion distribution using geometric hashing for spread
        geometric_hash = (np.arange(n) * 197) % 1000
        expansion = expansion_per_circle * (1.0 + 0.1 * (np.sin(geometric_hash * 0.01)))
        
        # Distribute expansion across all but the least constrained
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] = np.clip(
                    radii[i] + expansion[i], 
                    1e-4, 0.5
                )
        
        # Apply expansion with constraint validation in a safe way
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate configuration without re-optimizing
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If constraints are violated, slightly decrease expansion
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Create final perturbed vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with final configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})
    
    # Final clean-up before returning
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())
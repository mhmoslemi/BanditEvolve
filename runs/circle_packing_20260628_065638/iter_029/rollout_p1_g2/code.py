import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = 6
    rows = (n + cols - 1) // cols
    
    # Create spatial coordinates with adaptive hexgrid and non-uniform distribution
    xs = []
    ys = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        # Adaptive grid with higher density along horizontal
        x_center = (col_idx + 0.5) / cols
        y_center = (row_idx + 0.5) / rows
        # Alternate row offset
        if row_idx % 2 == 1:
            x_center += 0.5 / cols
        # Introduce non-uniform spatial distribution: first half get more variance
        if i < n // 2:
            x_offset = np.random.uniform(-0.2, 0.2)
            y_offset = np.random.uniform(-0.2, 0.2)
        else:
            x_offset = np.random.uniform(-0.02, 0.02)
            y_offset = np.random.uniform(-0.02, 0.02)
        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation with refined hexagonal grid coefficient
    r0 = 0.4 / cols
        
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize total by minimizing negative

    # Vectorized constraints using closures with explicit binding
    cons = []
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary constraint: 1 - (x_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary constraint: 1 - (y_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints between all pairs
    for i in range(n):
        for j in range(i+1, n):
            # Distance^2 between centers minus sum of radii squared
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)}) 

    # First optimization with tighter tolerances and enhanced sampling
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, 
                                             "eps": 1e-10, "disp": False})
    
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute distance matrix and detect key interaction pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists = np.tril(dists)
        
        # Find the pair of most dynamically interacting circles based on 
        # (distance < mean_radius*2) and (mutual impact potential) 
        mean_radius = np.mean(radii)
        interaction_mask = (dists < mean_radius * 2) & (dists > 1e-6)
        interaction_weights = np.dot(interaction_mask, radii) * np.dot(interaction_mask.T, radii)
        interaction_weights = np.tril(interaction_weights)
        interaction_weights = np.where(interaction_weights > 0, interaction_weights, 1e-6)
        interaction_weights = interaction_weights / np.max(interaction_weights)
        
        # Compute pairwise influence as weighted impact vector
        influence_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if interaction_mask[i][j]:
                    influence_matrix[i][j] = np.linalg.norm(centers[i] - centers[j]) * interaction_weights[i][j]
        
        # Find the two circles with the highest combined influence for targeted reconfiguration
        influence_sum = np.sum(influence_matrix, axis=1)
        top_idx = np.argsort(-influence_sum)[:2]
        i1, i2 = top_idx
        
        # Build a modified adjacency hash to prioritize reconfiguration between top pair
        adjacency_hash = np.random.rand(n, 4) * 0.1
        
        # Create a specialized perturbation vector that emphasizes top pair
        perturbed_v = v.copy()
        for i in range(n):
            # General spatial perturbation with radius scaling
            perturbed_v[3*i] += adjacency_hash[i, 0] * (radii[i] / np.mean(radii)) 
            perturbed_v[3*i+1] += adjacency_hash[i, 1] * (radii[i] / np.mean(radii))
            # Special directional shift to create spatial opportunity for expansion
            if i == i1 or i == i2:
                # Apply more aggressive directional shift between top pair
                perturbed_v[3*i] += adjacency_hash[i, 2] * (radii[i] / np.mean(radii)) 
                perturbed_v[3*i+1] += adjacency_hash[i, 3] * (radii[i] / np.mean(radii))
                # Add directional expansion bias towards opposite direction
                perturbed_v[3*i] -= adjacency_hash[i, 1] * (radii[i] / np.mean(radii)) * 0.75
                perturbed_v[3*i+1] -= adjacency_hash[i, 1] * (radii[i] / np.mean(radii)) * 0.25
        
        # Second optimization phase: reconfigure top interacting pair
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12,
                                                 "eps": 1e-10, "disp": False})
    
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute distance matrix for second pass
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the least constrained circle by maximizing min distance to all
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate growth based on current total sum and potential
        current_total = np.sum(radii)
        target_growth = 0.0065  # Incremental from standard, tailored for constrained space
        expansion_factor = target_growth / (n - 1) * (current_total / np.mean(radii))
        
        # Apply expansion to the least constrained circle and nearby high-influence circles
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                # Expansions based on spatial hashing, adjacency, and influence
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                expansion = expansion_factor * (1.0 + adjacency_hash[i, 2] * 0.3) 
                if adj_weight < 0.15:
                    expansion *= 1.1
                if adj_weight < 0.1:
                    expansion *= 1.25  # Boost for very close circles
                new_radii[i] += expansion
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expansion with optimized pairwise checking
            valid = True
            for i in range(n):
                for j in range(i+1, n):
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
                # Decrease expansion gradually for all
                scaling = 0.95
                new_radii = radii + (new_radii - radii) * scaling
        
        # Update configuration with reconfigured radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization pass to refine and ensure smooth expansion
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11,
                                                 "eps": 1e-10, "disp": False})
    
    # Fallback to safe configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final validation and correction for potential numerical issues
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid, reason = validate_packing(centers, radii)
    
    # If validation fails, revert to best configuration we achieved
    if not valid:
        centers, radii, _ = run_packing()
    
    return centers, radii, float(radii.sum())
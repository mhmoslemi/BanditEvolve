import numpy as np

def run_packing():
    n = 26
    cols, rows = 5, 6  # Hexagonal grid (more efficient than square in 2D packing)
    xs = []
    ys = []
    
    # Initialize with hexagonal grid with adaptive staggering
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Base grid center
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Alternate row shift for hexagonal pattern
        if row % 2 == 1:
            x_center += 0.5 / cols
        
        # Add small but purposeful spatial perturbation
        x = x_center + np.random.uniform(-0.03, 0.03) + i * 0.002 * (1 / n)
        y = y_center + np.random.uniform(-0.03, 0.03) + i * 0.002 * (1 / n)
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation based on hexagonal grid: ~ 1/(cols*sqrt(3))
    r0 = 0.40 / cols  # Slightly higher to allow expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure 3*n values for 3n variables

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize by minimizing negative

    # Build vectorized constraints for boundary and overlap conditions
    cons = []
    
    for i in range(n):
        # Left + radius <= 1.0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0.0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1.0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0.0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with i,j parameterization
    for i in range(n):
        for j in range(i+1, n):
            # Distance squared minus sum of radii squared
            cons.append({"type": "ineq", "fun": (lambda v, i=i, j=j: 
                (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                - (v[3*i+2] + v[3*j+2])**2)})

    # First optimization
    res_first = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-11, "eps": 1e-9})
    
    v = res_first.x
    radii = v[2::3]
    centers = np.column_stack([v[0::3], v[1::3]])

    # Apply targeted geometric dissection
    # Find top 2 most interacting circles to reconfigure
    dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
    dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Create matrix of pairwise distances (inverted to find interaction)
    interaction_matrix = 1.0 / (dists + 1e-12)
    interaction_matrix[np.diag_indices_from(interaction_matrix)] = 0.0

    # Find top 2 most interacting circles
    interaction_weights = np.sum(interaction_matrix, axis=1)
    top_interacting = np.argsort(interaction_weights)[::-1][:2]
    circle_1, circle_2 = top_interacting[0], top_interacting[1]

    # Create directional spatial perturbation for circle_1 and circle_2
    spatial_hash = np.random.rand(n, 2) * 0.04
    perturbed_v = v.copy()
    for i in range(n):
        perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i]/np.mean(radii))
        perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i]/np.mean(radii))

    # Re-optimize after reconfiguration
    res_second = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                         constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-9})

    # Final optimization with targeted expansion on least constrained
    v = res_second.x
    radii = v[2::3]
    centers = np.column_stack([v[0::3], v[1::3]])

    # Calculate distances matrix
    dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
    dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)

    # Find least constrained circle
    min_dists = np.min(dists, axis=1)
    least_constrained_idx = np.argmax(min_dists)

    # New radii vector with targeted expansion strategy
    # Add 0.65% expansion to least constrained circle
    # Add 0.3% expansion to neighbors of least constrained circle
    expanded_radii = radii.copy()
    
    # Targeted expansion vector
    # Apply directional expansion using adjacency and spatial hashing
    adjacency_hash = np.random.rand(n, 2) * 0.08
    new_radii = radii.copy()
    
    # Expand least constrained circle
    expansion_factor = 0.0075 * (np.sum(radii)/np.mean(radii))  # Growth factor based on packing intensity
    new_radii[least_constrained_idx] += expansion_factor * 1.1
    
    # Expand nearby circles with directional bias
    for i in range(n):
        if i != least_constrained_idx:
            # Calculate distance to least constrained
            dist = np.linalg.norm(centers[least_constrained_idx] - centers[i])
            if dist < 0.2:
                # Apply directional expansion based on adjacency hash
                expansion = expansion_factor * 1.2 * (1.0 + 0.2 * adjacency_hash[i, 0])
                new_radii[i] += expansion
            elif dist < 0.4:
                expansion = expansion_factor * 0.85
                new_radii[i] += expansion
            else:
                # Minor expansion for distant circles
                expansion = expansion_factor * 0.1 * adjacency_hash[i, 0]
                new_radii[i] += expansion

    # Apply expansion with constraint validation
    while True:
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
        
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
            # Apply backtracking if validation fails
            new_radii = radii + (new_radii - radii) * 0.98

    # Final optimization with new radius vector
    v_final = v.copy()
    v_final[2::3] = new_radii
    res_third = minimize(neg_sum_radii, v_final, method="SLSQP", bounds=bounds,
                         constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})

    v = res_third.x if res_third.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply final validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid, reason = validate_packing(centers, radii)
    
    if not valid:
        # Fallback to best configuration
        v = res_second.x if res_second.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)

    return centers, radii, float(radii.sum())
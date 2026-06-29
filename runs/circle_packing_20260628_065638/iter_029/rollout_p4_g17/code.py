import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with adaptive geometric hashing and perturbation 
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.33) / cols + np.random.uniform(-0.04, 0.04)
        y_center = (row + 0.33) / rows + np.random.uniform(-0.04, 0.04)
        # Create staggered grid by row parity, but with adaptive offset
        if row % 2 == 1:
            x_center += 0.5 / cols * np.random.rand() * 0.5
        xs.append(x_center)
        ys.append(y_center)
    
    # Base radius with adaptive scaling
    r0 = 0.45 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-3, 0.55)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda and captured i, and direct math
    cons = []
    for i in range(n):
        # Left side boundary constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right side boundary constraint: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise circle non-overlap constraint with efficient math 
    # Use broadcasting and vectorized matrix operations
    for i in range(n):
        for j in range(i + 1, n):
            # Efficiently use lambda with captured i and j
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
            })

    # Initial optimization with tight convergence and advanced tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8, "disp": False})
    
    # If success, perform multi-phase perturbative reconfiguration and geometric dissection
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # First phase: asymmetric spatial rehashing using geometric hashing with adaptive perturbation 
        # - Use a more aggressive perturbation scale depending on radii and proximity
        # - Calculate geometric importance via inverse distance to neighbors
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Skip self and compute geometric importance of each circle 
        geometric_importance = np.zeros(n)
        for i in range(n):
            # For each circle, find the average distance to all other circles
            avg_distance_to_others = np.mean(distances[i, np.arange(n)!=i])
            geometric_importance[i] = 1.0 / (avg_distance_to_others + 1e-6) if (avg_distance_to_others > 0) else 1.

        # Compute the normalized geometric importance
        geo_norm = geometric_importance / np.sum(geometric_importance)

        # Randomized geometric hashing with intensity based on geometric importance
        hash_map = np.random.rand(n, 2) * 0.15
        perturbation = hash_map * np.outer(geo_norm, np.ones(2))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        
        # Re-evaluate with new configuration using advanced constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})

    # If success, trigger geometric dissection on most spatially constrained circles 
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Re-compute distance matrix for all circles
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Find the most spatially constrained circles (min distance to others)
        min_dist_per_circle = np.min(distances, axis=1)
        index_most_constrained = np.argmin(min_dist_per_circle)
        index_second_most_constrained = np.argsort(min_dist_per_circle)[1]
        
        # Apply geometric dissection:
        # - Move the second most constrained circle to a more distant position
        # - Reduce radius of the most constrained circle to avoid overlap
        # - Apply a fixed shift to reduce spatial interactions
        shift_x = 0.15 * (1.0 - radii[index_second_most_constrained]) if radii[index_second_most_constrained] < 0.3 else 0.08
        shift_y = 0.15 * (1.0 - radii[index_second_most_constrained]) if radii[index_second_most_constrained] < 0.3 else 0.08
        
        # Ensure that the new position remains within the square and does not cause overlaps
        new_center_x = centers[index_second_most_constrained, 0] + shift_x
        new_center_y = centers[index_second_most_constrained, 1] + shift_y
        
        # Check and update x position with bounds
        if new_center_x - radii[index_second_most_constrained] < 0.0:
            while new_center_x - radii[index_second_most_constrained] < 0.0:
                new_center_x += 1e-4
        elif new_center_x + radii[index_second_most_constrained] > 1.0:
            while new_center_x + radii[index_second_most_constrained] > 1.0:
                new_center_x -= 1e-4
        
        # Check and update y position with bounds
        if new_center_y - radii[index_second_most_constrained] < 0.0:
            while new_center_y - radii[index_second_most_constrained] < 0.0:
                new_center_y += 1e-4
        elif new_center_y + radii[index_second_most_constrained] > 1.0:
            while new_center_y + radii[index_second_most_constrained] > 1.0:
                new_center_y -= 1e-4
        
        # Create new vector with adjusted centers
        v_new = v.copy()
        v_new[3*index_second_most_constrained] = new_center_x
        v_new[3*index_second_most_constrained + 1] = new_center_y
        v_new[3*index_most_constrained + 2] = max(radii[index_most_constrained] - 0.005, 1e-3)  # reduce radius if too close
        
        # Re-evaluate with adjusted parameters after geometric dissection
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})

    # If success, perform radius expansion with strict feasibility enforcement
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate current total sum and find the least constrained circle
        total = radii.sum()
        # Use the same geometric importance calculation as before
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Re-calculate geometric importance for least constrained expansion
        min_dist_per_circle = np.min(distances, axis=1)
        geometric_importance = 1.0 / (min_dist_per_circle + 1e-6)  # Inverse of min distance
        geo_norm = geometric_importance / np.sum(geometric_importance)  # Normalize
        
        # Find the circle with the least constraint (max geometric importance)
        expansion_circle = np.argmax(geometric_importance)
        
        # Compute safe expansion based on current configuration and spatial constraints
        # Define a safety margin based on the smallest min distance
        safety_factor = 0.9
        min_safe_distance = min(min_dist_per_circle)
        max_allowed_radius_increase = safety_factor * (min_safe_distance - radii[expansion_circle]) 
        
        # Compute expansion factor for this circle, using proportion to global importance
        expansion_factor_base = 0.01
        
        # Create an expansion vector with targeted expansion
        new_radii = radii.copy()
        # Expansion on the least constrained circle
        new_radii[expansion_circle] += expansion_factor_base * geo_norm[expansion_circle] + (0.004 if np.random.rand() < 0.2 else 0.0)
        
        # Distribute expansion to others in proportion to their geometric importance
        expansion_other = expansion_factor_base * np.sum(geo_norm[geo_norm != geo_norm[expansion_circle]] * (1.0+np.random.rand()))
        new_radii[geo_norm != geo_norm[expansion_circle]] += expansion_other * (geo_norm[geo_norm != geo_norm[expansion_circle]] / geo_norm[expansion_circle])
        # Set the expansion to not exceed safe limits per circle
        max_per_circle_expand = np.minimum(
            new_radii - radii, 
            np.full(n, max_allowed_radius_increase)
        )
        new_radii = np.where(new_radii - radii > max_per_circle_expand, new_radii, new_radii + max_per_circle_expand)

        # Apply the new radii to the vector, ensuring not to exceed max bounds
        v_new = v.copy()
        v_new[2::3] = np.clip(new_radii, 1e-3, 0.55)
        
        # Re-evaluate with new radii to enforce total sum and non-overlap
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
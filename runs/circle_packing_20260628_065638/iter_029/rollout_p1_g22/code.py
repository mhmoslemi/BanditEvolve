import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering
    # and asymmetric hexagonal staggered grid with directional spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Apply asymmetric spatial hashing for spatial displacement
        spatial_hash_x = np.random.uniform(-0.06, 0.06) * (1 + 0.4 * (np.random.rand() ** 0.5))
        spatial_hash_y = np.random.uniform(-0.06, 0.06) * (1 + 0.4 * (np.random.rand() ** 0.5))
        
        # Apply staggered grid with row-dependent shift and asymmetric displacement
        row_shift = 0.5 / cols * (1 + np.random.rand() * 0.2) if row % 2 == 1 else 0
        x = x_center + spatial_hash_x + row_shift
        y = y_center + spatial_hash_y
        
        # Prevent overlap with edges via clamping
        x = np.clip(x, 1e-8, 1 - 1e-8)
        y = np.clip(y, 1e-8, 1 - 1e-8)
        xs.append(x)
        ys.append(y)
    
    # Calculate starting radii based on grid efficiency with additional expansion allowance
    r0_base = 1.0 / (2 * np.sqrt(3)) / cols * 0.8  # Hexagonal base radius
    r0 = r0_base + 0.001  # Add slight buffer for expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Bounds for the optimization decision vector
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize sum by minimizing negative

    # Constraint list for boundaries and overlaps
    cons = []

    for i in range(n):
        # Left bound
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right bound
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom bound
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top bound
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})

    # Overlap constraints with optimized pairwise calculation
    for i in range(n):
        for j in range(i+1, n):
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with tight tolerances and aggressive iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-10})

    # Main reconfiguration phase with directional bias and spatial prioritization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash and adjacency hash based on current state
        spatial_hash = np.random.rand(n, 2) * 0.05 * (1.0 + 0.3 * (radii / np.mean(radii)))
        adjacency_hash = np.random.rand(n, 2) * 0.04 * (1.0 + 0.3 * (radii / np.mean(radii)))
        
        # Reconfigure with directional perturbation
        perturbed_v = v.copy()
        for i in range(n):
            # Add directional perturbations based on spatial and adjacency hash
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
            
            # Add adjacency-aware expansion bias (higher expansion for more connected circles)
            if i < n - 1:
                direction_weight = adjacency_hash[i, 0] * (1.0 + 0.4 * (radii[i] / np.mean(radii)))
                direction_weight = np.clip(direction_weight, 0.0, 0.005)
                # Apply directional perturbation in the direction of the center
                direction = (centers[i] - centers[i-1]) if i != 0 else np.zeros(2)
                direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 1e-8 else direction
                perturbed_v[3*i] += direction[0] * direction_weight
                perturbed_v[3*i+1] += direction[1] * direction_weight
        
        # First re-evaluation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-10})

    # Select two most dynamically interacting circles for focused reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix for fast pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find top 2 most interacting circles (least distance between them)
        interaction_strength = (1.0 / (dists + 1e-12))  # Inverse distance as a proxy
        top_2_idx = np.argsort(interaction_strength, axis=None)[-2:]
        circle0, circle1 = top_2_idx[0], top_2_idx[1]
        
        # Extract their centers and radii
        c0 = centers[circle0]
        c1 = centers[circle1]
        r0 = radii[circle0]
        r1 = radii[circle1]

        # Reconfigure spatial relationship between the two most interacting circles
        # Apply controlled displacement using directional vector and spatial hash
        direction_vector = (c1 - c0)
        direction_vector /= np.linalg.norm(direction_vector) if np.linalg.norm(direction_vector) > 1e-8 else 1.0
        displacement = direction_vector * (0.01 + 0.01 * np.random.rand())
        # Use spatial hash to determine the displacement direction
        spatial_hash = np.random.rand(2) * 0.03 * (1.0 + 0.3 * (radii / np.mean(radii)))
        displacement += (spatial_hash[0] - spatial_hash[1]) * (r0 + r1)
        
        # Apply spatial reconfiguration of the two interacting circles
        centers[circle0] += displacement
        centers[circle1] -= displacement
        
        # Apply radius adjustment with soft non-overlap constraints and directional bias
        target_growth = 0.004 * (np.sum(radii) / np.std(radii))  # Grow based on total and variance
        new_radii = radii.copy()
        new_radii[circle0] += target_growth * 1.1  # Boost growth on the more constrained circle
        new_radii[circle1] += target_growth * 0.9  # Slight decrease to prevent over-expansion
        
        # Enforce non-overlap through directional distance enforcement
        # Vectorized calculation of pairwise distance and enforcement
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Enforce non-overlap between circles with minimum safety margin
        min_safety_distance = 1e-6
        for i in range(n):
            for j in range(i+1, n):
                # Calculate minimum required distance and apply safety adjustment
                required_distance = new_radii[i] + new_radii[j] + min_safety_distance
                if dists[i,j] < required_distance - 1e-6:
                    # Adjust radii toward safety
                    new_radii[i] -= (required_distance - dists[i,j]) * 0.01 * (1.0 + 0.3)
                    new_radii[j] -= (required_distance - dists[i,j]) * 0.01 * (1.0 + 0.3)
        
        # Apply the new radii to the decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})

    # Final refinement with focused expansion on the least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distance matrix with vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the least constrained circle with a novel metric:
        # weighted sum of minimal distance and inverse distance to all other circles
        weights = np.where(dists > 0, 1.0 / (dists + 1e-6), 0.0)
        min_dists = np.min(dists, axis=1)
        min_dists_normalized = min_dists / (np.max(min_dists) + 1e-12)
        
        # Combine with a weight based on circle radius to prioritize larger circles
        min_dist_weight = min_dists_normalized * (1.0 + 0.8 * (radii / np.mean(radii)))
        least_constrained_idx = np.argmax(min_dist_weight)
        
        # Calculate potential expansion using the current total sum and radius distribution
        current_total = np.sum(radii)
        target_growth = 0.0065 * (current_total / np.sum(radii))
        
        # Expand the least constrained circle by a factor
        expansion_factor = target_growth * (1.0 + 0.3 * (radii[least_constrained_idx] / np.mean(radii)))
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor
        
        # Apply a soft expansion to neighboring circles with adjacency bias
        for i in range(n):
            # Use adjacency hash to give directional preference
            adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
            if adj_weight < 0.1:
                # Boost expansion for circles close to the least constrained one
                expansion = expansion_factor * 1.3
                new_radii[i] += expansion
            else:
                # Use directional expansion based on adjacency hash
                expansion = expansion_factor * (1.0 + 0.2 * (adjacency_hash[i, 0]))
                new_radii[i] += expansion
        
        # Apply expansion and validate with constraint check
        while True:
            # Create a test configuration with new radii
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check for overlaps
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    # Apply a tighter margin than base to ensure strict compliance
                    if dist < (new_radii[i] + new_radii[j]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, reduce expansion gradually
                new_radii = radii + (new_radii - radii) * 0.95

        # Update and re-optimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})

    # Final fallback or final decision
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final validation and return
    return centers, radii, float(radii.sum())
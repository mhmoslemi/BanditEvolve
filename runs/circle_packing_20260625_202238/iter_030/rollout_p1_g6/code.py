import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    def create_overlap_constraints():
        overlap_cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                overlap_cons.append({"type": "ineq", "fun": constraint_func})
        return overlap_cons

    cons += create_overlap_constraints()

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Compute constraint tightness
    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                constraint_tightness[i] += (v[3*i+2] + v[3*j+2] - dist)
                constraint_tightness[j] += (v[3*i+2] + v[3*j+2] - dist)
    
    # Sort indices by constraint tightness (most constrained first)
    sorted_indices = np.argsort(constraint_tightness)
    
    # Permute the decision vector based on sorted indices
    permuted_v = np.zeros_like(v)
    for i, idx in enumerate(sorted_indices):
        permuted_v[3*i] = v[3*idx]
        permuted_v[3*i+1] = v[3*idx+1]
        permuted_v[3*i+2] = v[3*idx+2]
    
    # Re-optimize with permuted initial guess
    res = minimize(neg_sum_radii, permuted_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else permuted_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply dual-phase geometric distortion
    # Phase 1: Logarithmic scaling of coordinates
    log_v = np.log(v + 1e-10)
    log_v[0::3] = (log_v[0::3] - np.min(log_v[0::3])) / (np.max(log_v[0::3]) - np.min(log_v[0::3]))
    log_v[1::3] = (log_v[1::3] - np.min(log_v[1::3])) / (np.max(log_v[1::3]) - np.min(log_v[1::3]))
    log_v[2::3] = (log_v[2::3] - np.min(log_v[2::3])) / (np.max(log_v[2::3]) - np.min(log_v[2::3]))
    
    # Phase 2: Re-seed with distorted coordinates
    distorted_v = np.copy(log_v)
    distorted_v[0::3] *= 1.2
    distorted_v[1::3] *= 1.2
    distorted_v[2::3] *= 0.8

    # Re-optimize with distorted initial guess
    res = minimize(neg_sum_radii, distorted_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else distorted_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Add penalty for overlapping circles to improve convergence
    def penalty(v):
        sum_penalty = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                    sum_penalty += max(0, (v[3*i+2] + v[3*j+2] - dist) ** 2)
        return sum_penalty

    # Re-optimize with penalty
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Jiggle heuristic for smallest circles
    if np.sum(radii) > 0:
        # Sort circles by radius (smallest first)
        sorted_indices = np.argsort(radii)
        # Select the smallest 10 circles
        small_circle_indices = sorted_indices[:10]
        # Perturb their positions slightly and re-optimize
        perturbation = 0.01
        for idx in small_circle_indices:
            i = idx
            v[3*i] += np.random.uniform(-perturbation, perturbation)
            v[3*i+1] += np.random.uniform(-perturbation, perturbation)
            v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
        # Re-optimize with penalty
        res = minimize(lambda v: -np.sum(v[2::3]) + 100 * penalty(v), v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 300, "ftol": 1e-9})
        v = res.x if res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)

    # Final cleanup pass: attempt to increase radii slightly without moving centers
    # This step is only performed if the current solution is valid and does not cause overlap
    if validate_packing(centers, radii)[0]:
        for i in range(n):
            # Try to increase radius by a small epsilon
            new_radius = radii[i] + 1e-6
            # Check if increasing this radius would cause overlap with any other circle
            overlap = False
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_radius + radii[j] - 1e-12:
                    overlap = True
                    break
            if not overlap:
                # Increase the radius and update the decision vector
                v[3*i+2] = new_radius
                # Update the radii and centers
                radii = np.clip(v[2::3], 1e-6, None)
                centers = np.column_stack([v[0::3], v[1::3]])

    # Apply geometric phase shift mutation
    phase_shift_v = np.copy(v)

    # Apply controlled non-linear spatial transformation to the entire layout
    # Phase 1: Logarithmic scaling of coordinates
    log_v = np.log(v + 1e-10)
    log_v[0::3] = (log_v[0::3] - np.min(log_v[0::3])) / (np.max(log_v[0::3]) - np.min(log_v[0::3]))
    log_v[1::3] = (log_v[1::3] - np.min(log_v[1::3])) / (np.max(log_v[1::3]) - np.min(log_v[1::3]))
    log_v[2::3] = (log_v[2::3] - np.min(log_v[2::3])) / (np.max(log_v[2::3]) - np.min(log_v[2::3]))
    
    # Phase 2: Re-seed with transformed coordinates
    transformed_v = np.copy(log_v)
    transformed_v[0::3] *= 1.1
    transformed_v[1::3] *= 1.1
    transformed_v[2::3] *= 0.9

    # Re-optimize with transformed initial guess
    res = minimize(neg_sum_radii, transformed_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    transformed_v = res.x if res.success else transformed_v
    transformed_centers = np.column_stack([transformed_v[0::3], transformed_v[1::3]])
    transformed_radii = np.clip(transformed_v[2::3], 1e-6, None)

    # Validate transformed solution and compare with current solution
    if validate_packing(transformed_centers, transformed_radii)[0] and np.sum(transformed_radii) > np.sum(radii):
        v = transformed_v
        centers = transformed_centers
        radii = transformed_radii

    # Final refinement: fine-tune tolerances and re-optimize
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-12})
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Final cleanup pass: attempt to increase radii slightly without moving centers
    # This step is only performed if the current solution is valid and does not cause overlap
    if validate_packing(centers, radii)[0]:
        for i in range(n):
            # Try to increase radius by a small epsilon
            new_radius = radii[i] + 1e-6
            # Check if increasing this radius would cause overlap with any other circle
            overlap = False
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_radius + radii[j] - 1e-12:
                    overlap = True
                    break
            if not overlap:
                # Increase the radius and update the decision vector
                v[3*i+2] = new_radius
                # Update the radii and centers
                radii = np.clip(v[2::3], 1e-6, None)
                centers = np.column_stack([v[0::3], v[1::3]])

    # Apply targeted geometric inversion on the three smallest circles
    # Step 1: Identify the three smallest circles
    sorted_by_radius = np.argsort(radii)
    smallest_circle_indices = sorted_by_radius[:3]
    
    # Step 2: Swap centers of the smallest circles
    # Create a copy of the current centers to manipulate
    centers_copy = np.copy(centers)
    
    # Swap center positions of the three smallest circles
    for i in range(len(smallest_circle_indices) // 2):
        idx1 = smallest_circle_indices[i]
        idx2 = smallest_circle_indices[len(smallest_circle_indices) - 1 - i]
        centers_copy[[idx1, idx2]] = centers_copy[[idx2, idx1]]
    
    # Step 3: Invert relative angles of the three smallest circles around their centroid
    # Compute the centroid of the three smallest circles
    centroid = np.mean(centers_copy[smallest_circle_indices], axis=0)
    
    # Rotate the positions of the smallest circles by 180 degrees around the centroid
    for idx in smallest_circle_indices:
        center = centers_copy[idx]
        dx = center[0] - centroid[0]
        dy = center[1] - centroid[1]
        centers_copy[idx, 0] = 2 * centroid[0] - center[0]
        centers_copy[idx, 1] = 2 * centroid[1] - center[1]
    
    # Step 4: Create modified decision vector with the new centers
    modified_v = np.zeros_like(v)
    for i in range(n):
        modified_v[3*i] = centers_copy[i, 0]
        modified_v[3*i+1] = centers_copy[i, 1]
        modified_v[3*i+2] = v[3*i+2]
    
    # Step 5: Re-optimize with the modified decision vector and a modified objective function
    # The modified objective function gives higher weight to the smallest circles
    def modified_neg_sum_radii(v):
        # Calculate the weighted sum of radii, giving higher weight to the smallest circles
        weights = np.zeros(n)
        for i in range(n):
            if i in smallest_circle_indices:
                weights[i] = 1.5
            else:
                weights[i] = 1.0
        return -np.sum(weights * v[2::3])
    
    # Re-optimize with the modified objective function
    res = minimize(modified_neg_sum_radii, modified_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else modified_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Final cleanup pass: attempt to increase radii slightly without moving centers
    # This step is only performed if the current solution is valid and does not cause overlap
    if validate_packing(centers, radii)[0]:
        for i in range(n):
            # Try to increase radius by a small epsilon
            new_radius = radii[i] + 1e-6
            # Check if increasing this radius would cause overlap with any other circle
            overlap = False
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_radius + radii[j] - 1e-12:
                    overlap = True
                    break
            if not overlap:
                # Increase the radius and update the decision vector
                v[3*i+2] = new_radius
                # Update the radii and centers
                radii = np.clip(v[2::3], 1e-6, None)
                centers = np.column_stack([v[0::3], v[1::3]])

    return centers, radii, float(radii.sum())
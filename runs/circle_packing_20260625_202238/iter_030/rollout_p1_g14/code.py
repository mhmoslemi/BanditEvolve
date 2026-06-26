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

    # Apply targeted geometric inversion
    # Isolate three smallest circles and invert their spatial relationships
    sorted_by_radius = np.argsort(radii)
    smallest_circle_indices = sorted_by_radius[:3]
    # Store their positions and radii
    stored_positions = v[3*smallest_circle_indices[0]:3*smallest_circle_indices[0]+3]
    stored_positions = np.vstack((v[3*smallest_circle_indices[0]:3*smallest_circle_indices[0]+3], 
                                 v[3*smallest_circle_indices[1]:3*smallest_circle_indices[1]+3], 
                                 v[3*smallest_circle_indices[2]:3*smallest_circle_indices[2]+3]))
    stored_radii = v[3*smallest_circle_indices[0]+2:3*smallest_circle_indices[0]+3]
    stored_radii = np.vstack((v[3*smallest_circle_indices[0]+2], 
                             v[3*smallest_circle_indices[1]+2], 
                             v[3*smallest_circle_indices[2]+2]))
    # Invert their positions
    inverted_positions = np.copy(stored_positions)
    inverted_positions[0] = 1.0 - stored_positions[0][0]
    inverted_positions[1] = 1.0 - stored_positions[1][0]
    inverted_positions[2] = 1.0 - stored_positions[2][0]
    inverted_positions[0] = 1.0 - stored_positions[0][1]
    inverted_positions[1] = 1.0 - stored_positions[1][1]
    inverted_positions[2] = 1.0 - stored_positions[2][1]
    # Update the decision vector
    for i, idx in enumerate(smallest_circle_indices):
        v[3*idx] = inverted_positions[i][0]
        v[3*idx+1] = inverted_positions[i][1]
        v[3*idx+2] = stored_radii[i]
    # Re-optimize with modified objective function
    def modified_neg_sum_radii(v):
        # Calculate the weighted sum of radii, giving higher weight to the smallest circles
        weights = np.zeros(n)
        for i in range(n):
            if i in smallest_circle_indices:
                weights[i] = 1.5
            else:
                weights[i] = 1.0
        return -np.sum(weights * v[2::3])
    
    res = minimize(modified_neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
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

    return centers, radii, float(radii.sum())
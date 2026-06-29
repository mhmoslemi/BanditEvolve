import numpy as np

def run_packing():
    n = 26
    # Strategic choice of grid: asymmetric 5x6 for better edge exploitation
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Seed for consistent reproducibility of spatial perturbations
    np.random.seed(42)
    
    # Initialize positions with enhanced geometric clustering and adaptive stagger
    xs = []
    ys = []
    base_radii = np.ones(n) * 0.35  # Base radius that gets refined
    
    for i in range(n):
        row = i // cols
        col = i % cols
        # Centralized initial placement with refined perturbation and grid staggering
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Strategic vertical staggering to avoid vertical overcrowding
        if row % 2 == 1:
            x += 0.3 / cols  # Less aggressive staggering to maintain horizontal cohesion
        # Horizontal spacing adjustment to ensure more uniform coverage
        if col == 0:
            x = x_center + np.random.uniform(0, 0.1)
        xs.append(x)
        ys.append(y)
    
    # Initial radii that are not too small to be constrained
    r0 = np.full(n, 0.35 / cols - 1e-3)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, consistent with v

    # Vectorized objective function with gradient information for better convergence
    def neg_sum_radii(v):
        radii = v[2::3]
        return -np.sum(radii)  # Objective is to maximize radius sum

    # Vectorized constraints: spatial containment
    cons = []
    for i in range(n):
        # Left bound: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right bound: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom bound: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top bound: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized constraints: circle-circle separation (non-overlap)
    for i in range(n):
        for j in range(i + 1, n):
            # Lambda capture with explicit i and j to avoid closure pitfalls
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization phase with high precision and dynamic tolerances
    initial_optimization_options = {"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-12, "eps": 1e-8, "jac": True, "disp": False}
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options=initial_optimization_options)

    # Phase 1: Dynamic geometric dissection of top two interacting circles
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Vectorized pairwise distance matrix to find top two interacting circles
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # For each pair, compute the distance to sum of radii ratio
        ratio = dists / (radii[:, np.newaxis] + radii[np.newaxis, :])

        # Find the two circles with largest (most constrained, least space) ratio
        # i.e., least space between them relative to their sizes
        # This avoids using argmin over sum (which could be zero) and instead
        # looks for circles with the highest constraint ratio
        best_indices = np.unravel_index(np.argsort(ratio, axis=None)[-3:], ratio.shape)
        best_pairs = set()
        for i, j in best_indices:
            if i < j:
                best_pairs.add((i, j))
        # Extract the two circles with the highest interaction score
        # We take the first and second largest interaction pairs or use all if not enough
        if len(best_pairs) < 2:
            best_pairs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in best_pairs:
                        continue
                    if len(best_pairs) >= 2:
                        break
                    best_pairs.add((i, j))
        # Convert to list
        best_circle_list = [i for (i, _) in best_pairs] if len(best_pairs) > 0 else list(range(n))

        # Now force reconfiguration of these two circles to enable tighter packing
        # We'll move them away from each other to increase their spatial capacity
        # but with dynamic adjustment to avoid overfitting and ensure others don't become over-constrained
        target_displacement = 0.03  # Small perturbation for fine control

        # Create a clone of the current state with the two circles moved
        displacement_v = v.copy()
        # For each of the constrained circles, we will move them in different directions
        for i in best_circle_list:
            # Find their current position and radius
            center_x, center_y, current_radius = v[3*i], v[3*i+1], v[3*i+2]
            # Move in perpendicular direction relative to the other circle
            if len(best_circle_list) == 2:
                other_i = best_circle_list[1] if i == best_circle_list[0] else best_circle_list[0]
                other_center_x, other_center_y = v[3*other_i], v[3*other_i+1]
                dx = other_center_x - center_x
                dy = other_center_y - center_y
                dist = np.hypot(dx, dy)
                if dist == 0:
                    move_x, move_y = np.random.uniform(-target_displacement, target_displacement, size=2)
                else:
                    norm = np.array([dx, dy]) / dist
                    move_dir = np.array([0, -1])  # Move in direction of vertical space
                    move_dir -= norm @ move_dir * norm
                    move_dir /= np.hypot(*move_dir)
                    move_x = move_dir[0] * target_displacement
                    move_y = move_dir[1] * target_displacement
            else:
                # No other circle, move based on overall layout direction
                move_x = np.random.uniform(-target_displacement, target_displacement)
                move_y = np.random.uniform(-target_displacement, target_displacement)
            displacement_v[3*i] += move_x
            displacement_v[3*i+1] += move_y

        # Apply spatial bounds to the displaced circles to avoid boundary violations
        for i in best_circle_list:
            displacement_v[3*i] = np.clip(displacement_v[3*i], 1e-12, 1.0 - 1e-12)
            displacement_v[3*i+1] = np.clip(displacement_v[3*i+1], 1e-12, 1.0 - 1e-12)
            displacement_v[3*i+2] = np.clip(displacement_v[3*i+2], 1e-6, 0.5)
        
        # Re-evaluate using the displaced configuration
        res = minimize(neg_sum_radii, displacement_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12, "eps": 1e-10})

    # Phase 2: Targeted radius expansion of least constrained circle with new adjacency constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # For each circle, compute min distance to any other
        min_dists = np.min(dists, axis=1)

        # Find the circle with the largest available "buffer" (i.e., the least constrained)
        least_constrained_idx = np.argmax(min_dists)

        # Create a new circle adjacency constraint that forces a reordering
        # for the two most dynamically interacting circles (from previous phase)
        # to create a new layout structure
        target_new_adjacent_pair = [best_circle_list[0], best_circle_list[1]] if len(best_circle_list) >= 2 else [0,1]
        i, j = target_new_adjacent_pair[0], target_new_adjacent_pair[1]
        # Create a new constraint to ensure that these two are adjacent by distance
        # (with dynamic offset to maintain non-overlap and enable reconfiguration)
        min_distance_for_reconfiguration = np.min(dists[i,j]) - 0.01  # 1% buffer
        max_distance_for_reconfiguration = np.max(dists[i,j]) + 0.05  # 5% expansion

        # Create a new constraint that allows the pair to remain within the range, enforcing reconfiguration
        new_reconfiguration_constraint = {"type": "ineq",
                                         "fun": lambda v, i=i, j=j: max_distance_for_reconfiguration - (v[3*i] - v[3*j]) **2 - (v[3*i+1] - v[3*j+1]) **2 - (v[3*i+2] + v[3*j+2])**2}

        # Ensure the constraint is added with proper capture
        cons.append(new_reconfiguration_constraint)

        # Re-evaluate using the new constraint
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-10, "gtol": 1e-12})
    
    # Phase 3: Refinement based on improved interaction metrics and expanded radius control
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute the sum of the radii and track the improvement
        current_sum = np.sum(radii)
        
        # Calculate the maximum possible growth based on spatial and boundary constraints
        # Using more refined metrics (avoiding brute-force)
        base_expansion = 0.002  # Base expansion per circle
        expansion_factor = 1.2  # Slight over-expansion allowed
        expansion_amount = base_expansion * expansion_factor
        # Target total expansion based on geometric space efficiency
        target_total_sum = current_sum + expansion_amount * 0.7  # 70% of total expansion

        # Find the least constrained circle (from earlier calculation)
        if len(best_circle_list) >= 1:
            least_constrained_idx = best_circle_list[0]
        else:
            # Fallback to previously computed minimum buffer
            min_dists = np.min(
                np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 + 
                        (centers[:, np.newaxis, 1] - centers[np.newaxis, 1])**2), 
                axis=1
            )
            least_constrained_idx = np.argmax(min_dists)

        # We focus on expanding the least constrained
        # Apply a new radius expansion that is spatially adaptive, based on current radii and distances
        # Calculate average radius and expand the least constrained circle proportionally
        avg_radius = np.mean(radii)
        expansion_coeff = (target_total_sum - current_sum) / (avg_radius * 1.1)  # Safe expansion limit

        # Create a new radial expansion vector with controlled increase
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_coeff * 0.8  # 80% of expansion coefficient

        # Create a new decision vector with updated radii
        updated_v = v.copy()
        updated_v[2::3] = new_radii
        
        # Check if the new radii configuration is feasible
        is_feasible = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = updated_v[3*i] - updated_v[3*j]
                dy = updated_v[3*i+1] - updated_v[3*j+1]
                distance = np.sqrt(dx**2 + dy**2)
                if distance < (new_radii[i] + new_radii[j]) - 1e-12:
                    is_feasible = False
                    break
            if not is_feasible:
                break
        
        if is_feasible:
            # Re-evaluate with the expanded configuration
            res = minimize(neg_sum_radii, updated_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})
        else:
            # If not feasible, adjust the expansion coefficient
            # Reduce expansion by 40%
            new_radii[least_constrained_idx] -= expansion_coeff * 0.4
            updated_v = v.copy()
            updated_v[2::3] = new_radii
            res = minimize(neg_sum_radii, updated_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 200, "ftol": 1e-11, "eps": 1e-10})
    
    # Final check in case of failure or need for final refinement
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Ensure valid radius values

    # Final check for boundary compliance (edge case prevention)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if not (0 <= x - r and x + r <= 1 and 0 <= y - r and y + r <= 1):
            # Adjust slightly to be within bounds
            if x - r < 0:
                v[3*i] = r
            elif x + r > 1:
                v[3*i] = 1 - r
            if y - r < 0:
                v[3*i+1] = r
            elif y + r > 1:
                v[3*i+1] = 1 - r
            v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
    
    # Final optimization pass with extremely tight constraints
    if res.success:
        v = res.x
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-11, "gtol": 1e-14})
    
    # Final boundary check as a safeguard
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            centers[i] = [max(0.0, min(x, 1.0)), max(0.0, min(y, 1.0))]
            radii[i] = np.clip(r, 1e-6, 0.5)
    
    return centers, radii, float(radii.sum())
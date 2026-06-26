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

    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

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
    
    # Final cleanup pass: try to infinitesimally increase radii without moving centers
    def cleanup_obj(v):
        # Keep centers fixed
        centers_fixed = np.array([v[0::3], v[1::3]])
        # Check if any circle can be slightly inflated
        max_increase = 0
        for i in range(n):
            # Calculate potential increase in radius
            can_increase = True
            for j in range(n):
                if i == j:
                    continue
                dx = centers_fixed[0, i] - centers_fixed[0, j]
                dy = centers_fixed[1, i] - centers_fixed[1, j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                    can_increase = False
                    break
            if can_increase:
                max_increase = max(max_increase, 1e-5)
        # If we can increase radii, do so
        if max_increase > 0:
            # We try to increase all radii slightly, keeping centers fixed
            new_r = v[2::3] + max_increase
            new_r = np.clip(new_r, 1e-6, 0.5)
            # Check if new radii would violate constraints
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers_fixed[0, i] - centers_fixed[0, j]
                    dy = centers_fixed[1, i] - centers_fixed[1, j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < new_r[i] + new_r[j] - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return -np.sum(new_r)
        # If not valid, return original objective
        return -np.sum(v[2::3])
    
    # Perform final cleanup
    res_cleanup = minimize(cleanup_obj, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
    v_cleanup = res_cleanup.x if res_cleanup.success else v
    centers_cleanup = np.column_stack([v_cleanup[0::3], v_cleanup[1::3]])
    radii_cleanup = np.clip(v_cleanup[2::3], 1e-6, None)
    
    # Return the improved result if valid, otherwise fall back to original
    valid, msg = validate_packing(centers_cleanup, radii_cleanup)
    if valid:
        return centers_cleanup, radii_cleanup, float(radii_cleanup.sum())
    else:
        return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Stagger alternate rows to reduce overlap
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Randomized geometric hashing for spatial constraint evaluation
    if res.success:
        v = res.x
        # Compute geometric hash for each circle (based on position and radius)
        hash_positions = np.column_stack([v[0::3], v[1::3], v[2::3]])
        # Hash function: round to nearest 0.01 to group nearby circles
        hash_positions[:, 0] = np.round(hash_positions[:, 0], 2)
        hash_positions[:, 1] = np.round(hash_positions[:, 1], 2)
        hash_positions[:, 2] = np.round(hash_positions[:, 2], 2)
        # Group circles by hash and enforce minimal spacing within each group
        unique_hashes = np.unique(hash_positions, axis=0)
        for hash_val in unique_hashes:
            idx = np.where((hash_positions[:, 0] == hash_val[0]) &
                           (hash_positions[:, 1] == hash_val[1]) &
                           (hash_positions[:, 2] == hash_val[2]))[0]
            if len(idx) > 1:
                # Enforce spacing between circles in the same hash group
                for i in idx:
                    for j in idx:
                        if i != j:
                            dx = v[3*i] - v[3*j]
                            dy = v[3*i+1] - v[3*j+1]
                            cons.append({"type": "ineq", 
                                         "fun": lambda v, dx=dx, dy=dy, 
                                         r_i=v[3*i+2], r_j=v[3*j+2]: 
                                         dx*dx + dy*dy - (r_i + r_j)**2})
        # Re-optimization with new constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Controlled radius expansion for smallest non-zero radius
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the smallest non-zero radius
        min_radius_idx = np.argmin(radii)
        if radii[min_radius_idx] > 1e-6:
            # Expand its radius slightly and adjust its position to maintain feasibility
            v[3*min_radius_idx + 2] += 0.002
            # Adjust position to minimize boundary violation
            x = v[3*min_radius_idx]
            y = v[3*min_radius_idx+1]
            r = v[3*min_radius_idx+2]
            # Move in direction of closest boundary
            dists = np.array([x - r, 1 - x - r, y - r, 1 - y - r])
            closest_boundary = np.argmin(dists)
            if closest_boundary == 0:
                v[3*min_radius_idx] -= 0.005
            elif closest_boundary == 1:
                v[3*min_radius_idx] += 0.005
            elif closest_boundary == 2:
                v[3*min_radius_idx+1] -= 0.005
            elif closest_boundary == 3:
                v[3*min_radius_idx+1] += 0.005
            # Re-evaluate with adjusted parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
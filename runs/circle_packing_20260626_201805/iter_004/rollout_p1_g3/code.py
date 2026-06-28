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

    def vectorized_overlap_constraint(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dx = x_centers[:, np.newaxis] - x_centers[np.newaxis, :]
        dy = y_centers[:, np.newaxis] - y_centers[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r_radii[:, np.newaxis] + r_radii[np.newaxis, :])**2
        return dist_sq - min_dist_sq

    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_cons.append({"type": "ineq", "fun": constraint_func})

    cons.extend(overlap_cons)

    def geometric_distortion(v):
        # Apply random rotation and scaling to the initial guess
        theta = np.random.uniform(-np.pi/4, np.pi/4)
        scale = np.random.uniform(0.8, 1.2)
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        
        # Rotate and scale the positions
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotated_x = x_centers * cos_theta - y_centers * sin_theta
        rotated_y = x_centers * sin_theta + y_centers * cos_theta
        distorted_x = rotated_x * scale
        distorted_y = rotated_y * scale
        
        # Apply clipping to ensure bounds are respected
        distorted_v = np.zeros_like(v)
        distorted_v[0::3] = np.clip(distorted_x, 0.0, 1.0)
        distorted_v[1::3] = np.clip(distorted_y, 0.0, 1.0)
        distorted_v[2::3] = r_radii
        return distorted_v

    # Apply a radical topological reconfiguration by splitting the current layout
    # into independent subcomponents and applying a global permutation
    def topological_reconfiguration(v):
        # Split into two subcomponents
        half_n = n // 2
        sub1 = v[:3*half_n]
        sub2 = v[3*half_n:]
        
        # Apply random permutation to the second subcomponent
        perm = np.random.permutation(half_n)
        sub2_permuted = sub2[3*perm]
        sub2_permuted[1::3] = sub2_permuted[1::3][np.random.permutation(half_n)]
        
        # Enforce constraint that at least one subcomponent expands by 10%
        # by increasing radii of the first subcomponent by 10% if possible
        r1 = sub1[2::3]
        r1_expanded = r1 * 1.1
        sub1_expanded = sub1.copy()
        sub1_expanded[2::3] = r1_expanded
        # Check if expansion is feasible (no overlap)
        x1 = sub1_expanded[0::3]
        y1 = sub1_expanded[1::3]
        r1_expanded = sub1_expanded[2::3]
        # Check with original sub2 positions
        dx = x1[:, np.newaxis] - sub2[0::3][np.newaxis, :]
        dy = y1[:, np.newaxis] - sub2[1::3][np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r1_expanded[:, np.newaxis] + sub2[2::3][np.newaxis, :])**2
        if np.all(dist_sq >= min_dist_sq - 1e-6):
            return sub1_expanded.tolist() + sub2_permuted.tolist()
        else:
            # If expansion causes overlap, keep original sub1 and permute sub2
            return sub1.tolist() + sub2_permuted.tolist()

    # Perturb initial guess with geometric distortion
    v_perturbed = geometric_distortion(v0)
    # Apply topological reconfiguration
    v_reconfigured = topological_reconfiguration(v_perturbed)
    res = minimize(neg_sum_radii, v_reconfigured, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
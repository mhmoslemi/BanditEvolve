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

    def get_overlap_constraints(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dx = x_centers[:, np.newaxis] - x_centers[np.newaxis, :]
        dy = y_centers[:, np.newaxis] - y_centers[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r_radii[:, np.newaxis] + r_radii[np.newaxis, :])**2
        return dist_sq - min_dist_sq

    def get_overlap_constraints_list(v):
        overlap_cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                overlap_cons.append({"type": "ineq", "fun": constraint_func})
        return overlap_cons

    def component_reconfiguration(v):
        # Split into two groups
        group1 = v[0::3][:-13]
        group2 = v[0::3][13:]
        group1_y = v[1::3][:-13]
        group2_y = v[1::3][13:]
        group1_r = v[2::3][:-13]
        group2_r = v[2::3][13:]

        # Randomly permute the groups
        permuted_group1 = np.random.permutation(group1)
        permuted_group1_y = np.random.permutation(group1_y)
        permuted_group1_r = np.random.permutation(group1_r)
        
        permuted_group2 = np.random.permutation(group2)
        permuted_group2_y = np.random.permutation(group2_y)
        permuted_group2_r = np.random.permutation(group2_r)
        
        # Rebuild the decision vector
        new_v = np.zeros(3 * n)
        new_v[0::3] = np.concatenate([permuted_group1, permuted_group2])
        new_v[1::3] = np.concatenate([permuted_group1_y, permuted_group2_y])
        new_v[2::3] = np.concatenate([permuted_group1_r, permuted_group2_r])
        
        # Ensure boundaries are respected
        new_v[0::3] = np.clip(new_v[0::3], 0.0, 1.0)
        new_v[1::3] = np.clip(new_v[1::3], 0.0, 1.0)
        new_v[2::3] = np.clip(new_v[2::3], 1e-4, 0.5)
        return new_v

    cons.extend(get_overlap_constraints_list(v0))
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0

    # Apply component reconfiguration
    v = component_reconfiguration(v)

    # Final optimization
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
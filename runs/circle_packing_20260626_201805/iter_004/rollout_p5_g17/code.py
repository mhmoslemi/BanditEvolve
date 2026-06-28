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

    def component_reconfiguration(v):
        # Divide circles into two components
        component1 = v[0::3][0:13]
        component2 = v[0::3][13:]
        component1_y = v[1::3][0:13]
        component2_y = v[1::3][13:]
        component1_r = v[2::3][0:13]
        component2_r = v[2::3][13:]

        # Define a random permutation of the second component
        permuted_component2 = component2[np.random.permutation(13)]
        permuted_component2_y = component2_y[np.random.permutation(13)]
        permuted_component2_r = component2_r[np.random.permutation(13)]

        # Reconstruct the decision vector
        new_v = np.zeros_like(v)
        new_v[0::3][0:13] = component1
        new_v[1::3][0:13] = component1_y
        new_v[2::3][0:13] = component1_r
        new_v[0::3][13:] = permuted_component2
        new_v[1::3][13:] = permuted_component2_y
        new_v[2::3][13:] = permuted_component2_r

        # Ensure bounds are respected
        new_v[0::3] = np.clip(new_v[0::3], 0.0, 1.0)
        new_v[1::3] = np.clip(new_v[1::3], 0.0, 1.0)
        new_v[2::3] = np.clip(new_v[2::3], 1e-4, 0.5)
        return new_v

    # Perform initial optimization
    v_initial = v0 + 0.05 * np.random.rand(3 * n)
    v_initial = np.clip(v_initial, 0.0, 1.0)
    v_initial[2::3] = np.clip(v_initial[2::3], 1e-4, 0.5)
    res = minimize(neg_sum_radii, v_initial, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0

    # Apply component reconfiguration
    v = component_reconfiguration(v)

    # Final optimization after reconfiguration
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
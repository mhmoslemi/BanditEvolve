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

    def topological_reconfiguration(v):
        clusters = []
        for i in range(0, n, 5):
            cluster = v[3*i:3*(i+5)]
            clusters.append(cluster)
        np.random.shuffle(clusters)
        reconfigured_v = np.concatenate(clusters)
        reconfigured_v[0::3] = np.clip(reconfigured_v[0::3], 0.0, 1.0)
        reconfigured_v[1::3] = np.clip(reconfigured_v[1::3], 0.0, 1.0)
        reconfigured_v[2::3] = np.clip(reconfigured_v[2::3], 1e-4, 0.5)
        return reconfigured_v

    def forced_radius_increase(v):
        r_radii = v[2::3]
        avg_radius = np.mean(r_radii)
        target_radius = avg_radius * 1.05
        indices = np.where(r_radii < target_radius)[0]
        if len(indices) > 0:
            indices = np.random.choice(indices, size=min(5, len(indices)), replace=False)
            for i in indices:
                v[3*i+2] = np.clip(target_radius, 1e-4, 0.5)
        return v

    v_distorted = v0 + 0.05 * np.random.rand(3 * n)
    v_distorted[0::3] = np.clip(v_distorted[0::3], 0.0, 1.0)
    v_distorted[1::3] = np.clip(v_distorted[1::3], 0.0, 1.0)
    v_distorted[2::3] = np.clip(v_distorted[2::3], 1e-4, 0.5)
    v_reconfigured = topological_reconfiguration(v_distorted)
    v_forced = forced_radius_increase(v_reconfigured)

    res = minimize(neg_sum_radii, v_forced, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0

    v = topological_reconfiguration(v)
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
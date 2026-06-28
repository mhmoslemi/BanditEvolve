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

    # Vectorized overlap constraints with better performance
    def vectorized_overlap_constraint(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dx = x_centers[:, np.newaxis] - x_centers[np.newaxis, :]
        dy = y_centers[:, np.newaxis] - y_centers[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r_radii[:, np.newaxis] + r_radii[np.newaxis, :])**2
        return dist_sq - min_dist_sq

    # Convert to list of functions for each pair
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_cons.append({"type": "ineq", "fun": constraint_func})

    cons.extend(overlap_cons)

    # Apply dual-phase mutation strategy

    # Phase 1: Random geometric distortion to the entire layout
    def apply_geometric_distortion(v):
        # Scale the initial guess by a random factor between 0.9 and 1.1
        scale = np.random.uniform(0.9, 1.1)
        v = v * scale
        # Shear the layout slightly
        shear = np.random.uniform(-0.05, 0.05)
        v[0::3] += shear * v[1::3]
        v[1::3] += shear * v[0::3]
        return v

    v_distorted = apply_geometric_distortion(v0)
    res = minimize(neg_sum_radii, v_distorted, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Phase 2: Localized perturbation of smallest radius circles
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    if radii.shape[0] > 0:
        sorted_indices = np.argsort(radii)
        smallest_indices = sorted_indices[:5]  # Perturb smallest 5 circles

        # Perturb centers and radii of smallest circles
        for idx in smallest_indices:
            v[3*idx] += np.random.uniform(-0.01, 0.01)
            v[3*idx+1] += np.random.uniform(-0.01, 0.01)
            v[3*idx+2] += np.random.uniform(-0.001, 0.001)

        # Re-optimize with perturbed values
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
        v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
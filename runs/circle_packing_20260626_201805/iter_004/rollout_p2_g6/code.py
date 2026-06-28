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

    # Vectorized overlap constraints for improved performance
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_cons.append({"type": "ineq", "fun": constraint_func})

    cons.extend(overlap_cons)

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Subcomponent reconfiguration and permutation strategy
    def reconfigure_subcomponents(centers, radii):
        # Split into 2 subcomponents (A and B)
        subcomponent_A = centers[:13]
        subcomponent_B = centers[13:]
        subcomponent_A_radii = radii[:13]
        subcomponent_B_radii = radii[13:]

        # Randomly permute subcomponents
        np.random.shuffle([subcomponent_A, subcomponent_B])
        np.random.shuffle([subcomponent_A_radii, subcomponent_B_radii])

        # Combine back
        centers = np.vstack([subcomponent_A, subcomponent_B])
        radii = np.concatenate([subcomponent_A_radii, subcomponent_B_radii])

        # Apply forced expansion to at least one subcomponent
        expand_indices = np.random.choice(13, size=1, replace=False)
        for idx in expand_indices:
            radii[idx] = min(radii[idx] * 1.1, 0.5)

        # Ensure all circles are within bounds
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0:
                centers[i, 0] = r
            elif x + r > 1:
                centers[i, 0] = 1 - r
            if y - r < 0:
                centers[i, 1] = r
            elif y + r > 1:
                centers[i, 1] = 1 - r

        return centers, radii

    # Apply the targeted reconfiguration strategy
    centers, radii = reconfigure_subcomponents(centers, radii)

    # Local refinement
    def local_refinement(centers, radii):
        for _ in range(100):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-6:
                        overlap = radii[i] + radii[j] - dist
                        dx /= dist
                        dy /= dist
                        centers[i] += dx * overlap * 0.5
                        centers[j] -= dx * overlap * 0.5
                        centers[i] += dy * overlap * 0.5
                        centers[j] -= dy * overlap * 0.5
            # Ensure circles are within bounds
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if x - r < 0:
                    centers[i, 0] = r
                elif x + r > 1:
                    centers[i, 0] = 1 - r
                if y - r < 0:
                    centers[i, 1] = r
                elif y + r > 1:
                    centers[i, 1] = 1 - r
        return centers, radii

    # Shake heuristic for escaping local minima
    def shake_heuristic(centers, radii):
        for _ in range(5):
            small_indices = np.argsort(radii)[:5]
            for idx in small_indices:
                perturbation = np.random.normal(0, 1e-3, 2)
                centers[idx] += perturbation
                x, y = centers[idx]
                r = radii[idx]
                if x - r < 0:
                    centers[idx, 0] = r
                elif x + r > 1:
                    centers[idx, 0] = 1 - r
                if y - r < 0:
                    centers[idx, 1] = r
                elif y + r > 1:
                    centers[idx, 1] = 1 - r
            new_v = np.zeros(3 * n)
            new_v[0::3] = centers[:, 0]
            new_v[1::3] = centers[:, 1]
            new_v[2::3] = radii
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
            if res.success:
                v = res.x
                centers = np.column_stack([v[0::3], v[1::3]])
                radii = np.clip(v[2::3], 1e-6, None)
        return centers, radii

    # Local refinement
    centers, radii = local_refinement(centers, radii)
    centers, radii = shake_heuristic(centers, radii)

    return centers, radii, float(radii.sum())
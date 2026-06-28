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

    def perturb_initial_guess(v):
        perturbation = 0.05 * np.random.rand(3 * n)
        return v + perturbation

    def shake_smallest(v):
        r_radii = v[2::3]
        smallest_indices = np.argsort(r_radii)[:5]
        perturbation = 0.05 * np.random.rand(3 * n)
        perturbation[3*smallest_indices] = 0.1 * np.random.rand(len(smallest_indices)*3)
        return v + perturbation

    def geometric_distortion(v):
        scale = 1.0 + 0.1 * np.random.rand()
        shear = 0.1 * np.random.rand() - 0.05
        centers = v[0::3] + v[1::3] * shear
        centers *= scale
        centers = np.clip(centers, 0.0, 1.0)
        radii = v[2::3] * scale
        radii = np.clip(radii, 1e-4, 0.5)
        new_v = np.zeros(3 * n)
        new_v[0::3] = centers
        new_v[1::3] = centers
        new_v[2::3] = radii
        return new_v

    def component_permutation(v):
        # Split into independent subcomponents
        subcomponent_size = 13
        subcomponents = np.split(v, n // subcomponent_size)
        np.random.shuffle(subcomponents)
        new_v = np.concatenate(subcomponents)
        return new_v

    v_perturbed = perturb_initial_guess(v0)
    res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply component permutation and geometric distortion
    v_distorted = geometric_distortion(v)
    v_permuted = component_permutation(v_distorted)
    res_distorted = minimize(neg_sum_radii, v_permuted, method="SLSQP", bounds=bounds,
                             constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res_distorted.x if res_distorted.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Force expansion of one subcomponent
    subcomponent_size = 13
    subcomponents = np.split(centers, n // subcomponent_size)
    subcomponents_radii = np.split(radii, n // subcomponent_size)
    for i in range(n // subcomponent_size):
        if i == 0:
            # Expand this subcomponent
            max_radius = subcomponents_radii[i].max()
            expansion_factor = 1.1
            new_radii = subcomponents_radii[i] * expansion_factor
            new_radii = np.clip(new_radii, 1e-4, 0.5)
            subcomponents_radii[i] = new_radii
            # Adjust centers to maintain non-overlap
            for j in range(len(subcomponents_radii[i])):
                for k in range(j + 1, len(subcomponents_radii[i])):
                    dx = subcomponents[j][k, 0] - subcomponents[j][j, 0]
                    dy = subcomponents[j][k, 1] - subcomponents[j][j, 1]
                    dist = np.hypot(dx, dy)
                    if dist < subcomponents_radii[i][j] + subcomponents_radii[i][k] - 1e-6:
                        overlap = subcomponents_radii[i][j] + subcomponents_radii[i][k] - dist
                        dx /= dist
                        dy /= dist
                        subcomponents[j][k] += np.array([dx, dy]) * overlap * 0.5
                        subcomponents[j][j] -= np.array([dx, dy]) * overlap * 0.5
            # Adjust boundaries
            for j in range(len(subcomponents_radii[i])):
                x, y = subcomponents[j][j]
                r = subcomponents_radii[i][j]
                if x - r < 0:
                    subcomponents[j][j, 0] = r
                elif x + r > 1:
                    subcomponents[j][j, 0] = 1 - r
                if y - r < 0:
                    subcomponents[j][j, 1] = r
                elif y + r > 1:
                    subcomponents[j][j, 1] = 1 - r
            centers = np.vstack(subcomponents)
            radii = np.concatenate(subcomponents_radii)

    # Local refinement
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

    return centers, radii, float(radii.sum())
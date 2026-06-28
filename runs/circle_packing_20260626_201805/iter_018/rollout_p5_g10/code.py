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
            overlap_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: vectorized_overlap_constraint(v)[i, j]})

    cons.extend(overlap_cons)

    # Apply asymmetric radial and angular relationships to 30% of circles
    def apply_asymmetric_constraints(v):
        selected = np.random.choice(n, size=int(0.3 * n), replace=False)
        for i in selected:
            # Radial constraint: distance from a random center
            cx, cy = np.random.rand(2)
            dist = np.random.uniform(0.1, 0.3)
            cons.append({"type": "ineq", "fun": lambda v, cx=cx, cy=cy, dist=dist: 
                         np.sqrt((v[0::3] - cx)**2 + (v[1::3] - cy)**2) - dist})
            # Angular constraint: angle with respect to a random direction
            angle = np.random.uniform(0, 2*np.pi)
            cons.append({"type": "ineq", "fun": lambda v, angle=angle: 
                         np.arctan2(v[1::3], v[0::3]) - angle})
            # Ensure minimum radius increase
            r = v[2::3][i]
            new_r = r * 1.07
            new_r = np.clip(new_r, 1e-4, 0.5)
            v[3*i+2] = new_r
        return v

    # Initial perturbation and optimization
    v_distorted = apply_asymmetric_constraints(v0)
    v_perturbed = localized_perturbation(v_distorted)
    v = shake_heuristic(v_perturbed)

    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    def local_refinement(centers, radii):
        for _ in range(100):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
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
        return centers, radii

    centers, radii = local_refinement(centers, radii)

    # Global permutation of subcomponents for non-local reconfiguration
    def permute_subcomponents(v):
        indices = np.arange(n)
        np.random.shuffle(indices)
        subcomponents = np.array_split(indices, 5)
        for i, sub in enumerate(subcomponents):
            for j, other_sub in enumerate(subcomponents):
                if i != j:
                    for idx in sub:
                        v[3*idx], v[3*other_sub[0]] = v[3*other_sub[0]], v[3*idx]
                        v[3*idx+1], v[3*other_sub[0]+1] = v[3*other_sub[0]+1], v[3*idx+1]
                        v[3*idx+2], v[3*other_sub[0]+2] = v[3*other_sub[0]+2], v[3*idx+2]
        return v

    v = permute_subcomponents(v)

    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    return centers, radii, float(radii.sum())
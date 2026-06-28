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
                r_i = v[3*i+2]
                r_j = v[3*j+2]
                return dx*dx + dy*dy - (r_i + r_j)**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Define subcomponents based on initial arrangement
    subcomponents = [np.arange(4), np.arange(4, 8), np.arange(8, 12), np.arange(12, 16),
                     np.arange(16, 20), np.arange(20, 24), np.arange(24, 26)]
    subcomponent_radii = [np.mean(radii[subcomp]) for subcomp in subcomponents]

    # Apply radical topological reconfiguration
    def reconfigure_packing(centers, radii, subcomponents):
        # Permute subcomponents
        permutation = np.random.permutation(len(subcomponents))
        new_centers = np.zeros_like(centers)
        new_radii = np.zeros_like(radii)
        for i, subcomp in enumerate(subcomponents):
            permuted_subcomp = subcomponents[permutation[i]]
            new_centers[permuted_subcomp] = centers[subcomp]
            new_radii[permuted_subcomp] = radii[subcomp]
        return new_centers, new_radii

    # Enforce constraint that at least one subcomponent expands by 10%
    def enforce_radius_growth(centers, radii, subcomponents):
        growth_factor = 1.1
        for i, subcomp in enumerate(subcomponents):
            if np.random.rand() < 0.3:  # 30% chance to expand
                new_center = centers[subcomp]
                new_radius = radii[subcomp] * growth_factor
                # Adjust positions to ensure no overlap
                for j in range(n):
                    if j not in subcomp:
                        dx = new_center[0] - centers[j, 0]
                        dy = new_center[1] - centers[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < radii[j] + new_radius - 1e-6:
                            # Move new circle away
                            overlap = radii[j] + new_radius - dist
                            dx /= dist
                            dy /= dist
                            new_center[0] += dx * overlap * 0.5
                            new_center[1] += dy * overlap * 0.5
                # Ensure within bounds
                x, y = new_center
                r = new_radius
                if x - r < 0:
                    new_center[0] = r
                elif x + r > 1:
                    new_center[0] = 1 - r
                if y - r < 0:
                    new_center[1] = r
                elif y + r > 1:
                    new_center[1] = 1 - r
                centers[subcomp] = new_center
                radii[subcomp] = new_radius
        return centers, radii

    # Apply reconfiguration and radius growth constraint
    centers, radii = reconfigure_packing(centers, radii, subcomponents)
    centers, radii = enforce_radius_growth(centers, radii, subcomponents)

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

    # Apply dual-phase mutation strategy
    def apply_geometric_distortion(centers, radii):
        # Random scale
        scale_factor = np.random.uniform(0.9, 1.1)
        centers *= scale_factor
        # Random shear
        shear = np.random.uniform(-0.1, 0.1)
        centers[:, 0] += shear * centers[:, 1]
        # Random rotation
        angle = np.random.uniform(-np.pi/12, np.pi/12)
        cos_theta, sin_theta = np.cos(angle), np.sin(angle)
        centers = np.column_stack([centers[:, 0]*cos_theta - centers[:, 1]*sin_theta,
                                   centers[:, 0]*sin_theta + centers[:, 1]*cos_theta])
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

    # Additional perturbation of the initial guess
    def perturb_initial_guess(v):
        perturbation = 0.05 * np.random.rand(3 * n)
        return v + perturbation

    v_perturbed = perturb_initial_guess(v0)
    res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    centers, radii = local_refinement(centers, radii)
    centers, radii = shake_heuristic(centers, radii)
    centers, radii = apply_geometric_distortion(centers, radii)
    centers, radii = shake_heuristic(centers, radii)
    return centers, radii, float(radii.sum())
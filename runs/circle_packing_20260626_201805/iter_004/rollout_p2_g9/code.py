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

    # Local refinement
    def local_refinement(centers, radii):
        for _ in range(100):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-6:
                        # Move circles apart
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
            # Identify smallest circles
            small_indices = np.argsort(radii)[:5]
            # Perturb their positions
            for idx in small_indices:
                # Small perturbations
                perturbation = np.random.normal(0, 1e-3, 2)
                centers[idx] += perturbation
                # Ensure they stay within bounds
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
            # Re-optimization
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

    # Split into subcomponents and apply permutation
    def subcomponent_permutation(centers, radii):
        # Split into 2 groups of 13
        group1_centers = centers[:13]
        group1_radii = radii[:13]
        group2_centers = centers[13:]
        group2_radii = radii[13:]
        
        # Randomly permute the groups
        np.random.shuffle(group1_centers)
        np.random.shuffle(group1_radii)
        np.random.shuffle(group2_centers)
        np.random.shuffle(group2_radii)
        
        # Combine back
        new_centers = np.vstack([group1_centers, group2_centers])
        new_radii = np.concatenate([group1_radii, group2_radii])
        return new_centers, new_radii

    # Enforce minimum expansion of one subcomponent
    def enforce_subcomponent_expansion(centers, radii):
        # Split into 2 groups of 13
        group1_centers = centers[:13]
        group1_radii = radii[:13]
        group2_centers = centers[13:]
        group2_radii = radii[13:]
        
        # Randomly select one group to expand
        expand_group = np.random.choice([0, 1])
        if expand_group == 0:
            # Expand group1 by a fixed percentage
            expansion_rate = 0.05
            new_group1_radii = group1_radii * (1 + expansion_rate)
            new_group1_radii = np.clip(new_group1_radii, 1e-4, 0.5)
            new_radii = np.concatenate([new_group1_radii, group2_radii])
            # Re-optimization
            new_v = np.zeros(3 * n)
            new_v[0::3] = centers[:13, 0]
            new_v[1::3] = centers[:13, 1]
            new_v[2::3] = new_radii[:13]
            new_v[3*n - 3::3] = centers[13:, 0]
            new_v[3*n - 2::3] = centers[13:, 1]
            new_v[3*n - 1::3] = new_radii[13:]
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
            if res.success:
                v = res.x
                centers = np.column_stack([v[0::3], v[1::3]])
                radii = np.clip(v[2::3], 1e-6, None)
        else:
            # Expand group2 by a fixed percentage
            expansion_rate = 0.05
            new_group2_radii = group2_radii * (1 + expansion_rate)
            new_group2_radii = np.clip(new_group2_radii, 1e-4, 0.5)
            new_radii = np.concatenate([group1_radii, new_group2_radii])
            # Re-optimization
            new_v = np.zeros(3 * n)
            new_v[0::3] = centers[:13, 0]
            new_v[1::3] = centers[:13, 1]
            new_v[2::3] = new_radii[:13]
            new_v[3*n - 3::3] = centers[13:, 0]
            new_v[3*n - 2::3] = centers[13:, 1]
            new_v[3*n - 1::3] = new_radii[13:]
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
            if res.success:
                v = res.x
                centers = np.column_stack([v[0::3], v[1::3]])
                radii = np.clip(v[2::3], 1e-6, None)
        return centers, radii

    # Apply subcomponent permutation and expansion
    centers, radii = subcomponent_permutation(centers, radii)
    centers, radii = enforce_subcomponent_expansion(centers, radii)
    centers, radii = local_refinement(centers, radii)
    centers, radii = shake_heuristic(centers, radii)
    centers, radii = apply_geometric_distortion(centers, radii)
    centers, radii = shake_heuristic(centers, radii)
    return centers, radii, float(radii.sum())
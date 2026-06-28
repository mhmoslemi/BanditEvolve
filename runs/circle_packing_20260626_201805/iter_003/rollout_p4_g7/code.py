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

    # Vectorized overlap constraints for better performance
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

    # Implement a 'shake' heuristic to perturb smallest circles
    def shake_heuristic(v):
        radii = v[2::3]
        small_indices = np.argsort(radii)[:5]  # Select 5 smallest circles
        perturbation = 0.05 * np.random.rand(3 * n)
        v_perturbed = v + perturbation
        # Ensure perturbation doesn't cause circles to escape the square
        for i in small_indices:
            v_perturbed[3*i] = np.clip(v_perturbed[3*i], 0.0, 1.0)
            v_perturbed[3*i + 1] = np.clip(v_perturbed[3*i + 1], 0.0, 1.0)
            v_perturbed[3*i + 2] = np.clip(v_perturbed[3*i + 2], 1e-4, 0.5)
        return v_perturbed

    # Implement a probabilistic swap heuristic for diversity
    def probabilistic_swap_heuristic(v):
        # Randomly select a subset of circles to swap positions
        swap_indices = np.random.choice(n, size=8, replace=False)
        for i in range(0, len(swap_indices), 2):
            if i + 1 < len(swap_indices):
                idx1 = swap_indices[i]
                idx2 = swap_indices[i + 1]
                # Swap x and y positions
                v[3*idx1], v[3*idx2] = v[3*idx2], v[3*idx1]
                v[3*idx1+1], v[3*idx2+1] = v[3*idx2+1], v[3*idx1+1]
        return v

    # Try shaking the initial guess
    v_shaken = shake_heuristic(v0)
    v_swapped = probabilistic_swap_heuristic(v_shaken)
    res = minimize(neg_sum_radii, v_swapped, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
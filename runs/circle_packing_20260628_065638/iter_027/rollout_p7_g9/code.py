import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with stochastic jitter for geometric diversity and asymmetric spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add asymmetric jitter based on row and column to avoid uniform packing
        x = x_center + np.random.uniform(-0.07, 0.07) * (col / cols * 2)
        y = y_center + np.random.uniform(-0.07, 0.07) * (row / rows * 2)
        # Stagger alternating rows with asymmetric spacing
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Boundary constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Non-overlapping constraints between every pair of circles
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized overlap constraint
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with enhanced tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-10})
    
    # Jiggle heuristic: Perturb small circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify small radii circles for targeted perturbation
        small_radius_mask = radii < np.percentile(radii, 30) + 1e-4
        small_radius_indices = np.argwhere(small_radius_mask).flatten()
        # If at least one small radius circle, apply perturbation
        if len(small_radius_indices) > 0:
            # Perturb centers with geometric scaling of radii
            perturbation_factor = 0.1 * np.sqrt(radii[small_radius_indices]) / np.sqrt(np.mean(radii[small_radius_indices]))
            # Add small perturbation to centers
            perturbed_v = v.copy()
            for idx in small_radius_indices:
                perturbed_v[3*idx] += np.random.uniform(-0.02, 0.02) * perturbation_factor
                perturbed_v[3*idx+1] += np.random.uniform(-0.02, 0.02) * perturbation_factor
            # Re-optimize with perturbed small radius circles
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Final optimization to maximize total radii
    if res.success:
        v = res.x
        radii = v[2::3]
        current_total = np.sum(radii)
        # Check if we can expand some radii without overlapping
        dists = np.zeros((n, n))
        dx = v[0::3].reshape(n, 1) - v[0::3].reshape(1, n)
        dy = v[1::3].reshape(n, 1) - v[1::3].reshape(1, n)
        dists = np.sqrt(dx**2 + dy**2)
        overlaps = (dists < (radii + radii.T - 1e-12)).astype(int)
        # Calculate per-circle expansion potential
        expansion_potential = np.zeros(n)
        for i in range(n):
            min_overlap_dist = np.min(dists[i, ~np.arange(n) == i]) if np.sum(overlaps[i, ~np.arange(n) == i]) > 0 else np.inf
            expansion_potential[i] = (min_overlap_dist - radii[i]) / 2 if min_overlap_dist > radii[i] else 0
        # Expand radius of circle with highest expansion potential
        if np.any(expansion_potential > 1e-8):
            max_potential_idx = np.argmax(expansion_potential)
            max_exp_base = expansion_potential[max_potential_idx]
            # Expand this circle by 20% of its potential
            expansion_amount = max_exp_base * 0.2
            new_radii = radii.copy()
            new_radii[max_potential_idx] += expansion_amount
            # Apply expansion and re-optimize
            new_v = v.copy()
            new_v[2::3] = new_radii
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
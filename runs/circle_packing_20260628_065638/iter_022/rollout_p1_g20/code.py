import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with staggered grid and randomized jitter for initial diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Jitter to break symmetry and reduce clustering
        jitter = np.random.uniform(-0.04, 0.04, size=2)
        x = x_center + jitter[0]
        y = y_center + jitter[1]
        # Stagger rows to avoid vertical alignment
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

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized circular distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply targeted shake heuristic to isolated circles
    if res.success:
        v = res.x
        radii = v[2::3]
        # Compute pairwise distances and interaction metrics
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Calculate interaction intensity (sum of distances to all other circles)
        interaction = np.sum(dists, axis=1)
        # Select top 3 isolated circles for perturbation
        least_interacted = np.argsort(interaction)[:3]
        for idx in least_interacted:
            # Small random perturbations to centers
            v[3*idx] += np.random.uniform(-0.02, 0.02)
            v[3*idx+1] += np.random.uniform(-0.02, 0.02)
            # Small adjustments to radii to avoid over-constrained space
            v[3*idx+2] += np.random.uniform(-0.001, 0.001)
        # Refine with smaller steps
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Trigger a controlled radius expansion on a least-constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        # Compute expansion factor
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        # Expand radius of least constrained circle
        v[3*least_constrained_idx+2] += expansion_factor * 1.1
        # Distribute expansion to other circles to maintain constraint satisfaction
        for i in range(n):
            if i != least_constrained_idx:
                v[3*i+2] += expansion_factor * 0.9
        # Re-optimize
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Final optimization with tight tolerances
    if res.success:
        v = res.x
        radii = v[2::3]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Final refinement with directional perturbation
        for i in range(n):
            if radii[i] < 0.05:
                # Perturb small circles more aggressively to find better placements
                v[3*i] += np.random.uniform(-0.04, 0.04)
                v[3*i+1] += np.random.uniform(-0.04, 0.04)
                v[3*i+2] += np.random.uniform(-0.002, 0.002)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
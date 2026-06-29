import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with a dynamic fractal-based placement
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Apply a fractal-like perturbation with row-dependent scaling
        x += np.random.uniform(-0.05, 0.05) * (1.0 / (row + 1))
        y += np.random.uniform(-0.05, 0.05) * (1.0 / (row + 1))
        # Alternate row staggering with scaled offset
        if row % 2 == 1:
            x += 0.5 / cols / (row + 1)
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

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Enforce radical spatial reconfiguration by perturbing outermost circles
    if res.success:
        v = res.x
        # Identify outermost circles based on distance to square boundaries
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        dist_to_boundary = np.zeros(n)
        for i in range(n):
            x, y, r = centers[0][i], centers[1][i], radii[i]
            dist_to_boundary[i] = min(x - r, 1.0 - x - r, y - r, 1.0 - y - r)
        # Perturb outermost circles to trigger new spatial layout
        outermost_indices = np.argsort(dist_to_boundary)[:5]
        perturbation = np.random.rand(len(outermost_indices), 2) * 0.08
        perturbed_v = v.copy()
        for i, idx in enumerate(outermost_indices):
            perturbed_v[3*idx] += perturbation[i, 0]
            perturbed_v[3*idx+1] += perturbation[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted expansion of outermost circle with boundary enforcement
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate boundary proximity for outermost circle
        dist_to_boundary = np.zeros(n)
        for i in range(n):
            x, y, r = centers[0][i], centers[1][i], radii[i]
            dist_to_boundary[i] = min(x - r, 1.0 - x - r, y - r, 1.0 - y - r)
        # Expand the most boundary-proximate circle with constraint enforcement
        most_boundary_idx = np.argmin(dist_to_boundary)
        # Expand radius slightly and adjust position to maintain feasibility
        v[3*most_boundary_idx + 2] += 0.003
        v[3*most_boundary_idx] += 0.005
        v[3*most_boundary_idx+1] += 0.005
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
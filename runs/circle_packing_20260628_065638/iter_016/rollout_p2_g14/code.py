import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
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
    
    # Apply 'shake' heuristic: perturb small circles and re-optimize
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the 5 smallest circles
        min_indices = np.argsort(radii)[:5]
        # Perturb their positions slightly
        perturbation = np.random.rand(5, 2) * 0.05
        for i in min_indices:
            v[3*i] += perturbation[i % 5, 0]
            v[3*i+1] += perturbation[i % 5, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion with hard constraint on total sum of radii
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius
        min_radius = np.min(radii)
        min_radius_idx = np.argmin(radii)
        # Expand radius slightly and adjust position to maintain feasibility
        # Add hard constraint: total sum of radii must not exceed a given upper limit
        # This encourages more compact configurations with larger individual radii
        max_total_radius = 2.65
        current_total_radius = np.sum(radii)
        # Calculate how much we can increase the smallest radius without exceeding the limit
        max_possible_increase = max_total_radius - current_total_radius
        max_increase_for_min_radius = max_possible_increase / (n - 1) if n > 1 else max_possible_increase
        # Expand the smallest radius by a fraction of this value
        v[3*min_radius_idx + 2] += max_increase_for_min_radius * 0.05
        # Adjust the position slightly to maintain feasibility
        v[3*min_radius_idx] += np.random.uniform(-0.005, 0.005)
        v[3*min_radius_idx+1] += np.random.uniform(-0.005, 0.005)
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
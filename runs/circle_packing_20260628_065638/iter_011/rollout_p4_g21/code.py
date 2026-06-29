import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with randomized geometric clustering and asymmetry
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
        # Alternate row staggering for asymmetry
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

    # Radical spatial reconfiguration: dynamic fractal-based placement
    if res.success:
        v = res.x
        # Generate new initial positions based on a fractal-like distribution
        new_xs = []
        new_ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            # Introduce fractal-like perturbation with row-dependent scaling
            fractal_perturbation = np.random.uniform(-0.03, 0.03) * (0.5 ** row)
            x += fractal_perturbation
            y += fractal_perturbation
            # Alternate row staggering for asymmetry
            if row % 2 == 1:
                x += 0.5 / cols
            new_xs.append(x)
            new_ys.append(y)
        # Create perturbed vector based on new positions
        perturbed_v = v.copy()
        perturbed_v[0::3] = np.array(new_xs)
        perturbed_v[1::3] = np.array(new_ys)
        # Re-evaluate with new spatial constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    # Targeted radius expansion: expand outermost circle while maintaining non-overlap
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Identify the outermost circle based on distance to boundary
        distances_to_boundary = np.zeros(n)
        for i in range(n):
            x, y, r = centers[0][i], centers[1][i], radii[i]
            dist_to_left = x - r
            dist_to_right = 1.0 - x - r
            dist_to_bottom = y - r
            dist_to_top = 1.0 - y - r
            distances_to_boundary[i] = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
        most_outer_idx = np.argmin(distances_to_boundary)
        # Expand its radius slightly and adjust its position
        v[3*most_outer_idx + 2] += 0.005
        v[3*most_outer_idx] += 0.002
        v[3*most_outer_idx+1] += 0.002
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Randomized offset with row-dependent asymmetry
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
    
    # Radical spatial reconfiguration with fractal-based placement
    if res.success:
        v = res.x
        # Create a fractal-like spatial configuration by alternating row spacing
        new_xs = []
        new_ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            base_x = (col + 0.5) / cols
            base_y = (row + 0.5) / rows
            # Introduce a fractal-like offset based on row and column indices
            fractal_offset = 0.03 * np.sin(2 * np.pi * (row + col) / n)
            new_x = base_x + fractal_offset
            new_y = base_y + fractal_offset
            new_xs.append(new_x)
            new_ys.append(new_y)
        # Perturb the new positions slightly to avoid exact repetition
        perturbation = np.random.rand(n, 2) * 0.03
        new_xs = np.array(new_xs) + perturbation[:, 0]
        new_ys = np.array(new_ys) + perturbation[:, 1]
        # Re-initialize the decision vector with new positions
        new_v = v.copy()
        new_v[0::3] = new_xs
        new_v[1::3] = new_ys
        # Re-evaluate with the new spatial configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Enforce strict non-overlapping boundaries for outermost circle
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Identify the outermost circle (maximum distance from center)
        distances_from_center = np.sqrt((centers[0] - 0.5)**2 + (centers[1] - 0.5)**2)
        outermost_idx = np.argmax(distances_from_center)
        # Expand its radius slightly while keeping it within bounds
        v[3*outermost_idx + 2] += 0.003
        # Re-adjust its position to maintain feasibility
        v[3*outermost_idx] += 0.005
        v[3*outermost_idx+1] += 0.005
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
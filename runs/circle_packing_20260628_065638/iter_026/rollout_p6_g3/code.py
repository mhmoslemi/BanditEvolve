import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initial positions using grid with random perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add randomized offset to prevent perfect symmetry
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
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
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Boundary constraints (x+radius ≤ 1, x-radius ≥ 0, y+radius ≤ 1, y-radius ≥ 0)
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints (dx^2 + dy^2 >= r_i^2 + r_j^2)
    for i in range(n):
        for j in range(i+1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})

    if res.success:
        v = res.x

        # Step 1: Randomized geometric hashing for layout reordering
        # Create a geometric hash based on random seed for each circle
        component_hash = np.random.rand(n, 2) * 0.04
        # Randomly perturb positions to trigger complete reordering
        for i in range(n):
            v[3*i] += component_hash[i, 0] * (1.0 + np.random.rand())
            v[3*i+1] += component_hash[i, 1] * (1.0 + np.random.rand())
        
        # Step 2: Second optimization after hashing
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Step 3: Find circle with smallest radius (least constrained)
        min_radius_idx = np.argmin(radii)
        min_radius = radii[min_radius_idx]

        # Step 4: Expand this circle more to trigger reordering
        target_total = np.sum(radii) + 0.01
        expansion = (target_total - np.sum(radii)) / (n - 1)
        # Over-expand the smallest circle to trigger global reordering
        expanded_radii = radii.copy()
        expanded_radii[min_radius_idx] += expansion * 1.3

        # Step 5: Construct new vector and re-optimizing
        v_expanded = v.copy()
        v_expanded[2::3] = expanded_radii

        # Step 6: Final optimization with reconfigured radii
        res_final = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                             constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
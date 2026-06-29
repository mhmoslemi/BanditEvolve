import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with staggered grid and random offsets
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base positions with staggered rows
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x = base_x + np.random.uniform(-0.05, 0.05)
        y = base_y + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii to a reasonable base
    r0 = 0.28 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for x, y, and radius
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to minimize (negative total sum of radii)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Define constraints for boundaries
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Define overlap constraints between circles
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-10})

    # Apply perturbation to smallest circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the smallest 5 circles
        smallest_indices = np.argsort(radii)[:5]
        # Apply small perturbations to their positions and radii
        for i in smallest_indices:
            v[3*i] += np.random.uniform(-0.02, 0.02)
            v[3*i+1] += np.random.uniform(-0.02, 0.02)
            v[3*i+2] += np.random.uniform(-0.002, 0.002)
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-10})

    # Apply targeted radius expansion if needed
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius
        min_radius_idx = np.argmin(radii)
        # Calculate current total sum
        total_sum = np.sum(radii)
        # Try expanding the smallest circle by a small amount
        expansion_factor = 0.01
        new_radii = radii.copy()
        new_radii[min_radius_idx] += expansion_factor
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
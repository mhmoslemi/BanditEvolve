import numpy as np

def run_packing():
    n = 26
    cols = 5  # Hexagonal grid with 5 columns for better distribution
    rows = (n + cols - 1) // cols  # Calculate the number of rows
    
    # Initialize positions using a hexagonal grid pattern with some variation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Offset even rows for hexagonal packing
        if row % 2 == 1:
            x += 0.5 / cols
        # Add slight randomness to break symmetry and allow better expansion
        xs.append(x + np.random.uniform(-0.02, 0.02))
        ys.append(y + np.random.uniform(-0.02, 0.02))
    
    r0 = 0.3 / cols - 1e-3  # Starting radius based on grid spacing
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure bounds match vector length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize negative sum to maximize radii

    cons = []
    # Add constraints for box boundaries
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Add constraints for circle overlaps
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First stage: Coarse global optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    # Second stage: Local refinement using L-BFGS-B with tighter tolerances
    if res.success:
        v = res.x
        # Find the circle with the largest radius to focus refinement
        max_radius_index = np.argmax(v[2::3])
        # Small local perturbation to escape local optima
        v[3*max_radius_index + 0] += 0.01 * np.random.rand()
        v[3*max_radius_index + 1] += 0.01 * np.random.rand()
        v[3*max_radius_index + 2] += 0.001 * np.random.rand()
        # Re-constrain to ensure validity
        v[3*max_radius_index + 2] = np.clip(v[3*max_radius_index + 2], 1e-4, 0.5)
        # Refine with L-BFGS-B for faster convergence
        res = minimize(neg_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
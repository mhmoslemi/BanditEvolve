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
    
    # Radical reconfiguration: enforce randomized geometric hashing and reordering
    if res.success:
        v = res.x
        # Apply randomized geometric hashing to break symmetries and explore new configurations
        hash_centers = np.random.rand(n, 2) * 0.8 + 0.1
        # Create new positions by mapping hash values to unit square
        hash_radii = np.random.rand(n) * 0.05 + 0.1
        # Compute new positions using inverse square root mapping
        new_x = hash_centers[:, 0] / np.sqrt(np.sum(hash_centers**2, axis=1))
        new_y = hash_centers[:, 1] / np.sqrt(np.sum(hash_centers**2, axis=1))
        new_x = new_x * 0.8 + 0.1
        new_y = new_y * 0.8 + 0.1
        # Reorder circles based on hash values
        order = np.argsort(hash_centers[:, 0])
        perturbed_v = np.zeros(3 * n)
        perturbed_v[0::3] = new_x[order]
        perturbed_v[1::3] = new_y[order]
        perturbed_v[2::3] = hash_radii[order]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion with strict non-overlap and topological reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        # Compute total sum of radii
        total_sum = np.sum(radii)
        # Enforce strict non-overlap while expanding the smallest radius
        for _ in range(3):
            # Create a copy for manipulation
            v_copy = v.copy()
            # Expand the smallest radius
            v_copy[3*smallest_radius_idx + 2] += 0.001
            # Re-evaluate with adjusted parameters
            res = minimize(neg_sum_radii, v_copy, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
            if res.success:
                v = res.x
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
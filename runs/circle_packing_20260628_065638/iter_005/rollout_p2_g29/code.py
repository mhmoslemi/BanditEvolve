import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a hexagonal grid with randomized offsets for perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Introduce random offset to break symmetry and allow better expansion
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
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
    
    # Vectorized overlap constraint calculation to improve performance
    if res.success:
        v = res.x
        # Extract centers and radii
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Create vectorized versions for efficient computation
        x_centers = np.reshape(centers[0], (n, 1))
        y_centers = np.reshape(centers[1], (n, 1))
        r = np.reshape(radii, (n, 1))
        # Compute pairwise distance squared
        dx = x_centers - np.swapaxes(x_centers, 0, 1)
        dy = y_centers - np.swapaxes(y_centers, 0, 1)
        dist_sq = dx*dx + dy*dy
        min_dist_sq = (r + np.swapaxes(r, 0, 1))**2
        # Apply constraints as a vectorized operation
        constraint_values = dist_sq - min_dist_sq
        # Apply a small perturbation to the most constrained circles
        constraint_mask = np.abs(constraint_values) < 1e-6
        isolated_indices = np.any(constraint_mask, axis=1)
        isolated_indices = np.where(isolated_indices)[0]
        if len(isolated_indices) > 0:
            idx = np.random.choice(isolated_indices)
            v[3*idx + 2] += 0.002
            v[3*idx + 0] += 0.005
            v[3*idx + 1] += 0.005
            # Re-optimize with perturbed parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
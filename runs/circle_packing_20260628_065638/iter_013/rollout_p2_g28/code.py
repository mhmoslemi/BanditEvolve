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
        # Randomized offset to break symmetry
        x += np.random.uniform(-0.07, 0.07)
        y += np.random.uniform(-0.07, 0.07)
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
    
    # Trigger complete topological overhaul with randomized geometric hashing
    if res.success:
        v = res.x
        # Create a randomized geometric hashing grid
        hash_grid = np.random.rand(n, 2)
        hash_grid[:, 0] = np.sort(hash_grid[:, 0])
        hash_grid[:, 1] = np.sort(hash_grid[:, 1])
        # Map to unit square
        hash_grid = hash_grid * 2 - 1
        hash_grid = (hash_grid + 1) / 2
        # Perturb positions to avoid exact same configuration
        perturbation = np.random.rand(n, 2) * 0.05
        new_v = np.zeros(3 * n)
        new_v[0::3] = hash_grid[:, 0] + perturbation[:, 0]
        new_v[1::3] = hash_grid[:, 1] + perturbation[:, 1]
        new_v[2::3] = v[2::3]
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Radical radius expansion for smallest non-zero radius
    if res.success:
        v = res.x
        radii = v[2::3]
        min_radius_idx = np.argmin(radii[radii > 1e-6])
        # Expand radius and adjust position
        v[3*min_radius_idx + 2] += 0.01
        v[3*min_radius_idx] += np.random.uniform(-0.02, 0.02)
        v[3*min_radius_idx + 1] += np.random.uniform(-0.02, 0.02)
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
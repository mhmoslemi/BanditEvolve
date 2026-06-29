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
        x += np.random.uniform(-0.08, 0.08)
        y += np.random.uniform(-0.08, 0.08)
        
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
    
    # Topological overhaul: replace positions with randomized geometric hashing
    if res.success:
        v = res.x
        # Random geometric hashing initialization
        hash_grid = np.random.rand(6, 6)
        hash_idx = np.argsort(hash_grid.flatten())[:n]
        xs_new = []
        ys_new = []
        for i in hash_idx:
            x = (i % 6 + 0.5) / 6
            y = (i // 6 + 0.5) / 6
            x += np.random.uniform(-0.05, 0.05)
            y += np.random.uniform(-0.05, 0.05)
            xs_new.append(x)
            ys_new.append(y)
        # Randomize small perturbation
        perturbation = np.random.rand(n, 2) * 0.05
        perturbed_v = np.zeros(3 * n)
        perturbed_v[0::3] = np.array(xs_new) + perturbation[:, 0]
        perturbed_v[1::3] = np.array(ys_new) + perturbation[:, 1]
        perturbed_v[2::3] = v[2::3]
        # Re-evaluate with new positions
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Radical radius expansion: expand the circle with smallest radius
    if res.success:
        v = res.x
        radii = v[2::3]
        min_radius_idx = np.argmin(radii)
        # Expand its radius and adjust position
        v[3*min_radius_idx + 2] += 0.005
        v[3*min_radius_idx] += 0.003
        v[3*min_radius_idx+1] += 0.003
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
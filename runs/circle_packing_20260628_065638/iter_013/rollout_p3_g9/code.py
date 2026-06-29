import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x += np.random.uniform(-0.1, 0.1)
        y += np.random.uniform(-0.1, 0.1)
        # Introduce local clusterization
        if np.random.rand() < 0.3:
            cluster_x = np.random.uniform(-0.05, 0.05)
            cluster_y = np.random.uniform(-0.05, 0.05)
            x += cluster_x
            y += cluster_y
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
    
    # Topological overhaul: randomize spatial configuration and re-optimize
    if res.success:
        v = res.x
        # Generate randomized spatial configuration
        xs_rand = np.random.uniform(0.05, 0.95, n)
        ys_rand = np.random.uniform(0.05, 0.95, n)
        # Perturb positions to trigger new configuration
        perturbation = np.random.rand(n, 2) * 0.05
        vRand = np.empty(3 * n)
        vRand[0::3] = xs_rand + perturbation[:, 0]
        vRand[1::3] = ys_rand + perturbation[:, 1]
        vRand[2::3] = v[2::3]  # Keep radii fixed for this step
        
        # Re-evaluate with randomized spatial configuration
        res = minimize(neg_sum_radii, vRand, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Radical radius expansion: expand the smallest circle and re-optimize
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the smallest circle
        min_radius_idx = np.argmin(radii)
        # Expand its radius and adjust its position to maintain feasibility
        v[3*min_radius_idx + 2] += 0.005  # Expand radius
        v[3*min_radius_idx] += 0.01  # Shift position
        v[3*min_radius_idx+1] += 0.01  # Shift position
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
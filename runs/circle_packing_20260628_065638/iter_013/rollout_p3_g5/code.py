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
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
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
    
    # Asymmetric reconfiguration: substitute spatial arrangement with randomized geometric hashing
    if res.success:
        v = res.x
        # Create a new randomized grid with hash-based distribution
        new_xs = []
        new_ys = []
        for i in range(n):
            # Randomized geometric hashing
            hash_val = np.random.rand()
            x = hash_val * 0.8 + 0.1
            y = hash_val * 0.8 + 0.1
            # Ensure no overlap with boundaries
            x = np.clip(x, 0.1, 0.9)
            y = np.clip(y, 0.1, 0.9)
            new_xs.append(x)
            new_ys.append(y)
        # Perturb the positions slightly to break symmetry
        perturbation = np.random.rand(n, 2) * 0.03
        new_v = np.empty(3 * n)
        new_v[0::3] = np.array(new_xs) + perturbation[:, 0]
        new_v[1::3] = np.array(new_ys) + perturbation[:, 1]
        new_v[2::3] = v[2::3]
        
        # Re-evaluate with new parameters and original constraints
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Radical radius expansion: expand the smallest circle and reconfigure layout
    if res.success:
        v = res.x
        radii = v[2::3]
        smallest_idx = np.argmin(radii)
        # Expand the smallest circle
        v[3*smallest_idx + 2] += 0.005
        # Perturb its position to trigger layout reconfiguration
        perturbation = np.random.rand(2) * 0.05
        v[3*smallest_idx] += perturbation[0]
        v[3*smallest_idx+1] += perturbation[1]
        # Re-evaluate with updated parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
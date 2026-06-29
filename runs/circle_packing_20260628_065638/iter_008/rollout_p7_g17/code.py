import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using Voronoi tessellation-inspired placement with enhanced randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Introduce variation to break symmetry and allow better expansion
        if row % 3 == 1:
            x += np.random.uniform(-0.05 / cols, 0.05 / cols)
        if row % 2 == 1:
            y += np.random.uniform(-0.05 / rows, 0.05 / rows)
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

    # Vectorized constraint functions for better performance
    def constraint_func(i, j, v):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
    
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(i, j, v)})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Asymmetric reconfiguration: randomize positions and expand least constrained circle
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        
        # Randomize positions for asymmetric reconfiguration
        random_mask = np.random.rand(n) < 0.2
        v[random_mask * 3] += np.random.uniform(-0.05, 0.05, np.sum(random_mask))
        v[random_mask * 3 + 1] += np.random.uniform(-0.05, 0.05, np.sum(random_mask))
        v[random_mask * 3 + 2] = np.clip(v[random_mask * 3 + 2], 1e-4, 0.5)
        
        # Find the least constrained circle (maximally spaced from others)
        dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dists[i] += np.sqrt(dx*dx + dy*dy)
        least_constrained_idx = np.argmax(dists)
        
        # Expand radius of least constrained circle with small perturbation
        v[3*least_constrained_idx + 2] = np.clip(v[3*least_constrained_idx + 2] + 0.001, 1e-4, 0.5)
        v[3*least_constrained_idx] += np.random.uniform(-0.01, 0.01)
        v[3*least_constrained_idx + 1] += np.random.uniform(-0.01, 0.01)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
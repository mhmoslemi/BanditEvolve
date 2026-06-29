import numpy as np

def run_packing():
    n = 26
    # Generate initial positions using a Voronoi tessellation-based approach
    # Start with a grid of points and then use a Poisson disk sampling technique
    # to ensure even distribution with minimum spacing
    
    # Initial grid of points
    cols = 6
    rows = 5
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Offset for hexagonal packing
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Perturb positions using a random walk to break symmetry
    np.random.seed(42)
    for i in range(n):
        xs[i] += np.random.uniform(-0.05, 0.05)
        ys[i] += np.random.uniform(-0.05, 0.05)
    
    # Clip positions to ensure they stay within the unit square
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)
    
    r0 = 0.5 / cols - 1e-3
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

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-10})
    
    # Local refinement step: perturb the most isolated circle and re-optimize
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dists[i] += np.linalg.norm(centers[i] - centers[j]) / radii[j]
        max_isolation_index = np.argmax(dists)
        v[3*max_isolation_index + 2] += 0.001  # Small radius increment
        v[3*max_isolation_index + 0] += 0.005  # Move circle slightly
        v[3*max_isolation_index + 1] += 0.005
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    # Use Voronoi-based initial placement to distribute circles more evenly
    # Generate initial points using a random grid with Voronoi tessellation
    np.random.seed(42)
    initial_points = np.random.rand(n, 2)
    # Initialize radii with a small value
    r0 = 0.01
    v0 = np.empty(3 * n)
    v0[0::3] = initial_points[:, 0]
    v0[1::3] = initial_points[:, 1]
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
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    
    # Local refinement step: perturb the most isolated circle and re-optimize
    if res.success:
        v = res.x
        radii = v[2::3]
        # Compute isolation measure: sum of inverse distances to all other circles
        def isolation_measure(v):
            center = v[:2]
            r = v[2]
            total = 0.0
            for i in range(n):
                if i != idx:
                    dx = v[3*i] - center[0]
                    dy = v[3*i+1] - center[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > r:
                        total += 1.0 / dist
            return total
        # Find the circle with the highest isolation measure
        max_isolation_index = np.argmax([isolation_measure(v[:3*i+3]) for i in range(n)])
        v = res.x
        # Perturb the most isolated circle
        v[3*max_isolation_index + 2] += 0.001  # Small radius increment
        v[3*max_isolation_index + 0] += 0.005  # Move circle slightly
        v[3*max_isolation_index + 1] += 0.005
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-9})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with dynamic fractal-based spatial distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid position
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Fractal-like offset with row-dependent asymmetry
        x += np.random.uniform(-0.1, 0.1)
        y += np.random.uniform(-0.1, 0.1)
        # Alternate row staggering with fractal-like perturbation
        if row % 2 == 1:
            x += np.random.uniform(-0.15, 0.15)
        xs.append(x)
        ys.append(y)
    
    # Initial radius calculation with tighter spacing
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

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with tighter tolerance
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Radial expansion of outermost circle with constraint enforcement
    if res.success:
        v = res.x
        # Calculate distances from all circles to the boundary
        boundary_distances = np.zeros(n)
        for i in range(n):
            x = v[3*i]
            y = v[3*i+1]
            r = v[3*i+2]
            boundary_distances[i] = min(x - r, 1.0 - x - r, y - r, 1.0 - y - r)
        # Identify the outermost circle
        outermost_idx = np.argmin(boundary_distances)
        # Expand its radius while maintaining boundary constraints
        v[3*outermost_idx + 2] += 0.005
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
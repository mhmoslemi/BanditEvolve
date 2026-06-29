import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Dynamic fractal-based placement to induce new spatial configurations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Fractal scaling to introduce non-uniform spacing
        scale = 1.0 / (2 ** (row + col))
        x += np.random.uniform(-0.5 * scale, 0.5 * scale)
        y += np.random.uniform(-0.5 * scale, 0.5 * scale)
        # Alternate row staggering for asymmetry
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
    
    # Radical spatial reconfiguration with fractal-based placement
    if res.success:
        v = res.x
        # Apply fractal-based reconfiguration to outermost circles
        outer_indices = np.argsort(np.hypot(v[0::3], v[1::3]))[:5]
        for i in outer_indices:
            v[3*i] = np.random.uniform(0.05, 0.95)
            v[3*i+1] = np.random.uniform(0.05, 0.95)
            v[3*i+2] = np.random.uniform(0.01, 0.05)
        # Re-evaluate with reconfigured parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Controlled expansion of outermost circle with strict boundary enforcement
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Identify outermost circle (maximum distance from center)
        distances = np.hypot(centers[0], centers[1])
        outermost_idx = np.argmax(distances)
        # Expand radius while maintaining non-overlap and boundary constraints
        v[3*outermost_idx + 2] += 0.003
        # Adjust position to maintain feasibility
        v[3*outermost_idx] += 0.005
        v[3*outermost_idx+1] += 0.005
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
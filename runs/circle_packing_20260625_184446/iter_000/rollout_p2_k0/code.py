import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

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
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint})

    # Try optimization with tighter tolerances
    try:
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9, "eps": 1e-9})
        v = res.x if res.success else v0
    except:
        v = v0

    # Final refinement: inflate radii slightly while keeping centers fixed
    # This is a heuristic to see if we can push radii further without violating constraints
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3].copy()
    max_radius_increase = 0.001
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - 1e-8:
                # If there is overlap, reduce the radii slightly
                radius_reduction = (dist - (radii[i] + radii[j] - 1e-8)) / 2
                radii[i] -= radius_reduction
                radii[j] -= radius_reduction
                radii[i] = np.clip(radii[i], 1e-6, None)
                radii[j] = np.clip(radii[j], 1e-6, None)
    
    # Ensure radii are within bounds and calculate final sum
    radii = np.clip(radii, 1e-6, 0.5)
    sum_radii = float(radii.sum())
    return centers, radii, sum_radii
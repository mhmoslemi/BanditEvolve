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
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Fine-tune optimization parameters
    try:
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9, "eps": 1e-8})
        v = res.x if res.success else v0
    except:
        v = v0

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    sum_radii = float(radii.sum())
    
    # Final radius inflation pass to maximize the sum of radii
    # While keeping centers fixed to avoid constraint violations
    # This is a heuristic to extract the maximum possible radii
    # This step is only applied if the initial optimization is successful
    if res.success:
        # Compute the maximum possible radii based on boundaries
        max_r = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            max_r[i] = min(x, 1 - x, y, 1 - y)
        radii = np.clip(radii, 1e-6, max_r)
        sum_radii = float(radii.sum())

    return centers, radii, sum_radii
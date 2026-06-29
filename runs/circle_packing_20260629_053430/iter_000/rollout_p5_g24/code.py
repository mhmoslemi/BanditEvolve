import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    
    # Initialize with hexagonal packing
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = np.full(n, 0.5 / (cols + 1))
    r0 = 0.5 / (cols + 1) - 1e-3
    v0 = np.empty(3 * n)
    
    for i in range(n):
        x = 0.5 + (np.cos(angles[i]) * (i // cols + 0.5) * (radii[i] * 2))
        y = 0.5 + (np.sin(angles[i]) * (i // cols + 0.5) * (radii[i] * 2))
        v0[3*i] = x
        v0[3*i+1] = y
        v0[3*i+2] = r0
    
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
                   constraints=cons, options={"maxiter": 5000, "ftol": 1e-10})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
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
        cons.append({"type": "ineq", "fun": (lambda i: lambda v: v[3*i] - v[3*i+2])(i)})
        cons.append({"type": "ineq", "fun": (lambda i: lambda v: 1.0 - v[3*i] - v[3*i+2])(i)})
        cons.append({"type": "ineq", "fun": (lambda i: lambda v: v[3*i+1] - v[3*i+2])(i)})
        cons.append({"type": "ineq", "fun": (lambda i: lambda v: 1.0 - v[3*i+1] - v[3*i+2])(i)})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": (lambda i, j: lambda v:
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                                 - (v[3*i+2] + v[3*j+2])**2)(i, j)})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
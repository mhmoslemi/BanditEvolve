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

    def constraint_outside(v, i):
        x = v[3*i]
        y = v[3*i+1]
        r = v[3*i+2]
        return x - r
    def constraint_inside(v, i):
        x = v[3*i]
        y = v[3*i+1]
        r = v[3*i+2]
        return 1.0 - x - r
    def constraint_left(v, i):
        x = v[3*i]
        y = v[3*i+1]
        r = v[3*i+2]
        return y - r
    def constraint_right(v, i):
        x = v[3*i]
        y = v[3*i+1]
        r = v[3*i+2]
        return 1.0 - y - r

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: constraint_outside(v, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: constraint_inside(v, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: constraint_left(v, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: constraint_right(v, i)})

    def constraint_overlap(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2

    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_overlap(v, i, j)})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-8})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
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

    def constraint_func(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        dist_sq = dx*dx + dy*dy
        min_dist_sq = (v[3*i+2] + v[3*j+2])**2
        return dist_sq - min_dist_sq

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v, i, j)})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Sub-component decomposition
    sub_parts = [np.arange(i*8, min((i+1)*8, n)) for i in range((n + 7) // 8)]
    sub_v = []
    for part in sub_parts:
        sub_v_part = v[3*part[0]:3*part[-1]+3]
        sub_v.append(sub_v_part)
    
    # Optimize each sub-component independently
    for i, part in enumerate(sub_parts):
        sub_bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * len(part)
        sub_cons = []
        for pi in part:
            sub_cons.append({"type": "ineq", "fun": lambda v, pi=pi: v[3*pi] - v[3*pi+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, pi=pi: 1.0 - v[3*pi] - v[3*pi+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, pi=pi: v[3*pi+1] - v[3*pi+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, pi=pi: 1.0 - v[3*pi+1] - v[3*pi+2]})
        for pi in part:
            for pj in part:
                if pi < pj:
                    sub_cons.append({"type": "ineq", "fun": lambda v, pi=pi, pj=pj: constraint_func(v, pi, pj)})
        res_sub = minimize(neg_sum_radii, sub_v[i], method="SLSQP", bounds=sub_bounds, constraints=sub_cons, options={"maxiter": 500, "ftol": 1e-9})
        sub_v[i] = res_sub.x if res_sub.success else sub_v[i]
    
    # Reassemble configuration
    for i, part in enumerate(sub_parts):
        v[3*part[0]:3*part[-1]+3] = sub_v[i]

    # Randomized spatial repositioning of sub-components
    np.random.seed(42)
    for i, part in enumerate(sub_parts):
        for pi in part:
            v[3*pi] += np.random.uniform(-0.05, 0.05)
            v[3*pi+1] += np.random.uniform(-0.05, 0.05)
            v[3*pi+2] += np.random.uniform(-0.01, 0.01)

    # Rebuild bounds and constraints for perturbed configuration
    bounds_final = []
    for _ in range(n):
        bounds_final += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_final = []
    for i in range(n):
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            cons_final.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v, i, j)})

    res_final = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds_final, constraints=cons_final, options={"maxiter": 500, "ftol": 1e-9})
    v = res_final.x if res_final.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
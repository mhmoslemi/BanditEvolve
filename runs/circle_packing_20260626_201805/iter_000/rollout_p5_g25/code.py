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

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Local refinement to resolve overlaps and boundary issues
    if res.success:
        def local_refinement(v, centers, radii):
            # Apply gentle penalty for overlaps and boundary violations
            penalty = 0.0
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    penalty += max(0, (x - r) - 1e-5) + max(0, (1 - x - r) - 1e-5) + \
                               max(0, (y - r) - 1e-5) + max(0, (1 - y - r) - 1e-5)
                for j in range(i + 1, n):
                    dx = x - centers[j][0]
                    dy = y - centers[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-5:
                        penalty += (radii[i] + radii[j] - dist) * 100
            return penalty

        # Refine the solution with a penalty function
        def constrained_neg_sum_radii(v):
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            penalty = local_refinement(v, centers, radii)
            return -np.sum(radii) + penalty

        res = minimize(constrained_neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
        v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
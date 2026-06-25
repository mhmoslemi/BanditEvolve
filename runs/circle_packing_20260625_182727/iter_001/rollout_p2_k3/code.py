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
        def make_ineq(i):
            def ineq(v):
                return v[3*i] - v[3*i + 2]
            return ineq
        cons.append({"type": "ineq", "fun": make_ineq(i)})
        def make_ineq2(i):
            def ineq(v):
                return 1.0 - v[3*i] - v[3*i + 2]
            return ineq
        cons.append({"type": "ineq", "fun": make_ineq2(i)})
        def make_ineq3(i):
            def ineq(v):
                return v[3*i + 1] - v[3*i + 2]
            return ineq
        cons.append({"type": "ineq", "fun": make_ineq3(i)})
        def make_ineq4(i):
            def ineq(v):
                return 1.0 - v[3*i + 1] - v[3*i + 2]
            return ineq
        cons.append({"type": "ineq", "fun": make_ineq4(i)})

    for i in range(n):
        for j in range(i + 1, n):
            def make_ineq_dist(i, j):
                def ineq(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i + 1] - v[3*j + 1]
                    return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
                return ineq
            cons.append({"type": "ineq", "fun": make_ineq_dist(i, j)})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Final projection pass to ensure constraints are satisfied
    def project(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0:
                v[3*i] = r
            elif x + r > 1:
                v[3*i] = 1 - r
            if y - r < 0:
                v[3*i+1] = r
            elif y + r > 1:
                v[3*i+1] = 1 - r
        return v

    v = project(v)
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Ensure no overlaps and small radii inflation
    for _ in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[3*i+2] + v[3*j+2]
                if dist_sq < r_sum*r_sum - 1e-8:
                    # Adjust radii to maintain tangency
                    delta = (r_sum*r_sum - dist_sq) / (2 * r_sum) - 1e-8
                    v[3*i+2] -= delta / 2
                    v[3*j+2] -= delta / 2
        # Reapply projection after adjustments
        v = project(v)

    return centers, radii, float(radii.sum())
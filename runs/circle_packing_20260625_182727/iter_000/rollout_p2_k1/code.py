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

    # Fine-tuned optimization parameters
    try:
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9, "eps": 1e-8})
        v = res.x if res.success else v0
    except:
        v = v0

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    sum_radii = float(radii.sum())

    # Final radius inflation pass with constraint checking
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < (radii[i] + radii[j])**2 - 1e-9:
                valid = False
                break
        if not valid:
            break

    if valid:
        # Try to slightly increase radii while maintaining constraints
        max_increase = 0.001
        new_radii = radii * (1 + max_increase)
        new_radii[new_radii > 0.5] = 0.5
        try:
            # Check if new radii are valid
            for i in range(n):
                if new_radii[i] < 1e-6 or new_radii[i] > 0.5:
                    valid = False
                    break
                x, y = centers[i]
                if not (0 <= x - new_radii[i] and x + new_radii[i] <= 1 and 0 <= y - new_radii[i] and y + new_radii[i] <= 1):
                    valid = False
                    break
            if valid:
                radii = new_radii
                sum_radii = float(radii.sum())
        except:
            pass

    return centers, radii, sum_radii
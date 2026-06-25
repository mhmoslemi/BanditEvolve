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

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Final cleanup pass to attempt infinitesimal radius inflation
    def cleanup(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Try to increase radius slightly if possible
            for dx in [0, 1e-5]:
                for dy in [0, 1e-5]:
                    new_x = x + dx
                    new_y = y + dy
                    # Check if new position would cause overlap
                    overlap = False
                    for j in range(n):
                        if i == j:
                            continue
                        dx_j = new_x - centers[j][0]
                        dy_j = new_y - centers[j][1]
                        dist_sq = dx_j*dx_j + dy_j*dy_j
                        r_sum = r + radii[j]
                        if dist_sq < r_sum*r_sum - 1e-8:
                            overlap = True
                            break
                    if not overlap:
                        # Check if new position is within the square
                        if new_x - r < 0 or new_x + r > 1 or new_y - r < 0 or new_y + r > 1:
                            continue
                        # If no overlap, try to increase radius
                        new_r = r + 1e-5
                        if new_r > 0.5:
                            continue
                        # Check if new radius would cause overlap
                        overlap = False
                        for j in range(n):
                            if i == j:
                                continue
                            dx_j = new_x - centers[j][0]
                            dy_j = new_y - centers[j][1]
                            dist_sq = dx_j*dx_j + dy_j*dy_j
                            r_sum = new_r + radii[j]
                            if dist_sq < r_sum*r_sum - 1e-8:
                                overlap = True
                                break
                        if not overlap:
                            # Update radius
                            v[3*i+2] = new_r
        return v

    # Attempt cleanup
    try:
        v = cleanup(v)
    except:
        pass

    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    # Use a hexagonal grid seeding pattern for better initial placement
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii_grid = np.linspace(0.2, 0.5, cols)
    rows = np.ceil(n / cols).astype(int)
    centers = []
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx >= n:
                break
            r = radii_grid[j] if j < len(radii_grid) else 0.2
            angle = angles[idx]
            x = 0.5 + r * np.cos(angle)
            y = 0.5 + r * np.sin(angle)
            centers.append([x, y])
    # Refine initial positions for better packing
    refined_centers = np.array(centers)
    refined_centers[:, 0] = np.clip(refined_centers[:, 0], 0.0, 1.0)
    refined_centers[:, 1] = np.clip(refined_centers[:, 1], 0.0, 1.0)
    v0 = np.empty(3 * n)
    v0[0::3] = refined_centers[:, 0]
    v0[1::3] = refined_centers[:, 1]
    v0[2::3] = np.full(n, 0.1)

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
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-8})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
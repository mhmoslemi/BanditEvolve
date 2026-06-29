import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using Voronoi tessellation-inspired placement with random perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Introduce variation to break symmetry and allow better expansion
        if row % 3 == 1:
            x += np.random.uniform(-0.05, 0.05) / cols
        if row % 2 == 1:
            y += np.random.uniform(-0.05, 0.05) / rows
        xs.append(x)
        ys.append(y)
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized overlap constraints
    def create_overlap_constraints():
        cons = []
        for i in range(n):
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Vectorized overlap constraints
        x_centers = v[0::3]
        y_centers = v[1::3]
        r = v[2::3]
        # Compute all pairwise distances squared and compare with sum of radii squared
        dist_sq = np.zeros((n, n))
        for i in range(n):
            dx = x_centers[i] - x_centers
            dy = y_centers[i] - y_centers
            dist_sq[i] = dx*dx + dy*dy
        for i in range(n):
            for j in range(i+1, n):
                cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: dist_sq[i,j] - (v[3*i+2] + v[3*j+2])**2})
        return cons

    cons = create_overlap_constraints()

    # Global optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Local refinement step: perturb the most isolated circle
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dists[i] += np.sqrt(dx*dx + dy*dy)
        isolated_index = np.argmin(dists)
        v[3*isolated_index + 2] += 0.002
        v[3*isolated_index + 0] += 0.005
        v[3*isolated_index + 1] += 0.005
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
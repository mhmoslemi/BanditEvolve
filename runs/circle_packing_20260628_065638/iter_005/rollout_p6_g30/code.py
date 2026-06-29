import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with a hexagonal grid and perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Introduce perturbation to break symmetry and allow better expansion
        if row % 2 == 1:
            x += 0.5 / cols
        x += np.random.uniform(-0.02, 0.02)
        y += np.random.uniform(-0.02, 0.02)
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

    # Vectorized constraint setup for overlap conditions
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Precompute all pairwise distances for vectorized constraint evaluation
    positions = v0[0::3] + 1j * v0[1::3]
    radii = v0[2::3]
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i] - positions[j]
            dist_sq = np.abs(dx) ** 2
            min_dist_sq = (radii[i] + radii[j]) ** 2
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: dist_sq - min_dist_sq})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Local refinement: perturb the most isolated circle
    if res.success:
        v = res.x
        centers = v[0::3] + 1j * v[1::3]
        radii = v[2::3]
        dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dists[i] += np.abs(centers[i] - centers[j])
        isolated_index = np.argmin(dists)
        v[3*isolated_index + 2] += 0.002
        v[3*isolated_index + 0] += 0.005
        v[3*isolated_index + 1] += 0.005
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    # Additional refinement: shake the smallest circles
    if res.success:
        v = res.x
        radii = v[2::3]
        small_indices = np.where(radii < np.median(radii))[0]
        for idx in small_indices:
            v[3*idx + 2] += 0.001
            v[3*idx + 0] += 0.002
            v[3*idx + 1] += 0.002
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 200, "ftol": 1e-10})
            v = res.x

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
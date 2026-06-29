import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    
    # Assign clusters to promote spatial diversity
    clusters = np.random.randint(0, 5, size=n)
    cluster_centers = np.random.rand(5, 2)
    cluster_centers = np.column_stack((cluster_centers[:, 0] * 0.8 + 0.1, 
                                       cluster_centers[:, 1] * 0.8 + 0.1))
    
    # Assign positions within clusters
    xs = np.array([cluster_centers[c, 0] + np.random.uniform(-0.1, 0.1) for c in clusters])
    ys = np.array([cluster_centers[c, 1] + np.random.uniform(-0.1, 0.1) for c in clusters])
    
    # Ensure positions are within the unit square
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

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

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Non-local reconfiguration: identify isolated cluster and expand radii
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Identify isolated cluster (least dense)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
                dists[j, i] = dists[i, j]
        
        cluster_dists = np.sum(dists, axis=1)
        isolated_idx = np.argsort(cluster_dists)[0]
        
        # Expand radii of isolated cluster by 10%
        perturbed_v = v.copy()
        perturbed_v[3*isolated_idx+2] = np.clip(
            radii[isolated_idx] * 1.1, 1e-4, 0.5
        )
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using randomized geometric clustering
    xs = []
    ys = []
    cluster_centers = np.random.rand(4, 2) * 0.8 + 0.1  # 4 clusters
    for i in range(n):
        # Assign to a cluster
        cluster = np.argmin(np.sum((np.array([i//cols, i%cols]) - cluster_centers) ** 2, axis=1))
        # Generate a random point within the cluster
        offset = np.random.rand(2) * 0.4 - 0.2
        x = cluster_centers[cluster, 0] + offset[0]
        y = cluster_centers[cluster, 1] + offset[1]
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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Radical reconfiguration: trigger a new cluster-based arrangement
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Identify the cluster with the tightest packing
        cluster_assignments = np.array([i // cols for i in range(n)])
        cluster_radii = np.zeros(4)
        for i in range(4):
            cluster_radii[i] = np.mean(radii[cluster_assignments == i])
        tightest_cluster = np.argmin(cluster_radii)
        # Expand radii in the tightest cluster and perturb positions
        v[2::3][cluster_assignments == tightest_cluster] += 0.005
        v[0::3][cluster_assignments == tightest_cluster] += np.random.uniform(-0.01, 0.01, size=n//4)
        v[1::3][cluster_assignments == tightest_cluster] += np.random.uniform(-0.01, 0.01, size=n//4)
        # Reoptimize
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
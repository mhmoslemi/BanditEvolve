import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    centers = np.random.rand(n, 2)
    # Ensure initial positions are spread out
    for i in range(n):
        centers[i] += np.random.uniform(-0.2, 0.2, 2)
        centers[i] = np.clip(centers[i], 0.0, 1.0)
    # Group centers into clusters to form a more flexible configuration
    cluster_centers = np.random.rand(4, 2)
    cluster_centers = np.clip(cluster_centers, 0.0, 1.0)
    cluster_sizes = [6, 7, 7, 6]
    centers = []
    for i in range(4):
        for _ in range(cluster_sizes[i]):
            perturbation = 0.05 * np.random.rand(2)
            new_center = cluster_centers[i] + perturbation
            new_center = np.clip(new_center, 0.0, 1.0)
            centers.append(new_center)
    centers = np.array(centers)
    # Assign initial radii
    r0 = 0.15
    v0 = np.empty(3 * n)
    v0[0::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
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
    
    # Controlled radius expansion on the most isolated cluster
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Identify the cluster with the largest minimum distance to others
        distances = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    distances[i] += np.sqrt(dx*dx + dy*dy)
        cluster_distances = np.array([np.mean(distances[i*6:(i+1)*6]) for i in range(4)])
        cluster_index = np.argmax(cluster_distances)
        # Increase radii of the most isolated cluster
        for i in range(4):
            if i == cluster_index:
                r_multiplier = 1.2
            else:
                r_multiplier = 1.0
            for j in range(cluster_sizes[i]):
                idx = sum(cluster_sizes[:i]) + j
                v[3*idx+2] *= r_multiplier
                v[3*idx+2] = np.clip(v[3*idx+2], 1e-4, 0.5)
        # Re-evaluate with adjusted radii
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
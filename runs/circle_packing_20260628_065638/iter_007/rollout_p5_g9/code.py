import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    radii = np.full(n, 0.05)
    
    # Cluster the points in a more compact region to enable tighter packing
    cluster_radius = 0.15
    cluster_centers = np.random.rand(5, 2)
    cluster_centers /= np.sqrt(np.sum(cluster_centers**2, axis=1, keepdims=True)) * 2
    cluster_centers *= 0.65
    
    for i in range(n):
        # Assign each circle to the closest cluster center
        distances = np.sum((xs[i] - cluster_centers[:, 0])**2 + (ys[i] - cluster_centers[:, 1])**2, axis=1)
        closest_cluster = np.argmin(distances)
        # Place the circle near the cluster center with some jitter
        xs[i] = cluster_centers[closest_cluster, 0] + np.random.uniform(-0.05, 0.05)
        ys[i] = cluster_centers[closest_cluster, 1] + np.random.uniform(-0.05, 0.05)
    
    # Ensure circles are within bounds and adjust radii based on cluster density
    r0 = 0.3 / cols - 1e-3
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

    # Add a 'shake' heuristic: perturb the smallest circles
    if res.success:
        v = res.x
        radii = v[2::3]
        smallest_indices = np.argsort(radii)[:4]  # Perturb the 4 smallest circles
        for idx in smallest_indices:
            v[3*idx + 0] += np.random.uniform(-0.01, 0.01)
            v[3*idx + 1] += np.random.uniform(-0.01, 0.01)
            v[3*idx + 2] += np.random.uniform(-0.001, 0.001)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10})

    # Additional refinement: perturb all circles with small radii
    if res.success:
        v = res.x
        radii = v[2::3]
        small_indices = np.where(radii < 0.05)[0]  # Perturb circles with radii < 0.05
        for idx in small_indices:
            v[3*idx + 0] += np.random.uniform(-0.01, 0.01)
            v[3*idx + 1] += np.random.uniform(-0.01, 0.01)
            v[3*idx + 2] += np.random.uniform(-0.001, 0.001)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
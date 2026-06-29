import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    # Generate random cluster centers and distribute circles within clusters
    np.random.seed(42)
    cluster_centers = np.random.rand(4, 2) * 0.8 + 0.1  # 4 clusters spread in the square
    cluster_sizes = np.random.randint(5, 10, size=4)  # Distribute 26 circles among 4 clusters
    cluster_sizes = np.append(cluster_sizes, 26 - np.sum(cluster_sizes))  # Ensure total of 26
    
    xs = []
    ys = []
    for i in range(4):
        for _ in range(cluster_sizes[i]):
            # Randomly place circles within the cluster area
            x = np.random.uniform(cluster_centers[i, 0] - 0.1, cluster_centers[i, 0] + 0.1)
            y = np.random.uniform(cluster_centers[i, 1] - 0.1, cluster_centers[i, 1] + 0.1)
            xs.append(x)
            ys.append(y)
    
    r0 = 0.15  # Start with a more generous initial radius
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

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Select the cluster with the most tightly packed circles for targeted expansion
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        cluster_radii = []
        for i in range(4):
            cluster_idx = np.arange(4*i, 4*(i+1))
            dists = np.zeros(len(cluster_idx))
            for idx in cluster_idx:
                for j in cluster_idx:
                    if idx != j:
                        dx = centers[0][idx] - centers[0][j]
                        dy = centers[1][idx] - centers[1][j]
                        dists[idx] += np.sqrt(dx*dx + dy*dy)
            avg_dist = np.mean(dists)
            cluster_radii.append(avg_dist / np.mean(radii[cluster_idx]))
        
        # Expand the cluster with the smallest average distance (most tightly packed)
        cluster_idx = np.argmin(cluster_radii)
        cluster_start = cluster_idx * 4
        cluster_end = (cluster_idx + 1) * 4
        cluster_radii = radii[cluster_start:cluster_end]
        cluster_centers = centers[0][cluster_start:cluster_end], centers[1][cluster_start:cluster_end]
        
        # Increase radius by 0.01 for all circles in the selected cluster
        radii[cluster_start:cluster_end] += 0.01
        # Adjust positions to accommodate the increase in radius
        for idx in range(cluster_start, cluster_end):
            x, y = centers[0][idx], centers[1][idx]
            r = radii[idx]
            # Move circle slightly to avoid overlap
            dx = np.random.uniform(-0.01, 0.01)
            dy = np.random.uniform(-0.01, 0.01)
            centers[0][idx] += dx
            centers[1][idx] += dy
        
        # Reconstruct the decision vector with updated positions and radii
        v = np.zeros(3 * n)
        v[0::3] = centers[0]
        v[1::3] = centers[1]
        v[2::3] = radii
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    np.random.seed(42)
    # Create random points in the unit square
    points = np.random.rand(n, 2)
    # Apply k-means clustering to form groups
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=cols, random_state=42).fit(points)
    # Get cluster centers
    clusters = kmeans.cluster_centers_
    # Create initial positions by distributing points in each cluster
    xs = []
    ys = []
    for i in range(cols):
        cluster_points = points[kmeans.labels_ == i]
        # Spread points within the cluster area
        spread = 0.1
        cluster_points += np.random.uniform(-spread, spread, cluster_points.shape)
        # Add to the list
        xs.extend(cluster_points[:, 0])
        ys.extend(cluster_points[:, 1])
    
    # Ensure we have exactly 26 points
    if len(xs) > n:
        xs = xs[:n]
        ys = ys[:n]
    
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
    def vectorized_overlap(v):
        radii = v[2::3]
        x = v[0::3]
        y = v[1::3]
        dist_sq = np.zeros((n, n))
        for i in range(n):
            dx = x - x[i]
            dy = y - y[i]
            dist_sq[:, i] = dx*dx + dy*dy
        return dist_sq - (radii[:, np.newaxis] + radii[np.newaxis, :]) ** 2

    # Create constraints for all pairs using vectorized calculation
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Create constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: vectorized_overlap(v)[i, j]})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Non-local reconfiguration: find the cluster with the smallest radius and expand
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the cluster with the smallest radius
        cluster_radii = np.array_split(radii, cols)
        min_cluster_idx = np.argmin([np.mean(cluster) for cluster in cluster_radii])
        # Get indices of the cluster
        cluster_indices = np.where(np.array_split(np.arange(n), cols)[min_cluster_idx])[0]
        # Increase radii of this cluster by a small amount
        delta = 0.005
        perturbed_v = v.copy()
        for i in cluster_indices:
            perturbed_v[3*i+2] += delta
            # Clip to ensure radii do not exceed the bounds
            perturbed_v[3*i+2] = np.clip(perturbed_v[3*i+2], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
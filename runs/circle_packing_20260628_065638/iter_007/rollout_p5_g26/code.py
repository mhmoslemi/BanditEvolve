import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Introduce random offset to break symmetry and allow better expansion
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
        if row % 2 == 1:
            x += 0.5 / cols
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

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Radical reconfiguration: cluster-based radius expansion
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        
        # Group circles into clusters using Voronoi tessellation
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[0][i] - centers[0][j]
                dy = centers[1][i] - centers[1][j]
                distances[i][j] = np.sqrt(dx*dx + dy*dy)
        
        # Compute Voronoi regions using nearest neighbors
        clusters = [[] for _ in range(5)]  # Assume 5 clusters for 26 circles
        for i in range(n):
            nearest = np.argsort(distances[i])[1:6]
            cluster_idx = np.argmin(np.array([np.min(distances[j]) for j in nearest]))
            clusters[cluster_idx].append(i)
        
        # Expand radii of the most tightly packed cluster
        for cluster in clusters:
            if cluster:
                cluster_radii = radii[cluster]
                cluster_centers = np.stack([centers[0][i], centers[1][i]] for i in cluster)
                avg_radius = np.mean(cluster_radii)
                avg_center = np.mean(cluster_centers, axis=0)
                
                # Increase radii of the cluster proportionally
                for i in cluster:
                    v[3*i + 2] += avg_radius * 0.2  # 20% increase for cluster
                    v[3*i + 0] += np.random.uniform(-0.01, 0.01)
                    v[3*i + 1] += np.random.uniform(-0.01, 0.01)
        
        # Re-evaluate with modified parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using randomized geometric clustering
    centers = np.random.rand(n, 2)
    radii = np.full(n, 0.05)
    
    # Randomly select a subset to form a cluster
    cluster_indices = np.random.choice(n, size=8, replace=False)
    cluster_center = np.array([0.5, 0.5])
    cluster_radius = 0.2
    for i in cluster_indices:
        # Place cluster members closer to the center
        dx = np.random.uniform(-0.15, 0.15)
        dy = np.random.uniform(-0.15, 0.15)
        centers[i] = cluster_center + np.array([dx, dy])
        # Ensure the cluster members are within the square
        centers[i] = np.clip(centers[i], [0.0, 0.0], [1.0, 1.0])
        # Set a larger radius for cluster members
        radii[i] = cluster_radius - np.random.uniform(0.01, 0.05)
    
    # Initialize decision vector
    v0 = np.empty(3 * n)
    v0[0::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
    v0[2::3] = radii

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
    
    # Apply controlled radius expansion to the most isolated cluster
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Identify the most isolated cluster
        distances = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    distances[i] += np.sqrt(dx*dx + dy*dy)
        cluster_indices = np.argsort(distances)[-8:]
        
        # Expand radii of the most isolated cluster by 5%
        expansion = 0.05
        for i in cluster_indices:
            new_radius = min(radii[i] * (1 + expansion), 0.5)
            if new_radius > radii[i]:
                # Perturb position slightly to maintain non-overlap
                perturbation = 0.02 * (np.random.rand(2) - 0.5)
                centers[i] += perturbation
                centers[i] = np.clip(centers[i], [0.0, 0.0], [1.0, 1.0])
                radii[i] = new_radius
        
        # Reconstruct decision vector
        v = np.empty(3 * n)
        v[0::3] = centers[:, 0]
        v[1::3] = centers[:, 1]
        v[2::3] = radii
        
        # Re-evaluate with expanded parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
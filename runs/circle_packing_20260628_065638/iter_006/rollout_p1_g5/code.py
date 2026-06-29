import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    # Generate initial cluster centers with controlled spacing
    cluster_centers = np.random.rand(4, 2)
    cluster_centers = (cluster_centers - 0.5) * 0.8 + 0.5  # Center and scale
    cluster_radii = np.random.rand(4) * 0.05 + 0.01
    
    # Generate initial positions by placing points within clusters
    xs = []
    ys = []
    for i in range(n):
        cluster_idx = i % 4
        cluster_center = cluster_centers[cluster_idx]
        cluster_radius = cluster_radii[cluster_idx]
        # Generate random point within cluster
        angle = np.random.rand() * 2 * np.pi
        radius = np.random.rand() * cluster_radius
        x = cluster_center[0] + radius * np.cos(angle)
        y = cluster_center[1] + radius * np.sin(angle)
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
    
    # Targeted expansion of the most isolated cluster
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Calculate distances from each circle to all others
        dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dists[i] += dx*dx + dy*dy
            dists[i] = np.sqrt(dists[i] / (n-1))
        
        # Identify the most isolated cluster (largest distance)
        isolated_idx = np.argmax(dists)
        # Increase radius of the most isolated circle by 10%
        perturbed_v = v.copy()
        perturbed_v[3*isolated_idx+2] = np.clip(radii[isolated_idx] * 1.1, 1e-4, 0.5)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
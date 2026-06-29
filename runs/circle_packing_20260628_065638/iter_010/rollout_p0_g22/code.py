import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering with controlled spatial distribution
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Add controlled random offset for spatial diversity
        xs[i] = base_x + np.random.uniform(-0.03, 0.03)
        ys[i] = base_y + np.random.uniform(-0.03, 0.03)
        # Alternate row staggering for asymmetry
        if row % 2 == 1:
            xs[i] += 0.5 / cols
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
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
    
    # Trigger radical reconfiguration with randomized geometric clustering
    if res.success:
        v = res.x
        # Generate new randomized clustered positions
        new_xs = np.zeros(n)
        new_ys = np.zeros(n)
        for i in range(n):
            row = i // cols
            col = i % cols
            base_x = (col + 0.5) / cols
            base_y = (row + 0.5) / rows
            new_xs[i] = base_x + np.random.uniform(-0.03, 0.03)
            new_ys[i] = base_y + np.random.uniform(-0.03, 0.03)
            if row % 2 == 1:
                new_xs[i] += 0.5 / cols
        # Keep current radii and reoptimize
        new_v = v.copy()
        new_v[0::3] = new_xs
        new_v[1::3] = new_ys
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion for most tightly packed cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate packing density for each cluster
        cluster_indices = np.arange(n)
        cluster_ids = np.zeros(n, dtype=int)
        for i in range(n):
            cluster_ids[i] = i // 5  # Group circles into clusters
        cluster_densities = np.zeros(n)
        for cluster_id in np.unique(cluster_ids):
            mask = cluster_ids == cluster_id
            cluster_radii = radii[mask]
            cluster_centers = np.column_stack([centers[0][mask], centers[1][mask]])
            # Calculate total area occupied by cluster
            total_area = np.sum(np.pi * cluster_radii**2)
            # Calculate bounding box area for cluster
            min_x, max_x = cluster_centers[:, 0].min(), cluster_centers[:, 0].max()
            min_y, max_y = cluster_centers[:, 1].min(), cluster_centers[:, 1].max()
            bounding_box_area = (max_x - min_x) * (max_y - min_y)
            # Compute density as ratio of area to bounding box
            cluster_densities[mask] = total_area / bounding_box_area
        # Identify cluster with highest density
        densest_cluster_mask = cluster_ids == np.unique(cluster_ids)[np.argmax(cluster_densities)]
        # Expand radii of circles in densest cluster
        for i in range(n):
            if densest_cluster_mask[i]:
                v[3*i + 2] += 0.003
                v[3*i] += 0.005
                v[3*i + 1] += 0.005
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
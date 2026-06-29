import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering with controlled spacing
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    radii = np.random.rand(n) * 0.1 + 0.05
    # Normalize and adjust positions to form a grid-like structure
    xs = (xs * cols - 0.5) / cols + 0.5
    ys = (ys * rows - 0.5) / rows + 0.5
    # Add random jitter to break symmetry
    xs += np.random.uniform(-0.05, 0.05, n)
    ys += np.random.uniform(-0.05, 0.05, n)
    # Alternate row staggering for asymmetry
    for i in range(n):
        if (i // cols) % 2 == 1:
            xs[i] += 0.5 / cols
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = radii

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
    
    # Radical reconfiguration: randomized geometric clustering
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Randomized geometric clustering with controlled spacing
        xs = np.random.rand(n)
        ys = np.random.rand(n)
        radii_new = np.random.rand(n) * 0.1 + 0.05
        # Normalize and adjust positions to form a grid-like structure
        xs = (xs * cols - 0.5) / cols + 0.5
        ys = (ys * rows - 0.5) / rows + 0.5
        # Add random jitter to break symmetry
        xs += np.random.uniform(-0.05, 0.05, n)
        ys += np.random.uniform(-0.05, 0.05, n)
        # Alternate row staggering for asymmetry
        for i in range(n):
            if (i // cols) % 2 == 1:
                xs[i] += 0.5 / cols
        # Create new initial guess
        v_new = np.empty(3 * n)
        v_new[0::3] = xs
        v_new[1::3] = ys
        v_new[2::3] = radii_new
        # Re-evaluate with new parameters
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion: expand the most tightly packed cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate pair-wise distances and identify clusters
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[0][i] - centers[0][j]
                dy = centers[1][i] - centers[1][j]
                distances[i, j] = np.sqrt(dx*dx + dy*dy)
        # Identify clusters using connectivity
        cluster_labels = np.zeros(n, dtype=int)
        current_label = 0
        for i in range(n):
            if cluster_labels[i] == 0:
                current_label += 1
                cluster_labels[i] = current_label
                for j in range(n):
                    if distances[i, j] < 0.3 * (radii[i] + radii[j]):
                        if cluster_labels[j] == 0:
                            cluster_labels[j] = current_label
        # Find the largest cluster
        cluster_sizes = np.bincount(cluster_labels)
        largest_cluster = np.argmax(cluster_sizes)
        # Identify indices in the largest cluster
        cluster_indices = np.where(cluster_labels == largest_cluster)[0]
        # Expand the radii of the largest cluster
        for i in cluster_indices:
            v[3*i + 2] += 0.005
            v[3*i] += 0.005
            v[3*i + 1] += 0.005
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    
    # Randomized geometric clustering initialization
    def cluster_positions():
        # Create a grid of cluster centers
        cols = 5
        rows = (n + cols - 1) // cols
        centers = np.zeros((n, 2))
        for i in range(n):
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            # Add small random offset for diversification
            x += np.random.uniform(-0.05, 0.05)
            y += np.random.uniform(-0.05, 0.05)
            if row % 2 == 1:
                x += 0.5 / cols
            centers[i] = [x, y]
        return centers
    
    # Initialize with clustered positions
    centers = cluster_positions()
    # Randomly select 5 circles to be "anchors" that will be less perturbed
    anchor_indices = np.random.choice(n, size=5, replace=False)
    # Assign initial radii based on cluster spacing
    dists = np.zeros(n)
    for i in range(n):
        dists[i] = np.min(np.linalg.norm(centers[i] - centers[j], axis=1) for j in range(n) if j != i)
    r0 = np.clip(dists.mean() / 2.5, 0.01, 0.4)
    radii = np.full(n, r0)
    
    # Decision vector v = [x0,y0,r0, x1,y1,r1, ...], length 3*n
    v0 = np.empty(3 * n)
    v0[0::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
    v0[2::3] = radii
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]   # length 3*n, matches v
    
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
    
    # Apply controlled radius expansion to the most isolated cluster
    if res.success:
        v = res.x
        # Find the cluster with maximum minimal distance to others
        current_centers = v[0::3], v[1::3]
        dists = np.zeros(n)
        for i in range(n):
            row_dist = np.linalg.norm(current_centers[0][i] - current_centers[0], axis=1)
            col_dist = np.linalg.norm(current_centers[1][i] - current_centers[1], axis=1)
            dists[i] = np.min(np.minimum(row_dist, col_dist))
        isolated_index = np.argmax(dists)
        
        # Increase radius of the isolated cluster slightly
        perturbation = 0.01 * np.random.rand()
        perturbed_v = v.copy()
        perturbed_v[3*isolated_index+2] += perturbation
        perturbed_v[3*isolated_index+2] = np.clip(perturbed_v[3*isolated_index+2], 1e-4, 0.5)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12})
    
    # Asymmetric reconfiguration with geometric hashing
    if res.success:
        v = res.x
        # Hash positions to groups
        grid_size = 0.2
        hash_grid = np.zeros((int(1/grid_size), int(1/grid_size)), dtype=int)
        for i in range(n):
            x = v[3*i]
            y = v[3*i+1]
            gx = int(x / grid_size)
            gy = int(y / grid_size)
            hash_grid[gx, gy] += 1
        
        # Find clusters with high density
        cluster_indices = []
        for gx in range(hash_grid.shape[0]):
            for gy in range(hash_grid.shape[1]):
                if hash_grid[gx, gy] > 2:
                    for i in range(n):
                        x = v[3*i]
                        y = v[3*i+1]
                        if gx - 1 <= int(x / grid_size) <= gx + 1 and gy - 1 <= int(y / grid_size) <= gy + 1:
                            cluster_indices.append(i)
        
        # Random perturbation to clusters
        if cluster_indices:
            cluster_indices = np.unique(cluster_indices)
            perturbation = np.random.rand(len(cluster_indices) * 3) * 0.05
            perturbed_v = v.copy()
            idx = 0
            for i in cluster_indices:
                perturbed_v[3*i] += perturbation[idx]
                perturbed_v[3*i+1] += perturbation[idx+1]
                perturbed_v[3*i+2] += perturbation[idx+2]
                idx += 3
            # Re-evaluate with perturbed parameters
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion with non-overlap enforcement
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Find smallest radius
        min_radius_idx = np.argmin(radii)
        # Expand radius while enforcing non-overlap
        for _ in range(10):
            # Expand radius slightly
            new_radius = radii[min_radius_idx] + 0.002
            # Check if expansion is feasible
            feasible = True
            for j in range(n):
                if j == min_radius_idx:
                    continue
                dx = centers[0][min_radius_idx] - centers[0][j]
                dy = centers[1][min_radius_idx] - centers[1][j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_radius + radii[j] - 1e-8:
                    feasible = False
                    break
            if feasible:
                # Adjust position to allow expansion
                for j in range(n):
                    if j == min_radius_idx:
                        continue
                    dx = centers[0][min_radius_idx] - centers[0][j]
                    dy = centers[1][min_radius_idx] - centers[1][j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    overlap = new_radius + radii[j] - dist
                    if overlap > 1e-8:
                        # Move the expanded circle away
                        angle = np.arctan2(dy, dx)
                        move_dist = overlap / 2
                        move_x = move_dist * np.cos(angle)
                        move_y = move_dist * np.sin(angle)
                        v[3*min_radius_idx] += move_x
                        v[3*min_radius_idx+1] += move_y
                # Update radius
                v[3*min_radius_idx+2] = new_radius
                # Re-evaluate with adjusted parameters
                res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
                break
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
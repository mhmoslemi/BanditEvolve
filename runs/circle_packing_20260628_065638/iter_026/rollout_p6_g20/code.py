import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering, staggered grid, and
    # probabilistic geometric hashing to break symmetry and enable reconfiguration
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
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
    
    # Vectorized overlap constraints with geometric hashing and directional constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-11})
    
    # Disruptive geometric transformation
    if res.success:
        v = res.x
        # Vectorized distance calculation using broadcasting
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        radii = v[2::3]
        adj = dists <= (radii + radii.reshape(-1, 1))
        
        # Apply geometric hashing to the spatial domain for reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        
        # Asymmetric radius expansion with directional enforcement for least constrained circle
        if res.success:
            v = res.x
            radii = v[2::3]
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmin(min_dists)  # Find least constrained with minimal min distance
            smallest_radius_idx = np.argmin(radii)  # Find smallest radius
            
            # Compute current total sum and expansion target
            total_sum = np.sum(radii)
            target_total_sum = total_sum + 0.0085
            expansion = (target_total_sum - total_sum) / (n - 1)
            
            # Apply targeted expansion to minimal constraint circle and its neighbors
            new_radii = radii.copy()
            for i in range(n):
                if (i == least_constrained_idx or i == smallest_radius_idx) or (i in np.argsort(dists[i, :] <= dists[i, :].max())[0:3]):
                    new_radii[i] += expansion * 1.2  # Over-expand for reordering
                else:
                    new_radii[i] += expansion
            
            # Create new decision vector and re-evaluate with new constraints
            v_new = v.copy()
            v_new[2::3] = new_radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final optimization with tight tolerances and dynamic constraint enforcement
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Update min_dists and least constrained circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        
        # Final targeted expansion to enforce spatial reordering
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.0085
        expansion = (target_total_sum - total_sum) / (n - 1)
        
        new_radii = radii.copy()
        for i in range(n):
            if i == least_constrained_idx:
                new_radii[i] += expansion * 1.2
            else:
                new_radii[i] += expansion
        
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
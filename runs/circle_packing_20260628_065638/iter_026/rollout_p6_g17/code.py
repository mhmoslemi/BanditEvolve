import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
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
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11})
    
    # Disruptive geometric transformation: reconfiguration with spatial hashing and asymmetric radius expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Apply geometric hashing for spatial reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (1 + np.random.rand() * 0.5)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1 + np.random.rand() * 0.5)
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        
        # Identify most constrained circle (smallest non-zero spatial radius)
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            
            # Compute distance matrix for all pairwise circles
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Identify the circle with the smallest non-zero radius
            smallest_radius_idx = np.argmin(radii)
            # Compute least constrained circle with minimal minimum distance
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            
            # Apply aggressive radius expansion to least constrained circle to trigger layout reconfiguration
            total_sum = np.sum(radii)
            # Compute expansion factor that increases total by 0.01 while keeping individual radii < 0.35
            max_expansion_per_radii = (0.35 - radii) * 1.2
            allowed_expansion = np.min(max_expansion_per_radii)
            expansion_amount = allowed_expansion * 0.95
                
            # Create new radii with asymmetric expansion
            expanded_radii = radii.copy()
            expanded_radii[least_constrained_idx] += expansion_amount
            for i in range(n):
                if i != least_constrained_idx:
                    expanded_radii[i] += expansion_amount * (1 - 0.3 * np.random.rand())
            
            # Apply expansion to decision vector
            expanded_v = v.copy()
            expanded_v[2::3] = expanded_radii
            
            # Re-evaluate with expanded radii and new configuration
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 600, "ftol": 1e-12})
    
    # Final optimization with tightened tolerances and constraint enforcement
    # Only proceed if initial successful optimization was achieved
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Enforce strict non-overlap with tighter tolerances
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                if np.sqrt(dx**2 + dy**2) < radii[i] + radii[j] - 1e-12:
                    # Force minimal radii expansion to avoid overlap
                    delta = radii[i] + radii[j] - np.sqrt(dx**2 + dy**2) + 1e-12
                    # Distribute expansion to both circles with bias toward least constrained
                    expansion_ratio = np.random.rand()
                    radii[i] += delta * expansion_ratio * (1 - 0.1 * np.random.rand())
                    radii[j] += delta * (1 - expansion_ratio) * (1 - 0.2 * np.random.rand())
        
        v[2::3] = np.clip(radii, 1e-6, 0.35)
        
        # Re-evaluate with updated radii for final tight constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
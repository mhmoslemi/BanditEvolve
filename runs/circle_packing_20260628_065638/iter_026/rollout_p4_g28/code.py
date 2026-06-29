import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric hashing
    seed = np.random.randint(0, 1000000)
    np.random.seed(seed)
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Geometric hash for spatial scrambling
        x = x_center + np.random.uniform(-1.0/cols, 1.0/cols)
        y = y_center + np.random.uniform(-1.0/rows, 1.0/rows)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.25 / cols - 1e-3
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
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
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "maxls": 100})

    # Disruptive geometric transformation: Target the smallest radius and expand
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Validate the current configuration
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-12 or x + r > 1 + 1e-12 \
               or y - r < -1e-12 or y + r > 1 + 1e-12:
                res.success = False
                break
        
        # Ensure minimal radius has non-zero value
        min_radius_idx = np.argmin(radii)
        if radii[min_radius_idx] < 1e-6:
            res.success = False

        if res.success:
            # Create a geometric hash to trigger spatial reconfiguration
            spatial_hash = np.random.rand(n, 2) * 0.02
            perturbed_v = v.copy()
            for i in range(n):
                perturbed_v[3*i] += spatial_hash[i, 0]
                perturbed_v[3*i+1] += spatial_hash[i, 1]
            
            # Re-evaluate with perturbed spatial configuration
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "maxls": 100})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find smallest radius and its neighbors for targeted expansion
        min_radius_idx = np.argmin(radii)
        neighbors = np.argsort(dists[min_radius_idx])[:5]
        target_neighbors = np.unique(neighbors)
        
        # Compute radius expansion with geometric constraints
        def target_expansion(v):
            # Calculate pairwise distances
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Compute total area expansion constraint
            expansion_target = 0.005
            expansion = np.mean(radii) + expansion_target / (n - 1)
            
            # Compute maximum allowable expansion
            max_exp = 0.0
            for i in range(n):
                if i == min_radius_idx:
                    max_exp = max(max_exp, 0.5 - radii[i])  # Boundary constraint
                else:
                    if i in target_neighbors:
                        max_exp = max(max_exp, max(0.2 - radii[i], expansion - radii[i]))
                    else:
                        max_exp = max(max_exp, 0.3 - radii[i])  # General constraint
            
            # Compute radius expansion vector
            new_radii = radii.copy()
            new_radii[min_radius_idx] += max(0.0, min(expansion - radii[min_radius_idx], max_exp))
            
            # Apply expansion to neighbors
            for i in target_neighbors:
                if i != min_radius_idx:
                    new_radii[i] += np.clip(max(0.0, expansion - radii[i]), -radii[i]*0.2, max_exp)
            
            return -np.sum(new_radii)
        
        # Apply radius expansion
        expansion_v = v.copy()
        expansion_v[2::3] = np.full(n, np.mean(radii) + 0.005 / (n - 1))
        
        res = minimize(target_expansion, expansion_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11, "maxls": 100})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
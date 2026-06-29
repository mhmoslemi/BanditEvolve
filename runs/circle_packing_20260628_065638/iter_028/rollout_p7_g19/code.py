import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized geometry and adaptive spacing for better initial distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce randomness and symmetry breaking with gradient bias
        x = x_center + np.random.uniform(-0.09, 0.09)
        y = y_center + np.random.uniform(-0.09, 0.09)
        if row % 2 == 1:  # staggered grid
            x += 0.35 / cols
        # Avoid center clustering by adjusting for row/column weight
        if abs(x) > 0.9 or abs(y) > 0.9:
            x *= 0.8
            y *= 0.8
        xs.append(x)
        ys.append(y)
    
    r0 = 0.4 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Optimized boundary constraints with lambda closure fixing
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Optimized overlap constraint using tuple unpacking and parameter caching
    for i in range(n):
        for j in range(i + 1, n):
            # Use tuple unpacking to prevent lambda capture issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Stage 1: Initial optimization with enhanced tolerances and geometry-aware initial guess
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-12, "gtol": 1e-12})
    
    # Stage 2: Post-optimization refinement with spatial hashing and radius-based perturbation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Spatial hashing for adaptive perturbation
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            if radii[i] > np.mean(radii) * 0.8:  # perturb larger circles more
                perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
                perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-12})
    
    # Stage 3: Geometric dissection on two most interacting circles
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute distance matrix with vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find top two interacting circles, using both distance and radial contribution
        interaction = np.sum(dists, axis=1) * np.sqrt(np.sum(radii))  # distance × size
        top_idx = np.argsort(interaction)[-2:]  # most interactive circles
        
        # Create new configuration for top two
        top_v = v.copy()
        for i in top_idx:
            # Introduce random offset with radius scaling
            top_v[3*i] += np.random.uniform(-0.04, 0.04) * (radii[i] / np.mean(radii))
            top_v[3*i+1] += np.random.uniform(-0.04, 0.04) * (radii[i] / np.mean(radii))
            # Slight radius increase to encourage better packing
            top_v[3*i+2] += np.random.uniform(-0.002, 0.002)
        
        # Re-evaluate with new top two configuration
        res = minimize(neg_sum_radii, top_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 350, "ftol": 1e-11, "gtol": 1e-12})
    
    # Stage 4: Targeted radius expansion on most isolated circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Find circle with largest min distance
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute isolation score as (sum of reciprocal distances)
        isolation_scores = np.zeros(n)
        for i in range(n):
            # avoid division by zero, take reciprocal of min distance
            d_min = np.min(dists[i, np.arange(n) != i]) + 1e-8
            isolation_scores[i] = 1.0 / d_min
        
        isolated_idx = np.argmax(isolation_scores)
        
        # Expand the isolated circle's radius
        current_total = np.sum(radii)
        # calculate possible expansion based on current configuration and isolation
        # use gradient descent for radius increment to maintain constraints
        expansion = 0.004
        while True:
            new_radii = radii.copy()
            new_radii[isolated_idx] += expansion
            
            # check all pairwise overlaps
            new_centers = np.column_stack([v[0::3], v[1::3]])
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist_sq = dx**2 + dy**2
                    r_sum = new_radii[i] + new_radii[j]
                    if dist_sq < r_sum**2 - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # back off expansion if invalid
                expansion *= 0.95
        
        v_new = v.copy()
        v_new[2::3] = new_radii
        # Re-evaluate with new radii configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-12})
    
    # Final check and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
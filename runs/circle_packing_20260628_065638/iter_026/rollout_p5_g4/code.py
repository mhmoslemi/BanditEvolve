import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    # Initialize with staggered grid and adaptive spacing with random perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply adaptive random offsets to break symmetry
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
        # Stagger even and odd rows
        if row % 2 == 1:
            x += 0.5 / cols
        # Ensure boundary clearance
        if row == rows - 1:
            y += np.random.uniform(-0.03, 0.03)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.42 / cols - 1e-3
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
        # Left edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter constraints and tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-12, "eps": 1e-11})

    # Asymmetric reconfiguration: trigger stochastic repositioning with adaptive hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hashing with dynamic scaling
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            # Adjust perturbation magnitude with radius
            perturb_factor = 0.05 * (1 + np.random.rand()) * (radii[i]/np.mean(radii))
            perturbed_v[3*i] += spatial_hash[i, 0] * perturb_factor
            perturbed_v[3*i+1] += spatial_hash[i, 1] * perturb_factor
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    # Targeted radial expansion with asymmetric constraint violation handling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Identify least constrained circles by min distance to cluster
        cluster_distances = np.sum(1.0 / (dists + 1e-3), axis=1)
        least_constrained_idx = np.argmin(cluster_distances)
        
        # Apply asymmetric radial expansion with constraint-aware scaling
        target_total = np.sum(radii) + 0.01  # Small but impactful increase
        expansion_amount = (target_total - np.sum(radii)) / (n - 1)
        
        # Create expansion vector with soft constraints
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_amount * 1.1
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_amount * (1.0 + 0.1 * np.random.rand())
        
        # Validate and refine expansion
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
        
        # Validate distances with constraint relaxation
        max_overlap = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < new_radii[i] + new_radii[j] - 1e-12:
                    overlap = new_radii[i] + new_radii[j] - dist
                    max_overlap = max(max_overlap, overlap)
        
        if max_overlap < 0.005:
            # If expansion is safe, proceed
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
        else:
            # If overexpanded, keep original result
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
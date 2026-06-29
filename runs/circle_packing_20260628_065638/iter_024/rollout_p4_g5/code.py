import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    grid_size = 1.0 / cols
    grid_spacing = grid_size * 0.85

    # Generate grid positions with randomized geometric tiling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = col * grid_spacing
        base_y = row * grid_spacing
        # Use hexagonal tiling pattern (staggered)
        if row % 2 == 1:
            base_x += grid_spacing / 2
        # Add small randomized offsets to avoid symmetry
        x = base_x + np.random.uniform(-grid_spacing * 0.1, grid_spacing * 0.1)
        y = base_y + np.random.uniform(-grid_spacing * 0.1, grid_spacing * 0.1)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.25 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n elements

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

    # Initial optimization with high precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-12})

    # Apply geometric hashing for spatial reconfiguration
    if res.success:
        v = res.x
        # Compute spatial hashes to perturb positions
        spatial_hash = np.random.rand(n, 2) * grid_spacing * 0.3
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_hash[i, 0]
            new_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-12})

    # Targeted radius expansion using gradient-based method
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Compute minimum distance to all other circles
        dists = np.zeros((n, n))
        for i in range(n):
            dx = centers[:, 0] - centers[i, 0]
            dy = centers[:, 1] - centers[i, 1]
            dists[i, :] = np.sqrt(dx**2 + dy**2)

        # Identify the circle with the minimal effective constraint
        effective_constraints = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(effective_constraints)

        # Calculate max possible expansion based on current configuration
        max_expansion = 0.0
        for j in range(n):
            if j != least_constrained_idx:
                dx = centers[least_constrained_idx, 0] - centers[j, 0]
                dy = centers[least_constrained_idx, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                max_expansion = max(max_expansion, dist - radii[least_constrained_idx] - radii[j])

        # Expand the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += max_expansion * 0.85
        for j in range(n):
            if j != least_constrained_idx:
                dx = centers[least_constrained_idx, 0] - centers[j, 0]
                dy = centers[least_constrained_idx, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                # Ensure no overlap with nearby circles
                if dist < radii[least_constrained_idx] + radii[j] + 1e-10:
                    new_radii[least_constrained_idx] = dist - radii[j] - 1e-10

        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii

        # Re-evaluate with updated radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
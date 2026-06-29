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

    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using broadcasting for efficiency
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-10})

    if res.success:
        # Trigger a disruptive geometric transformation
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Create geometric hash for spatial hashing
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        
        # Add hash-based perturbation with asymmetric scaling
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (1 + np.random.rand() * 0.3)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1 + np.random.rand() * 0.3)

        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-10})
    
    if res.success:
        # Find the circle with the smallest radius to trigger expansion
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Calculate pairwise distances for adjacency matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find least constrained circle (largest minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        smallest_radius_idx = np.argmin(radii)

        # Apply asymmetric expansion to trigger global reconfiguration
        target_radii_sum = np.sum(radii) + 0.0085
        expansion_factor = (target_radii_sum - np.sum(radii)) / (n - 1)

        # Create new radii with expansion on smallest radius
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != smallest_radius_idx:
                new_radii[i] += expansion_factor * (1 + np.random.rand() * 0.2)

        # Update decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    
    if res.success:
        # Final optimization pass for convergence
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Calculate pairwise distances for adjacency matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Enforce non-overlap with hard constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx_val = centers[i, 0] - centers[j, 0]
                dy_val = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx_val**2 + dy_val**2)
                if dist < radii[i] + radii[j] - 1e-12:
                    # Adjust radii to enforce non-overlap
                    overlap = (radii[i] + radii[j]) - dist
                    adjustment = overlap * 0.9 / (n - 1)
                    radii[i] -= adjustment
                    radii[j] -= adjustment

        v = res.x
        v[2::3] = np.clip(radii, 1e-6, 0.5)

        # Final re-evaluation to ensure constraints are satisfied
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Return the final result
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
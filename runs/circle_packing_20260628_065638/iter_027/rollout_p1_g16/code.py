import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Generate optimized initial positions using staggered hexagonal tiling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid: center at [col + 0.5]/cols, [row + 0.5]/rows
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        x = base_x + np.random.uniform(-0.05, 0.05)
        y = base_y + np.random.uniform(-0.05, 0.05)
        # Stagger alternate rows to form hexagonal pattern
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with tighter geometric spacing
    # Start with 0.38 / cols - 1e-3 for tighter spacing than previous SOTA
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bound entries for 26 circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraint with function signature fix
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # First optimization run with increased max iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-12})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Radical geometric reconfiguration using non-local spatial hashing
        # Generate a non-local geometric hash to break symmetry
        spatial_hash = np.random.rand(n, 2) * 0.08
        # Apply non-linear spatial reconfiguration based on radius
        for i in range(n):
            # Scale spatial displacement by radius to preserve spatial integrity
            v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))

        # Re-evaluate with geometric tiling constraint
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-13})

    # Non-local expansion of least constrained circle with spatial-aware topological adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)

        # Identify least constrained circle with spatial topology awareness
        # Least constrained circle is the one with largest minimum distance to others
        least_constrained_idx = np.argmax(min_dists)
        # Select least constrained circles with highest min distance as candidates
        candidate_indices = np.argsort(min_dists)[-3:]  # Select 3 least constrained circles

        # Calculate potential total expansion based on current sum and potential
        current_total = np.sum(radii)
        # Set growth target to 2% of current total with spatial awareness
        target_total_sum = current_total + 0.02 * current_total * np.mean(radii) / np.min(radii)
        expansion_factor = (target_total_sum - current_total) / (n - 1)

        # Stochastic expansion applied to all three candidates with radius-based scaling
        new_radii = radii.copy()
        for idx in candidate_indices:
            # Over-express for least constrained, under-express for others
            new_radii[idx] += expansion_factor * 1.15 * (np.sqrt(radii[idx]) / np.sqrt(np.mean(radii)))
            for i in range(n):
                if i != idx:
                    # Randomized expansion with inverse scaling to less constrained circles
                    new_radii[i] += expansion_factor * (1.0 + 0.2 * np.random.rand()) * (np.sqrt(radii[i]) / np.sqrt(np.mean(radii)))

        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, reduce expansion by 15% in a controlled manner
                new_radii = radii + (new_radii - radii) * 0.85
        
        # Final optimization pass with tighter constraints and spatial regularization
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-13})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
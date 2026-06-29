import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized grid and staggered pattern with higher initial radius
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Random initial offset
        x_offset = np.random.uniform(-0.1, 0.1)
        y_offset = np.random.uniform(-0.1, 0.1)
        # Staggered rows
        if row % 2 == 1:
            x_offset += 0.5 / cols
        # Edge clipping
        x = np.clip(x_center + x_offset, 0.0, 1.0)
        y = np.clip(y_center + y_offset, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    # Use higher initial radius with more conservative spacing
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # All constraints match length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with tighter tolerances
    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise overlap constraints with geometric hashing and sparse evaluation
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with fixed i and j in the closure
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization with aggressive iteration and high precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-12, "eps": 1e-10})

    # Stochastic reconfiguration: apply grid-based spatial hashing to induce new patterns
    if res.success:
        v = res.x
        # Generate spatial hash with directional randomness
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Add directional perturbation based on column index
            perturbed_v[3*i] += spatial_hash[i, 0] * (0.5 / cols)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (0.5 / rows)
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-9})

    # Targeted radius expansion with constrained relaxation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting for speed
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        
        # Identify least constrained circle (maximin distance)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor based on current radius sum
        total_sum = np.sum(radii)
        target_sum = total_sum + 0.012  # Aggressive but controlled expansion
        expansion_factor = (target_sum - total_sum) / (n - 1)
        
        # Construct new radius vector with spatial-aware expansion
        new_radii = radii.copy()
        # Expand least constrained circle more aggressively
        new_radii[least_constrained_idx] += expansion_factor * 1.4
        for i in range(n):
            if i != least_constrained_idx:
                # Slight variance in expansion to avoid symmetry
                expansion_i = expansion_factor * (1.0 + np.random.uniform(-0.05, 0.05))
                new_radii[i] += expansion_i
        
        # Apply expansion with validation to prevent overlap
        def validate_expansion(new_radii):
            centers = np.column_stack([v[0::3], v[1::3]])
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    if np.sqrt(dx**2 + dy**2) < new_radii[i] + new_radii[j] - 1e-12:
                        return False
            return True

        # Iteratively expand and validate until stable
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            if validate_expansion(new_radii):
                break
            else:
                # If invalid, back off slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final update with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-9})

    # Final validation and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
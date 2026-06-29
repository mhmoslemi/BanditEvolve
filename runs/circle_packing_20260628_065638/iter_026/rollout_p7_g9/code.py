import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    
    # Initialize with geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / cols
        # Add randomized offsets for spatial diversity
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Staggered grid for alternate rows
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

    # Construct constraints with delayed lambda capture
    cons = []
    for i in range(n):
        for offset, dir in enumerate([(1, 'L'), (0, 'R'), (1, 'B'), (0, 'T')]):
            # Use partials to ensure correct closure capture
            def make_bound_func(i, offset, dir):
                def bound_func(v):
                    if dir == 'L':
                        return 1.0 - v[3*i] - v[3*i + 2]
                    elif dir == 'R':
                        return v[3*i] - v[3*i + 2]
                    elif dir == 'B':
                        return 1.0 - v[3*i + 1] - v[3*i + 2]
                    elif dir == 'T':
                        return v[3*i + 1] - v[3*i + 2]
                    return 0.0
                return bound_func
            cons.append({"type": "ineq", "fun": make_bound_func(i, offset, dir)})

    # Vectorized overlap constraints using precomputed indices
    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_func(i, j):
                def overlap_func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i + 1] - v[3*j + 1]
                    return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
                return overlap_func
            cons.append({"type": "ineq", "fun": make_overlap_func(i, j)})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})

    # Asymmetric reconfiguration: introduce stochastic spatial hashing
    if res.success:
        v = res.x
        # Compute baseline distances to identify tightest packing constraints
        dx = v[0::3][:, np.newaxis] - v[0::3]
        dy = v[1::3][:, np.newaxis] - v[1::3]
        base_dists = np.sqrt(dx**2 + dy**2)
        base_radii = v[2::3]
        
        # Identify circles with minimal distance to others (tightest packing)
        min_distances = np.min(base_dists, axis=1)
        tightest_idx = np.argsort(min_distances)[:3]
        
        # Generate new spatial hash map with controlled jitter
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        
        # Apply jitter to tightest circles only
        for i in tightest_idx:
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i + 1] += spatial_hash[i, 1]
        
        # Re-optimize with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        dx = v[0::3][:, np.newaxis] - v[0::3]
        dy = v[1::3][:, np.newaxis] - v[1::3]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate base constraint tightness for expansion control
        base_overlap = dists[least_constrained_idx, :] - (radii[least_constrained_idx] + radii)
        tightest_overlap = np.min(base_overlap[base_overlap > 0]) if np.any(base_overlap > 0) else 0.1
        
        # Compute expansion vector with spatial constraints
        target_total_sum = np.sum(radii) + 0.006
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Apply expansion on all circles except least constrained
        new_radii = np.array(radii)
        new_radii[least_constrained_idx] = radii[least_constrained_idx]
        
        # Apply expansion to other circles with spatial-aware adjustments
        for i in range(n):
            if i != least_constrained_idx:
                # Add random expansion factor for spatial diversity
                expansion = expansion_factor * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion
        
        # Validate expansion with spatial constraints
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
                # If invalid, increase spacing slightly by reducing expansion
                new_radii -= (new_radii - radii) * 0.1
        
        # Update decision vector and re-optimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
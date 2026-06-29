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
        
        # Apply a more sophisticated randomized offset with spatial correlation
        # This creates cluster-aware and non-uniform spatial distribution
        x_offset = np.random.normal(0, 0.02) * (1.0 / (cols + 1))
        y_offset = np.random.normal(0, 0.015) * (1.0 / (rows + 1))
        x = x_center + x_offset
        y = y_center + y_offset
        
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

    # Vectorized constraints for boundaries with closure capture
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
    
    # Vectorized overlap constraints with geometric hashing and efficient computation
    def compute_overlap_distance(i, j):
        return lambda v: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": compute_overlap_distance(i, j)})
    
    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Asymmetric topological disruption: apply randomized geometric hashing
    if res.success:
        v = res.x
        # Random geometric hashing to force spatial reconfiguration
        random_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Targeted radius expansion on the most under-constrained circle with adjacency-aware optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized pairwise distance calculation using broadcasting
        C = centers
        dists = np.sqrt(np.sum((C[:, np.newaxis] - C[np.newaxis, :]) ** 2, axis=2))
        
        # Calculate minimum distance for each circle
        min_dists = np.min(dists, axis=1)
        # Find the most under-constrained circle (maximum minimum distance)
        least_constrained_idx = np.argmax(min_dists)
        # Find the circle with smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        
        # Compute current total sum of radii
        total_sum = np.sum(radii)
        # Calculate expansion factor based on total radius budget
        expansion_factor = 0.01 / max(1, n - 1)  # Controlled expansion to unlock new configuration
        
        # Create optimized radius vector with adjacency-aware expansion
        new_radii = radii.copy()
        # Expand the most under-constrained circle more to trigger layout re-adjustment
        new_radii[least_constrained_idx] += expansion_factor * 1.4
        new_radii[smallest_radius_idx] += expansion_factor * 1.4
        
        # Update the radii vector to maintain non-overlap
        for i in range(n):
            if i != least_constrained_idx and i != smallest_radius_idx:
                # Apply expansion in a gradient-based manner with spatial awareness
                # Calculate how far this circle is from its neighbors to determine expansion
                mean_dist = np.mean(dists[i, dists[i] != 0])
                new_radii[i] += expansion_factor * (1 + 0.5 * (1 - radii[i] / mean_dist))
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized grid and asymmetric offset for spatial diversification
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add randomized asymmetric offset with spatial hashing to avoid symmetry
        x_offset = np.random.uniform(-0.12, 0.12) - (row % 4) * 0.04
        y_offset = np.random.uniform(-0.12, 0.12) - (col % 4) * 0.04
        x = x_center + x_offset
        y = y_center + y_offset
        # Stagger alternate rows to promote non-uniform distribution
        if row % 2 == 1:
            x += 0.5 / cols * (0.9 + np.random.rand() * 0.1)
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on spatial hash and geometric expansion
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds with strict radius constraints
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3*n

    # Objective function to maximize total radii (minimize negative sum)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized non-overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            # Use explicit lambda with fixed i and j to avoid capture issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tight tolerance and convergence checks
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-12})
    
    # Disruptive geometric transformation: randomized spatial hashing and radius expansion
    if res.success:
        v = res.x
        # Compute centers and radii
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Apply spatial hashing for reconfiguration
        hashes = np.random.rand(n, 2) * 0.04
        for i in range(n):
            v[3*i] += hashes[i, 0] * (1.1 + np.random.rand() * 0.1)
            v[3*i+1] += hashes[i, 1] * (1.1 + np.random.rand() * 0.1)
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-12})
    
    # Targeted expansion on the circle with the minimal expansion potential
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros((n, n), dtype=np.float64)
        
        # Vectorized distance matrix calculation (avoid nested loops)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the least constrained circle by minimal required expansion
        min_dists = np.min(dists, axis=1)
        min_expansion_index = np.argmin(min_dists)
        
        # Expand this circle aggressively and distribute the gain
        current_total = np.sum(radii)
        target_total = current_total + 0.010  # Aggressive but controlled expansion
        expansion_per_circle = (target_total - current_total) / (n - 1)
        expansion_multiplier = 1.2  # Extra for triggering layout change
        
        # Create new radii with asymmetrical expansion and validation
        new_radii = radii.copy()
        new_radii[min_expansion_index] += expansion_per_circle * expansion_multiplier
        for i in range(n):
            if i != min_expansion_index:
                new_radii[i] += expansion_per_circle
        
        # Apply new radii and re-evaluate with updated constraints
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-12})
    
    # Final refinement and validation pass
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros((n, n), dtype=np.float64)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Additional validation pass to ensure constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = dists[i, j]
                if dist < radii[i] + radii[j] - 1e-12:
                    # If overlap, reduce expansion slightly
                    reduction = (radii[i] + radii[j] - dist) * 0.05
                    radii[i] = np.clip(radii[i] - reduction, 1e-6, None)
                    radii[j] = np.clip(radii[j] - reduction, 1e-6, None)
        
        # Final update to radii
        v[2::3] = radii
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
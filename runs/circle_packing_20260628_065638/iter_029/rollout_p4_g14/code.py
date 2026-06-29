import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols  # 5 rows
    
    # Initialize with refined spatial tessellation and dynamic jittering
    # First pass - base grid
    xs_base = np.array([(i % cols + 0.5)/cols for i in range(n)])
    ys_base = np.array([(i // cols + 0.5)/rows for i in range(n)])
    
    # Second pass - spatial jitter with dynamic noise scale
    jitter_x = np.random.uniform(-0.02, 0.02, n)  # Reduced noise for better stability
    jitter_y = np.random.uniform(-0.02, 0.02, n)
    
    # Alternate row staggering
    xs = xs_base + jitter_x
    ys = ys_base + jitter_y
    for i in range(n):
        if (i // cols) % 2 == 1:
            xs[i] += 0.05 / cols  # 5% of unit cell width for staggering

    # Base radius using row-wise spacing
    base_radius = 0.4 / cols - 1e-4  # 0.0367 for 5 cols, reduced from 0.035 for tighter fit
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, base_radius)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda closure and i binding
    cons = []
    for i in range(n):
        def make_bound_func(kind, i=i):
            def constraint_func(v):
                if kind == 'x_left':
                    return v[3*i] - v[3*i+2]  # x - r >= 0
                elif kind == 'x_right':
                    return 1.0 - v[3*i] - v[3*i+2]  # 1 - x - r >= 0
                elif kind == 'y_bottom':
                    return v[3*i+1] - v[3*i+2]  # y - r >= 0
                elif kind == 'y_top':
                    return 1.0 - v[3*i+1] - v[3*i+2]  # 1 - y - r >= 0
                assert False
            return constraint_func
        
        cons.append({"type": "ineq", "fun": make_bound_func('x_left')})
        cons.append({"type": "ineq", "fun": make_bound_func('x_right')})
        cons.append({"type": "ineq", "fun": make_bound_func('y_bottom')})
        cons.append({"type": "ineq", "fun": make_bound_func('y_top')})

    # Vectorized pairwise distance constraints with caching for efficiency
    pair_dist_squared = np.zeros(n * n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = i * n + j
            idx_j = j * n + i
            def make_overlap_func(i=i, j=j):
                def constraint_func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    r1 = v[3*i+2]
                    r2 = v[3*j+2]
                    return dx*dx + dy*dy - (r1 + r2)**2
                return constraint_func
            cons.append({"type": "ineq", "fun": make_overlap_func()})
    
    # Optimization phase 1: base spatial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-10})
    
    # Optimization phase 2: post-spatial perturbation with adaptive geometric hashing
    if res.success:
        v = res.x
        # Compute current radii and centers
        current_radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling based on spatial density
        spatial_hash = np.random.rand(n, 2) * 0.03 * np.repeat(current_radii, 2) / current_radii.mean()
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Phase 2 optimization with tighter tolerances
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    # Optimization phase 3: targeted expansion under soft geometric constraints
    if res.success:
        v = res.x
        current_radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance matrix using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle by minimizing the maximum pairwise distance
        max_distances = np.max(dists, axis=1)
        least_constrained_idx = np.argmin(max_distances)  # Circle with minimal "reach" is least constrained
        
        # Compute expansion potential based on spatial density and existing radii
        current_radius = current_radii[least_constrained_idx]
        expansion_factor = 0.0062 / (n * 0.5)  # Conservative estimate, can be tuned if needed
        new_radius = current_radius + expansion_factor * (current_radius / np.mean(current_radii))
        
        # Create a radial expansion plan with gradient-based soft constraints
        expansion_v = v.copy()
        expansion_v[3*least_constrained_idx+2] = new_radius
        
        # Re-evaluate with new radii
        res = minimize(neg_sum_radii, expansion_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    # Final cleanup and validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
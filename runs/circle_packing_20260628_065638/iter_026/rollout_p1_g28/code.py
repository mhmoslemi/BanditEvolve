import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Use fixed seed for deterministic initialization, but with enhanced spatial scattering
    np.random.seed(42)
    
    # Initialize positions with geometric clustering, staggered grid, and randomized offsets for better distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add more substantial randomized offset for better spatial distribution
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
        # Shift alternate rows for staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Larger initial radius with a smaller margin for better early optimization
    r0 = 0.35 / cols - 1e-4  # Reduced margin to allow more radial space
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Same length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # Left edge constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right edge constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom edge constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top edge constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized non-overlap constraint using geometric hashing and precomputed indices
    overlap_constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_constraints.append({"type": "ineq", "fun": constraint_func})

    cons.extend(overlap_constraints)

    # Initial optimization with tight tolerances and increased iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-12, "disp": False})
    
    # Asymmetric reconfiguration with spatial perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial perturbation from Gaussian noise
        perturbation = np.random.normal(0, 0.015, (n, 2)) / cols
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += perturbation[i, 0]
            new_v[3*i+1] += perturbation[i, 1]
        
        # Re-optimize with perturbed positions
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12, "disp": False})
    
    # Targeted radius expansion using gradient-aware least constrained circle selection
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate min distances and find least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        
        # Calculate expansion factor and perform expansion
        target_total_sum = np.sum(radii) + 0.009  # Add 0.9% increase
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Introduce spatially aware expansion with gradient direction guidance
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1 + 0.0005  # Small directional boost
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validation
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_ij = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_ij = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_ij**2 + dy_ij**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion with exponential decay for faster convergence
                new_radii = radii + (new_radii - radii) * (1 - 0.95 ** 5)
        
        # Final optimization with tightened constraints
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12, "disp": False})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
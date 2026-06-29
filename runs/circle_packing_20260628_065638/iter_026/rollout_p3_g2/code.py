import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with asymmetric randomized grid + staggered layout
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add asymmetrical randomized offsets with gradient-based distribution
        x_offset = np.random.uniform(-0.04, 0.04)
        y_offset = np.random.uniform(-0.04, 0.04)
        # Introduce subtle spatial asymmetry for improved packing
        if row % 2 == 1:
            x_offset += 0.02 / cols
        if col % 2 == 1:
            y_offset += 0.02 / rows
        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimate with dynamic scaling
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

    # Vectorized constraints for boundaries - use lambda with capture
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with spatial-aware gradient approximation
    for i in range(n):
        for j in range(i + 1, n):
            # Create closure with fixed i and j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                # Precompute squared distance for better gradient control
                dist_sq = dx*dx + dy*dy
                # Use squared radius sum to avoid sqrt for faster calculation
                # Constraint ensures dist_sq >= (r_i + r_j)^2
                return dist_sq - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization: refined initial configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Asymmetric reconfiguration: apply spatially-distributed stochastic perturbation
    if res.success:
        v = res.x
        # Create spatial perturbation matrix with gradient-aware scaling
        perturbation = np.random.rand(n, 2) * 0.05  # Adjusted perturbation range
        perturbation *= (n - (1 - np.cos(2 * np.pi * np.linspace(0, 1, n)))) / (n - 1)  # Gradient-aware scaling
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-12})
    
    # Asymmetric radius expansion on least-constrained circle with gradient refinement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized pairwise distance and minimum distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        
        # Find optimal least-constrained circle with gradient-aware index selection
        least_constrained_idx = np.argmax(min_dists)
        # Use cosine-based expansion factor to enhance less constrained areas
        expansion_factor = 0.0075 / (n - 1)  # Controlled gradient-aware expansion
        
        # Create radius adjustment vector with spatial gradient refinement
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.4  # Boost expansion for least constrained
        for i in range(n):
            # Apply spatial gradient-aware adjustment
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (0.9 + 0.1 * np.cos(np.pi * i / n))  # Gradient refinement
        
        # Apply expansion with constraint validation using optimized distance check
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Compute pairwise distances using vectorized broadcasting
            dx = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1]
            dist_sq = dx**2 + dy**2
            # Compute pairwise radii sum
            radii_sum = new_radii[:, np.newaxis] + new_radii[np.newaxis, :]
            # Check constraint for all pairs using vectorized comparison
            valid = (dist_sq >= radii_sum**2 - 1e-12).all()
            
            if valid:
                break
            else:
                # If invalid, reduce expansion based on gradient-aware decay
                new_radii = radii + (new_radii - radii) * 0.96
        
        # Update decision vector with refined configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with refined configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
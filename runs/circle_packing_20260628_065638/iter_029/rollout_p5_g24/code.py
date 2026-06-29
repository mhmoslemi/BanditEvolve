import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    # Optimize grid spacing: dynamically adapt row height to increase column usage
    # Avoid rigid row/column mapping, allow for denser packing
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Centered column placement with dynamic offset
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / (rows + i % 2)  # row spacing adapts for staggered grid
        
        # Add small randomized perturbation for non-symmetrical clustering
        base_x = x_center + np.random.uniform(-0.02, 0.02)
        base_y = y_center + np.random.uniform(-0.02, 0.02)
        
        # Alternate row staggering with adaptive offset
        if row % 2 == 1:
            x_center += np.random.uniform(-0.08, 0.08) * (1.0 / (cols + 1))  # small dynamic shift
            base_x = x_center + np.random.uniform(-0.02, 0.02)
        
        xs.append(base_x)
        ys.append(base_y)
    
    # Use more aggressive initial radii based on spacing
    r0 = 0.4 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, matches v
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    # Use lambda capture with fixed i to avoid closures
    cons = []
    for i in range(n):
        # Left + radius <= 1
        def f1(v, i=i):
            return 1.0 - v[3*i] - v[3*i+2]
        cons.append({"type": "ineq", "fun": f1})
        # Right - radius >= 0
        def f2(v, i=i):
            return v[3*i] - v[3*i+2]
        cons.append({"type": "ineq", "fun": f2})
        # Bottom + radius <= 1
        def f3(v, i=i):
            return 1.0 - v[3*i+1] - v[3*i+2]
        cons.append({"type": "ineq", "fun": f3})
        # Top - radius >= 0
        def f4(v, i=i):
            return v[3*i+1] - v[3*i+2]
        cons.append({"type": "ineq", "fun": f4})
    
    # Vectorized overlap constraints with adaptive scaling
    # Use lambda capture to reduce overhead
    for i in range(n):
        for j in range(i + 1, n):
            def f_overlap(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": f_overlap})
    
    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-12, "eps": 1e-13})
    
    # Asymmetric reconfiguration: spatial perturbation with adaptive radius adjustment
    # Trigger localized spatial hashing with radius-aware perturbation
    if res.success:
        v = res.x
        current_radii = v[2::3]
        avg_radius = np.mean(current_radii)
        # Generate spatial hash with radius-adjusted stochasticity
        spatial_hash = np.random.rand(n, 2) * (0.08 * (avg_radius / 0.25))
        perturbed_v = v.copy()
        for i in range(n):
            # Radius-aware perturbation that scales with local spacing
            perturb_x = spatial_hash[i, 0] * (current_radii[i] / avg_radius)
            perturb_y = spatial_hash[i, 1] * (current_radii[i] / avg_radius)
            perturbed_v[3*i] += perturb_x
            perturbed_v[3*i+1] += perturb_y
            
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-13})

    # Trigger targeted radius expansion on least constrained circle
    # Use gradient-aware selection and iterative expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance matrix calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to neighbors
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate growth with adaptive coefficient
        current_total = np.sum(radii)
        # Use a growth target derived from empirical performance and constraints
        target_growth = 0.0065  # 0.65% increase
        expansion_factor = (target_growth / (n-1)) * (current_total / avg_radius)
        
        # Create expansion vector with adaptive distribution
        # Base expansion vector with slight variation to promote dynamic optimization
        new_radii = radii.copy()
        # Apply targeted expansion to least constrained circle
        expansion = expansion_factor * (1.0 + 0.1 * np.random.rand())  # slight random variation
        new_radii[least_constrained_idx] += expansion
        
        # Apply soft expansion to other circles with radius-aware scaling
        for i in range(n):
            if i != least_constrained_idx:
                # Add expansion proportional to radius to promote even growth
                new_radii[i] += expansion_factor * (radii[i] / avg_radius) * 1.01
        
        # Apply expansion with careful constraint checking
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Re-evaluate with expanded configuration
        # Use optimized constraint checking for performance
        max_iter = 400
        for it in range(max_iter):  # Iteratively adjust for non-overlapping
            # Recalculate centers to reflect new radii
            adjusted_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = adjusted_centers[i, 0] - adjusted_centers[j, 0]
                    dy = adjusted_centers[i, 1] - adjusted_centers[j, 1]
                    dist = np.hypot(dx, dy)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # If invalid, reduce expansion slightly
                reduction = 0.98 ** (it / max_iter)  # exponential damping
                new_radii = radii + (new_radii - radii) * (1 - reduction)
                expanded_v[2::3] = new_radii
        
        # Update decision vector with final expansion
        v = expanded_v
    
    # Final configuration with clipping for numerical safety
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
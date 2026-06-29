import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using grid-based seed with improved random dispersion and staggered offsets
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add randomized offset for improved initial dispersion
        x = x_center + np.random.uniform(-0.02, 0.02)
        y = y_center + np.random.uniform(-0.02, 0.02)
        # Apply staggered grid for spatial efficiency
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimate based on grid spacing and optimization tolerance
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure the bounds list has 3*n entries for the vector of length 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with stable lambda capture
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with optimized lambda capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with enhanced tolerances and higher max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11, "gtol": 1e-11})

    # Radical geometric reconfiguration using hierarchical spatial hashing + gradient refinement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash for hierarchical reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Apply spatial displacement based on normalized radius and hash
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Reinforce spatial constraints with gradient-based perturbation
        for iter in range(3):
            perturbed_v = v.copy()
            for i in range(n):
                perturbed_v[3*i] += np.random.normal(0, 0.005) * radii[i]
                perturbed_v[3*i+1] += np.random.normal(0, 0.005) * radii[i]
            
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})
        
        v = res.x if res.success else v
    
    # Topological reordering with adaptive radius expansion and constraint validation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances using vectorized operations
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with smallest radius and maximum spatial freedom
        min_radius = np.min(radii)
        smallest_radius_idx = np.where(radii == min_radius)[0][0]
        min_dists = np.min(dists, axis=1)
        max_free_idx = np.argmax(min_dists)
        primary_idx = max_free_idx if max_free_idx != smallest_radius_idx else smallest_radius_idx
        
        # Calculate expansion factor based on total potential
        current_total = np.sum(radii)
        target_growth = 0.0075
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii)) * 1.1
        
        # Apply differential expansion with constraint validation
        new_radii = radii.copy()
        new_radii[primary_idx] += expansion_factor * 1.2
        
        # Apply stochastic expansion to other circles while maintaining constraints
        for i in range(n):
            if i != primary_idx:
                expansion_i = expansion_factor * (1.0 + np.random.uniform(-0.3, 0.3))
                new_radii[i] += expansion_i
        
        # Validate and refine expanded radii with local refinement
        iterations = 0
        while iterations < 2:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between circles
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
                # If overlap detected, reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.96
                iterations += 1
        
        # Final re-evaluation with optimized configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
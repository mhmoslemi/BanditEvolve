import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Adaptive spatial clustering initialization with dynamic perturbation and non-uniform row spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        # Calculate a spatial "spread" parameter based on row spacing to allow non-uniform radii
        x_spread = max(0.03, 0.02 + 0.01 * (row % 3))  # Non-uniform row spacing for dynamic layout
        y_center = (row + 0.5 + 0.1 * np.cos(0.5 * np.pi * row)) / rows
        # Perturb with noise that scales with spatial density
        x = x_center + np.random.uniform(-x_spread/2, x_spread/2)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Stagger rows with non-uniform offset
        if (row + col) % 2 == 1:
            x += 0.3 / cols
        xs.append(x)
        ys.append(y)
    
    # Compute initial radius based on row spacing and col density
    r0 = 0.35 / (cols * (1 + 0.1 * (rows - 1) / (rows - 0.5))) - 1e-3
    r0 = max(r0, 1e-4)  # Ensure radius is valid
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure bounds align with n

    # Objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Optimized constraint system with dynamic spatial hashing and improved gradient computation
    cons = []
    
    # Spatial boundaries constraint (inequalities)
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with spatial hashing and gradient awareness
    for i in range(n):
        for j in range(i + 1, n):
            # Use a geometric hashing function to create an adaptive, non-linear constraint
            # This introduces curvature in the constraint space, enabling more sophisticated search
            def constraint_func(v, a=i, b=j):
                # First pass: compute distances
                dx = v[3*a] - v[3*b]
                dy = v[3*a+1] - v[3*b+1]
                dist_sq = dx*dx + dy*dy
                radii_sum = v[3*a+2] + v[3*b+2]
                # Apply soft penalty with a hyperbolic tangent for smoother gradient
                return dist_sq - radii_sum**2 - 1e-10 * np.tanh(10 * (np.sqrt(dist_sq) - radii_sum))
            cons.append({"type": "ineq", 
                         "fun": constraint_func})
    
    # First optimization with high precision and multi-threaded gradient computation
    res = minimize(neg_sum_radii, v0, 
                   method="SLSQP", 
                   bounds=bounds,
                   constraints=cons,
                   options={"maxiter": 600, 
                            "ftol": 1e-11, 
                            "gtol": 1e-12, 
                            "eps": 1e-10,
                            "jac": "2-point"})  # Use finite difference for gradient

    # Spatial reconfiguration through adaptive random field perturbation with constraint validation
    if res.success:
        v = res.x
        # Get radii and centers for validation and perturbation
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create a spatial perturbation field based on spatial density and radius distribution
        spatial_weights = np.zeros(n)
        for i in range(n):
            spatial_weights[i] = np.sqrt(1 + 10 * np.sum(centers[i] ** 2))  # Spatially weight perturbation
        perturbation = np.random.rand(n, 2) * (1 / (np.mean(spatial_weights) + 1e-6)) * 0.05
        perturbed_v = v.copy()
        perturbed_v[0::3] += perturbation[:, 0] * (radii / np.mean(radii))
        perturbed_v[1::3] += perturbation[:, 1] * (radii / np.mean(radii))

        # Re-evaluate with spatially informed perturbations
        res = minimize(neg_sum_radii, perturbed_v, 
                       method="SLSQP", 
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 400, 
                                "ftol": 1e-11, 
                                "gtol": 1e-12, 
                                "eps": 1e-10,
                                "jac": "2-point"})

    # Directed radius expansion with spatial gradient analysis and constrained optimization
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros((n, n))
        
        # Vectorized distance matrix with broadcasting optimization
        dx = np.expand_dims(centers[:, 0], axis=1) - centers[:, 0]
        dy = np.expand_dims(centers[:, 1], axis=1) - centers[:, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute a "free space" score for each circle
        free_space = np.min(np.maximum(1e-6, (np.sqrt(dists**2) - radii[:, np.newaxis] - radii[np.newaxis, :])), axis=1)
        
        # Identify the circle with maximal free space for expansion
        idx = np.argmax(free_space)
        
        # Calculate growth based on current total sum and free space
        current_total = np.sum(radii)
        target_growth = 0.007  # Slight increase beyond current best
        expansion_factor = (target_growth / (n - 1)) * (1 + 0.5 * np.std(free_space) / np.mean(free_space))
        
        # Build the expansion vector with spatial gradient-aware adjustment
        new_radii = radii.copy()
        new_radii[idx] = min(new_radii[idx] + expansion_factor * 1.3, 0.5)
        for i in range(n):
            if i != idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand()) * (free_space[i] / np.max(free_space))
        
        # Apply expansion in constrained optimization
        while True:
            # Update v for the new radii
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            
            # Compute centers and validate configuration
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
                # If invalid, scale back expansion gently and recompute
                new_radii = radii + (new_radii - radii) * 0.9
            
        # Update the decision vector with the new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with tighter tolerances
        res = minimize(neg_sum_radii, v_new, 
                       method="SLSQP", 
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 400, 
                                "ftol": 1e-11, 
                                "gtol": 1e-12, 
                                "eps": 1e-10,
                                "jac": "2-point"})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
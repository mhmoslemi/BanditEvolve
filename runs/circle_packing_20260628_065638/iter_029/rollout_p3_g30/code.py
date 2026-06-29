import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions: advanced spatial hashing to break symmetry and ensure spatial diversity
    xs = []
    ys = []
    # Generate cluster centers with multi-layered perturbation
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add spatially aware perturbations: geometric hashing + direction-aware displacement
        # Use different seed for each particle to maximize unique spatial arrangements
        perturbation = np.random.rand(2, 2) * 0.04
        x = x_center + perturbation[0, 0] - perturbation[0, 1]
        y = y_center + perturbation[1, 0] - perturbation[1, 1]
        
        # Staggered layout with phase shift to create non-uniform patterns
        if row % 2 == 1:
            x += 0.4 / cols
            y += 0.15 / rows
        xs.append(x)
        ys.append(y)
    
    # Set initial radius bounds with spatial awareness
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure the bounds list has 3*n entries to match the vector length
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with captured i to prevent lambda closure issues
    cons = []
    for i in range(n):
        # left + radius <= 1: x[3*i] + radius[3*i+2] <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # right - radius >= 0: x[3*i] - radius[3*i+2] >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # bottom + radius <= 1: y[3*i+1] + radius[3*i+2] <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # top - radius >= 0: y[3*i+1] - radius[3*i+2] >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise overlap constraints with spatial-aware gradient approximation
    for i in range(n):
        for j in range(i + 1, n):
            # Compute distance constraint with spatially sensitive bounds for tight control
            # Use dynamic bounds with spatial awareness (radius + radius vs. current spacing)
            # This is the crux of the strategy: dynamically adjusting overlap bounds
            # to explore non-symmetric optimal configurations
            cons.append({"type": "ineq",
                          "fun": lambda v, i=i, j=j: 
                              (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                              - (np.sqrt(v[3*i+2] + v[3*j+2]) + 1e-6)**2})

    # Initial optimization with increased iterations, tighter tolerance, and enhanced gradient control
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 1500,
                       "ftol": 1e-10,
                       "gtol": 1e-10,
                       "eps": 1e-8,
                       "disp": False
                   })

    if res.success:
        v = res.x
        # Use local spatial mapping and spatial hashing for refined perturbation
        spatial_map = np.random.rand(n, 2) * 0.06
        # Apply directional spatial perturbation based on particle radius and position
        perturbed_v = v.copy()
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        for i in range(n):
            # Add directional perturbation proportional to radius for more efficient space exploration
            perturbation_x = spatial_map[i, 0] * (radii[i] / np.mean(radii))
            perturbation_y = spatial_map[i, 1] * (radii[i] / np.mean(radii))
            perturbed_v[3*i] += perturbation_x
            perturbed_v[3*i+1] += perturbation_y
        
        # Re-optimization with perturbed positions: this enhances local exploration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 400,
                           "ftol": 1e-12,
                           "eps": 1e-8
                       })

    # Targeted radius expansion with soft constraint-aware strategy
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))

        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute spatial influence matrix for informed expansion
        influence = np.zeros(n)
        for i in range(n):
            influence[i] = np.mean(dists[i, dists[i] > 1e-3])  # Ignore near-zero distances
            
        # Calculate growth based on distance to nearest neighbors and radii
        least_constrained_idx = np.argmax(influence)
        current_total = np.sum(radii)
        target_growth = 0.0075
        max_growth_per_circle = 0.025
        
        # Calculate expansion vector with adaptive growth allocation
        # Prioritize circles with minimal spatial constraints
        # Distribute growth to maximize radii growth while maintaining feasibility
        new_radii = radii.copy()
        avg_radius = np.mean(radii)
        for i in range(n):
            if i != least_constrained_idx:
                growth = (target_growth * (1 + 0.2 * np.random.rand())) / (n - 1)
                growth = np.clip(growth, 0, max_growth_per_circle * (1.0 + 0.5 * (radii[i]/avg_radius)))
                new_radii[i] += growth
        
        # Create expansion vector
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Validate new configuration before re-optimization
        # Use a direct validation check for performance
        try:
            # Apply constraints explicitly during validation
            # This ensures we have a valid configuration before attempting re-optimization
            # Avoid recomputing constraints during optimization for speed
            for i in range(n):
                x, y = expanded_v[3*i], expanded_v[3*i+1]
                r = expanded_v[3*i+2]
                # Check boundaries
                if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
                    raise ValueError("Boundary violation")
            
            for i in range(n):
                xi, yi = expanded_v[3*i], expanded_v[3*i+1]
                ri = expanded_v[3*i+2]
                for j in range(i+1, n):
                    xj, yj = expanded_v[3*j], expanded_v[3*j+1]
                    rj = expanded_v[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < ri + rj - 1e-12:
                        raise ValueError("Overlap found")
            # If all constraints satisfied, re-validate and optimize
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={
                               "maxiter": 300,
                               "ftol": 1e-12,
                               "eps": 1e-8
                           })
        except Exception:
            # If validation fails, keep the current solution
            pass
    else:
        # Fallback to original configuration if optimization fails
        v = v0
    
    # Final cleanup and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
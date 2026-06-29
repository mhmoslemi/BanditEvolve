import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with adaptive clustering and dynamic staggering
    xs = []
    ys = []
    
    # Create a grid where each cell is a hexagon for better circle arrangement
    # Calculate minimum spacing between circles based on rows and columns
    min_spacing = 0.25 / (cols ** 0.5)
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base position based on grid
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        
        # Add dynamic jitter to break symmetry and improve circle alignment
        jitter = np.random.uniform(-0.05, 0.05, size=2)
        
        # Adjust x position based on row to implement staggered hexagonal packing
        if row % 2 == 1:
            x_base += 0.5 / cols
        
        x = x_base + jitter[0]
        y = y_base + jitter[1]
        
        # Clamp to unit square
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with a reasonable minimum and initial guess
    # Start with higher radii for better optimization trajectory
    # Based on circle density and hexagonal packing
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for optimization
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        """Objective function: minimize -sum(radii) == maximize sum(radii)"""
        return -np.sum(v[2::3])

    # Define constraints: 4 for each circle (boundaries)

    # Note: lambda closures with i capture can be dangerous - use positional capture
    cons = []
    for i in range(n):
        # x <= 1 - r
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # x >= r
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # y <= 1 - r
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # y >= r
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Define circle-circle constraints in vectorized form with efficient access
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j:
                       (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                       - (v[3*i+2] + v[3*j+2])**2)
            })

    # Initial optimization with aggressive settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 2000,
                       "ftol": 1e-12,
                       "eps": 1e-12,
                       "disp": False
                   })

    # Advanced reconfiguration phase using targeted perturbation
    # We aim to perturb circles with smaller radii to escape local minima
    # We use a "shake" technique with radius-aware perturbation
    shake_factor = 0.15
    if res.success:
        v = res.x
        radii_current = v[2::3]
        centers_current = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distances between all circles for constraint awareness
        dx = centers_current[:, np.newaxis, 0] - centers_current[np.newaxis, :, 0]
        dy = centers_current[:, np.newaxis, 1] - centers_current[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute distance to nearest neighbor for each circle
        min_dists = np.min(dists, axis=1)
        
        # Identify circles that are most constrained to perturb
        constraint_mask = (min_dists < 0.8 * np.max(min_dists))  # Adjust threshold
        
        # Create perturbation vector with radius-aware weighting
        # Larger perturbations on more constrained (tighter packed) circles
        shake_weights = np.where(constraint_mask, 1.0, 0.5)
        perturbations = np.random.uniform(-shake_factor * radii_current, 
                                          shake_factor * radii_current, 
                                          size=3*n)
        perturbations[2::3] *= shake_weights  # Apply more force on constrained
        
        # Apply perturbation to current solution
        v_perturbed = v + perturbations
        
        # Re-evaluate with perturbed vector
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 400,
                           "ftol": 1e-12,
                           "eps": 1e-12,
                           "disp": False
                       })

    # Final optimization phase with advanced radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix for constrained circle detection
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with the most space to grow
        # We use the minimal distance to nearest neighbor as proxy for "space"
        min_dists = np.min(dists, axis=1)
        circle_to_expand = np.argmin(min_dists)  # smallest distance has most space to grow
        
        # Compute current total radii sum
        curr_total = np.sum(radii)
        # Aim for growth of 0.008 relative to current total
        target_total = curr_total + 0.008
        growth_per_circle = (target_total - curr_total) / n
        
        # Create a new_radius array with growth applied
        # Apply more growth to the circle with most space
        new_radii = radii + growth_per_circle * (1.0 + 0.2 * (min_dists / np.mean(min_dists)))
        
        # Ensure that radii do not exceed max bounds
        new_radii = np.clip(new_radii, 1e-6, 0.5)
        
        # Create a new vector with updated radii
        v_expanded = v.copy()
        v_expanded[2::3] = new_radii
        
        # Re-evaluate with expanded radii
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 400,
                           "ftol": 1e-12,
                           "eps": 1e-12,
                           "disp": False
                       })

    # Final fallback if optimization failed
    v = res.x if res.success else v0
    centers_final = np.column_stack([v[0::3], v[1::3]])
    radii_final = np.clip(v[2::3], 1e-6, None)
    
    return centers_final, radii_final, float(radii_final.sum())
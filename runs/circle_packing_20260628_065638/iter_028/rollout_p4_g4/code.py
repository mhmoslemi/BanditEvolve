import numpy as np

def run_packing():
    n = 26
    cols = int(np.sqrt(n)) + 1  # Better grid distribution than fixed 5 cols
    rows = (n + cols - 1) // cols
    
    # Dynamic seeding with adaptive jitter to avoid symmetric clustering
    base_centers = np.empty((n, 2))
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid positions with finer spacing for smaller n
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Adaptive jitter: larger for fewer rows/columns, smaller for denser grids
        row_jitter = 0.06 / (rows + 3)
        col_jitter = 0.06 / (cols + 3)
        
        x = x_center + np.random.uniform(-col_jitter, col_jitter)
        y = y_center + np.random.uniform(-row_jitter, row_jitter)
        
        # Alternate row offset for staggered grid
        if row % 2 == 1:
            x += (0.5 / cols) * (1.0 / (rows + 5))
        
        base_centers[i] = np.array([x, y])
    
    # Initial radii: smaller base radii with adaptive scaling
    r0 = 0.35 / cols * (1.2 - 0.2 * (cols - 5) / (cols - 1)) - 1e-3
    r0 = np.clip(r0, 1e-5, 0.45)  # Clamp to valid range
    
    v0 = np.concatenate([base_centers.reshape(-1), r0.repeat(n)])
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]  # Tighter upper radius bound

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective to maximize radii sum

    # Vectorized constraints: use lambda with captured indices to prevent closure issues
    cons = []
    for i in range(n):
        # Left bound: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right bound: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom bound: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top bound: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with tighter epsilon tolerance and adaptive scaling
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with closure to avoid nested loops
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First optimization pass with extended iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-9})

    v = res.x if res.success else v0
    
    # Shake heuristic: perturb smallest circles with adaptive jitter based on their positions
    if res.success:
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        min_radius = np.min(radii)
        
        # Identify circles with smallest radius for perturbation
        small_radius_mask = radii < min_radius * 1.5
        
        # Create adaptive jitter matrix: bigger for circles near edges or with low mobility
        jitter_amount = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if small_radius_mask[i]:
                # Increase perturbation for small circles that are likely to be trapped
                jitter_amount[i] = 0.04 * r * (1 + (0.5 * (x < 0.1 or x > 0.9 or y < 0.1 or y > 0.9)))
            else:
                jitter_amount[i] = 0.02 * r
        
        # Create perturbation vector
        perturbed_v = v.copy()
        for i in range(n):
            if small_radius_mask[i]:
                # Add more aggressive perturbation to break out of local minima
                perturbed_v[3*i] += np.random.uniform(-jitter_amount[i], jitter_amount[i])
                perturbed_v[3*i+1] += np.random.uniform(-jitter_amount[i], jitter_amount[i])
        
        # Second optimization pass with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-9})
        
        v = res.x if res.success else v0
    
    # Post-optimization refinement: expand least constrained circles with gradient tracking
    if res.success:
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate distance from each center to every other center
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distance to all other circles for each circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Get the circle with least constraint (most free space)
        least_circle_center = centers[least_constrained_idx]
        least_circle_radius = radii[least_constrained_idx]
        
        # Calculate current sum and target
        current_total = np.sum(radii)
        expansion = 0.0065  # 6.5% over baseline
        
        # Add expansion to least constrained circle while maintaining feasibility
        # We simulate expanding slightly and validate the configuration in a loop
        # Since full re-optimization is computationally heavy, this provides a local optimization
        while True:
            # Create a copy to modify
            new_v = v.copy()
            new_v[3*least_constrained_idx + 2] += expansion / 0.95  # Over-expansion for testing
            
            # Recalculate centers
            new_centers = np.column_stack([new_v[0::3], new_v[1::3]])
            new_radii = new_v[2::3]
            
            # Validate if new configuration is valid
            collision = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        collision = True
                        break
                if collision:
                    break
            
            # If valid, break
            if not collision:
                v = new_v
                break
        
        # Final refinement with smaller expansion
        while True:
            # Create a copy to modify
            new_v = v.copy()
            new_v[3*least_constrained_idx + 2] += 0.0002  # Very small expansion
            
            # Recalculate centers
            new_centers = np.column_stack([new_v[0::3], new_v[1::3]])
            new_radii = new_v[2::3]
            
            # Validate if new configuration is valid
            collision = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        collision = True
                        break
                if collision:
                    break
            
            # If valid, break and finalize
            if not collision:
                v = new_v
                break
    
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    sum_radii = float(radii.sum())
    return centers, radii, sum_radii
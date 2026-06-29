import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n))) + 1
    rows = (n + cols - 1) // cols

    # Initialize with a more efficient grid and random spatial offset clustering
    xs = []
    ys = []

    for i in range(n):
        col = i % cols
        row = i // cols
        # Base grid coordinate
        base_x = (col + 0.25) / cols
        base_y = (row + 0.25) / rows
        # Random offset with shrinking variance to avoid overlapping
        x_offset = np.random.uniform(-0.15, 0.15) * (1.0 - (row + col) / (rows + cols))
        y_offset = np.random.uniform(-0.15, 0.15) * (1.0 - (row + col) / (rows + cols))
        # Staggered vertical shift for rows with odd index
        if row % 2 == 1:
            base_x += 0.5 / cols
        # Apply offset
        x = base_x + x_offset
        y = base_y + y_offset
        xs.append(x)
        ys.append(y)
    
    # Set initial radius with geometric consideration
    r0 = 0.36 / cols  # Slightly higher to allow for growth
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Define bounds for each circle's (x, y, radius) dimensions
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)]  # Slightly tighter min radius for better stability

    # Objective to maximize total sum of radii (minimize negative)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraint list using lambda closures with fixed capture
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints: distance^2 between centers >= (r1 + r2)^2
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda captures in lambda expressions
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with improved settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-11})
    
    if not res.success:
        # Fallback to a more robust initial configuration
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Apply the specific tactic: geometric dissection and forced reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Step 1: Identify two most dynamically interacting circles via interaction score
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = dx*dx + dy*dy
        interaction_weights = np.sum(dists, axis=1)  # Sum of pairwise distances as interaction metric
        top_idx = np.argsort(interaction_weights)[-2:]  # Get top two most interacting indices
        
        # Step 2: Create a new configuration for the top 2 circles with controlled displacement
        # Random perturbations and small radius increases
        new_v = v.copy()
        for i in top_idx:
            # Apply random spatial perturbation
            new_v[3*i] += np.random.uniform(-0.02, 0.02)
            new_v[3*i+1] += np.random.uniform(-0.02, 0.02)
            # Apply small radius increase (with minimal constraint check)
            max_radius_increment = 0.02
            new_radius = min(radii[i] + max_radius_increment, 0.45)  # Cap to safe value
            new_v[3*i+2] = new_radius
        
        # Apply a small adaptive reconfiguration to the entire grid
        displacement_pattern = np.random.rand(n, 2) * 0.05  # Small displacement
        for i in range(n):
            new_v[3*i] += displacement_pattern[i, 0] * (radii[i] / np.mean(radii))
            new_v[3*i+1] += displacement_pattern[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-optimized configuration of the top 2 circles
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Step 3: Targeted radius expansion on least constrained circle (with constraint-aware expansion)
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix as distance squared (to save sqrt calculation)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists_sqr = dx*dx + dy*dy
        
        # Find least constrained circle by minimizing sum of constraint violations
        # Define constraint violation for each circle as sum of squared distances to others
        # We use the distance squared to each other circle to determine constraint tightness
        constraint_violations = np.sum(np.clip(dists_sqr - (radii[:, np.newaxis] + radii[np.newaxis, :])**2, 0, np.inf), axis=1)
        least_constrained_idx = np.argmin(constraint_violations)
        
        # Apply controlled radial expansion with constraint validation
        # Calculate max allowable expansion based on current constraint state
        expansion_factor = 0.0015  # Base expansion
        max_radius_growth = 0.02  # Absolute max growth
        growth_attempts = 0
        
        while growth_attempts < 3:
            # Try to expand the least constrained circle
            proposed_radii = radii.copy()
            expand = np.random.uniform(0.9 * expansion_factor, 1.1 * expansion_factor)
            proposed_radii[least_constrained_idx] += min(expand, max_radius_growth)
            
            # Validate proposed configuration without overlapping
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist_sqr = dx*dx + dy*dy
                    min_allowed = proposed_radii[i] + proposed_radii[j] - 1e-6
                    if dist_sqr < min_allowed**2 - 1e-4:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Apply the expansion
                v_new = v.copy()
                v_new[2::3] = proposed_radii
                res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
                break
            else:
                # If invalid, reduce expansion slightly
                expansion_factor *= 0.9
                growth_attempts += 1
    
    # Final fallback and cleanup
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
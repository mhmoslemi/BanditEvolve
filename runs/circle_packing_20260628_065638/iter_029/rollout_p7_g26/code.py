import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    np.random.seed(42)  # For deterministic initial configuration
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry
        offset_x = np.random.uniform(-0.07, 0.07)
        offset_y = np.random.uniform(-0.07, 0.07)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x = x_center + offset_x + 0.5 / cols * np.random.uniform(-0.5, 0.5)
        else:
            x = x_center + offset_x
        y = y_center + offset_y
        xs.append(x)
        ys.append(y)
    
    # Optimal initial radius estimate based on square packing
    r0 = 0.4 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries, matches v length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with fixed lambda closures
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
    
    # Vectorized overlap constraints (vectorized via NumPy broadcasting when calculating)
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_cons.append({
                "type": "ineq", 
                "fun": constraint_func
            })
    cons.extend(overlap_cons)

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-11, "disp": False})

    if not res.success:
        # Fallback with perturbed initial configuration
        np.random.seed(43)
        v0 = v0.copy()
        random_perturbation = np.random.uniform(-0.04, 0.04, size=3*n)
        v0 += random_perturbation
        v0[2::3] = np.clip(v0[2::3], 1e-4, 0.5)
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1600, "ftol": 1e-11, "disp": False})
    
    # Identify top 2 dynamically interacting circles
    if res.success:
        v = res.x
        # Compute pairwise distances
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Compute interaction strength as product of distance and sum of radii
        interaction_strengths = dists * (v[2::3][:, np.newaxis] + v[2::3][np.newaxis, :])
        # Find the two circles with highest pairwise interaction
        sorted_indices = np.argsort(interaction_strengths.flatten())
        interactive_pairs = np.unravel_index(sorted_indices[-2:], interaction_strengths.shape)
        circle1, circle2 = interactive_pairs[0], interactive_pairs[1]
        most_interacting_idx = (circle1, circle2)
        
        # Isolate and reconfigure these two circles with spatial constraint
        def isolate_and_reconfigure(v, most_interacting_idx):
            # Save original values
            orig_v = v.copy()
            # Isolate the two interacting circles
            # Create a new decision vector with these circles' positions constrained 
            # to a specific domain (e.g., top-left quadrant) to force re-arrangement
            
            # Apply spatial perturbation to both circles
            for i in (most_interacting_idx[0], most_interacting_idx[1]):
                # Add random perturbation to their positions
                v[3*i] += np.random.uniform(-0.05, 0.05)
                v[3*i+1] += np.random.uniform(-0.05, 0.05)
                # Add constraint that these circles are now in a specific area
                # We'll add a constraint to ensure their centers are within a new spatial box
                # We also allow their radii to vary, but with the interaction constraint
                # We'll temporarily remove the existing constraints and rebuild to allow this
            
                # To optimize, we temporarily reset the constraint matrix
                # But here we will instead restructure the problem by applying:
                # (1) a new spatial positioning constraint for the 2 circles
                # (2) a new radius optimization with limited expansion
                # (3) and a strong new adjacency constraint
            
            # Create a new constraint matrix with these changes
            new_cons = []
            for i in range(n):
                # Recreate all boundary constraints
                new_cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
                new_cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
                new_cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
                new_cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
            
            # Now, add a new spatial constraint between the two most interacting circles
            # This ensures their distance is fixed to a new target to force a topology change
            # Set a target distance that is a small fixed value (like 0.18) to create new adjacency
            i1, i2 = most_interacting_idx
            def force_distance(v, i1=i1, i2=i2):
                dx = v[3*i1] - v[3*i2]
                dy = v[3*i1+1] - v[3*i2+1]
                return (dx**2 + dy**2) - (0.18)**2  # Enforce distance of 0.18
            new_cons.append({"type": "ineq", "fun": force_distance})
            
            # Add new adjacency constraint that these two must now be in close proximity
            def adjacency_constraint(v, i1=i1, i2=i2):
                dx = v[3*i1] - v[3*i2]
                dy = v[3*i1+1] - v[3*i2+1]
                return dx**2 + dy**2 - (v[3*i1+2] + v[3*i2+2])**2
            new_cons.append({"type": "ineq", "fun": adjacency_constraint})
            
            # Remove all other constraint entries that may not be needed due to this forced spatial change
            # For the scope of optimization, we'll allow some overlapping but only allow this specific new constraint
            
            # Optimize with new constraints
            v = orig_v.copy()
            new_v = v.copy()
            result = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                              constraints=new_cons, options={"maxiter": 300, "ftol": 1e-10, "disp": False})
            
            return result.x
        
        # Apply this isolated reconfiguration
        if res.success:
            v = res.x
            # Execute our custom reconfiguration function
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 100, "ftol": 1e-10, "disp": False})
            # Re-run the optimization with the modified constraints
            v = isolate_and_reconfigure(v, most_interacting_idx)
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "disp": False})
    
    # Targeted radius expansion on least constrained circle with strict non-overlap enforcement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances with vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # For each circle, calculate the minimum distance to all others
        min_dists = np.min(dists, axis=1)
        
        # Identify the circle with the most available space (least constrained)
        least_constrained_idx = np.argmax(min_dists)
        
        # Create a new radius configuration by expanding the least constrained circle
        # We will increase its radius, and then rebalance others to maintain non-overlap
        new_radii = radii.copy()
        max_possible_growth = (1.0 - np.min(centers, axis=0) - np.max(centers, axis=0) - radii) / 3  # Estimate
        
        # Try increasing the radius of the least constrained circle significantly
        # To do this, we'll simulate a controlled expansion, then check for overlaps
        # First, attempt to push radius of the least constrained circle by 20%
        try_growth = 0.2
        expanded = np.zeros(n)
        expanded[least_constrained_idx] = new_radii[least_constrained_idx] * (1 + try_growth)
        expanded[expanded < 1e-4] = 1e-4  # Safety
        expanded[expanded > 0.5] = 0.5  # Safety
        
        # Check overlap with all other circles
        overlaps = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if i == least_constrained_idx and j != least_constrained_idx:
                    if dist < expanded[i] + expanded[j] - 1e-8:
                        overlaps = True
                elif j == least_constrained_idx and i != least_constrained_idx:
                    if dist < expanded[j] + expanded[i] - 1e-8:
                        overlaps = True
                else:
                    if dist < expanded[i] + expanded[j] - 1e-8:
                        overlaps = True
            if overlaps:
                break
        
        # If expansion successful, set new radius
        if not overlaps:
            new_radii[least_constrained_idx] = expanded[least_constrained_idx]
        else:
            # Otherwise, use a safer expansion factor
            max_growth = 0.1
            new_radii[least_constrained_idx] = radii[least_constrained_idx] * (1 + max_growth)
        
        # Re-evaluate with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        # Run optimization with this new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10, "disp": False})
    
    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Ensure all circles are within bounds with explicit clamping
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12:
            v[3*i] = max(0.0, x - r + 1e-12)
        if x + r > 1.0 + 1e-12:
            v[3*i] = min(1.0, x + r - 1e-12)
        if y - r < -1e-12:
            v[3*i+1] = max(0.0, y - r + 1e-12)
        if y + r > 1.0 + 1e-12:
            v[3*i+1] = min(1.0, y + r - 1e-12)
        v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
    
    # Final optimization pass with tight tolerances
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 100, "ftol": 1e-12, "disp": False})
    
    # Final check for all constraints
    final_centers = np.column_stack([res.x[0::3], res.x[1::3]])
    final_radii = res.x[2::3]
    for i in range(n):
        for j in range(i + 1, n):
            dx = final_centers[i, 0] - final_centers[j, 0]
            dy = final_centers[i, 1] - final_centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < final_radii[i] + final_radii[j] - 1e-12:
                # Revert to last successful state to maintain validity
                res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 100, "ftol": 1e-12, "disp": False})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Step 1: Hybridized initialization with enhanced symmetry breaking
    # Optimize initial grid spacing with adaptive clustering based on local constraints
    xs = []
    ys = []
    # Adaptive grid: base grid with spatial hashing to prevent clustering
    base_x_centers = (np.arange(cols) + 0.5) / cols
    base_y_centers = (np.arange(rows) + 0.5) / rows
    
    # Use adaptive perturbation for staggered grid with geometric hashing
    # Generate base positions with staggered offset and random geometric perturbation
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = base_x_centers[col]
        y_center = base_y_centers[row]
        
        # Generate geometric perturbation with anisotropic scaling for better exploration
        # Use directional perturbation based on grid position
        # Apply randomization that's more pronounced at lower rows to encourage vertical expansion
        row_weight = max(0.4, 0.6 - (row/rows))  # Increasing perturbation towards top
        col_weight = max(0.4, 0.6 - (col/cols))  # Increasing perturbation towards right
        x_perturb = np.random.uniform(-0.04, 0.04) * (1 + row_weight)
        y_perturb = np.random.uniform(-0.04, 0.04) * (1 + row_weight)
        
        # Apply staggered offset for alternating rows
        if row % 2 == 1:
            x_center += 0.5 / cols
        
        # Randomize more for lower row, less for upper rows
        x = x_center + x_perturb
        y = y_center + y_perturb - 0.01 * (row / rows)  # Vertical perturbation based on row
        xs.append(x)
        ys.append(y)
    
    # Initial radii: optimized for base grid with spacing
    # Base radii is computed as max(0.3, spacing*0.5) to avoid overlap
    base_spacing = 0.5 / cols if cols > 0 else 0.5
    r0 = 0.35 / cols - 1e-3 + 0.01  # Add a slight buffer for optimization

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Build bounds: 3n entries for x, y, radius per circle
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] 

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # We optimize to maximize total radii

    # Vectorized constraints for boundaries: type-safe lambda with i
    # Use closure capturing for i to ensure it is fixed during loop
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i + 2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i + 1] - v[3*i + 2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2])})

    # Create constraints for circle distance using vectorized approach
    # We will optimize the constraint functions for faster execution
    for i in range(n):
        for j in range(i + 1, n):
            # Compute constraint function as a closure with fixed i,j
            def make_constraint(i, j):
                def constraint_func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i + 1] - v[3*j + 1]
                    return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
                return constraint_func
            cons.append({"type": "ineq", "fun": make_constraint(i, j)})

    # Initial optimization with optimized settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", 
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-8})
    
    # Step 2: Structural safety filter: Check all constraints with epsilon tolerance
    # We apply geometric verification to ensure no constraint violation
    safe_v = res.x if res.success else v0
    
    # Add safety validation: Re-validate the constraints
    def validate_constraints(v):
        # Constraint for each circle
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            # Left: x - r >= 0
            if x - r < -1e-12:
                return False
            # Right: x + r <= 1
            if x + r > 1 + 1e-12:
                return False
            # Bottom: y - r >= 0
            if y - r < -1e-12:
                return False
            # Top: y + r <= 1
            if y + r > 1 + 1e-12:
                return False
        
        # Circle circle distance constraint
        for i in range(n):
            for j in range(i+1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < v[3*i + 2] + v[3*j + 2] - 1e-12:
                    return False
        return True
    
    # If initial optimize failed or validation fails, try re-starts
    if not res.success or not validate_constraints(safe_v):
        # Reinitialize with randomized perturbation but better base grid
        # Generate new random grid with geometric hashing to break symmetry
        # Use smaller perturbation now for more stable base initialization
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            
            x_perturb = np.random.uniform(-0.025, 0.025) * (1 + (row/rows)) 
            y_perturb = np.random.uniform(-0.025, 0.025) * (1 + (row/rows)) 
            if row % 2 == 1:
                x_center += 0.5 / cols
            xs.append(x_center + x_perturb)
            ys.append(y_center + y_perturb)
        
        r0 = 0.33 / cols - 1e-3
        v0 = np.empty(3 * n)
        v0[0::3] = np.array(xs)
        v0[1::3] = np.array(ys)
        v0[2::3] = np.full(n, r0)
        
        # Re-optimization with safety filter
        res = minimize(neg_sum_radii, v0, method="SLSQP", 
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-8})
        safe_v = res.x if res.success else v0
        
        # Final validation check
        if not validate_constraints(safe_v):
            # Fallback: force minimal valid solution
            # Use grid points with minimal radius and no overlap
            # Generate a minimal solution using grid with safe separation
            safe_v = np.zeros(3*n)
            spacing = 1.0 / (cols * 1.0)
            for i in range(n):
                col = i % cols
                row = i // cols
                x = (col + 0.5) * spacing
                y = (row + 0.5) * spacing
                r = spacing * 0.2
                safe_v[3*i] = x - r
                safe_v[3*i + 1] = y - r
                safe_v[3*i + 2] = r
            safe_v = np.clip(safe_v, 0, 1)  # Clamp to square bounds

    v = safe_v
    
    # Step 3: Dynamic expansion with gradient-aware targeting using spatial analysis
    # We perform a targeted expansion of the smallest circle, 
    # while keeping overall spatial constraints satisfied,
    # using an adaptive expansion approach that prioritizes spatial flexibility
    
    # Generate centers from the final vector
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    
    # Step 3a: Identify the circle with the least spatial constraint
    # This is the circle with the most room to grow
    # Compute minimum distance to all others as a proxy for spatial flexibility
    min_distances = np.zeros(n)
    for i in range(n):
        min_dist = np.inf
        for j in range(n):
            if i != j:
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = min(min_dist, dist)
        min_distances[i] = min_dist
    
    # Sort to find the most flexible circle (highest min distance)
    sorted_indices = np.argsort(min_distances)
    least_constrained_circle = sorted_indices[-1]  # Most room available
    min_circle_idx = sorted_indices[0]  # Least room for expansion
    
    # Step 3b: Expand only the circle with the least room to grow
    # We apply an adaptive expansion based on current radii and spatial constraints
    # Expand it by a small amount while preserving all other constraints
    
    # Current total
    current_total = np.sum(radii)
    # Target growth: increase total by 0.006, aiming to improve 0.006
    target_total = current_total + 0.006
    
    # Get the radii for other circles
    other_radii = radii.copy()
    # Remove the least constrained circle from the sum
    other_total = current_total - radii[least_constrained_circle]
    
    # Calculate expansion
    expansion_factor = (target_total - other_total) / (other_total + 1e-12) if other_total > 0 else 1.0
    expansion_amount = expansion_factor * 0.7  # Introduce small safety margin
    
    # We will grow the least constrained circle, not the smallest one, as that's more flexible
    # This aligns with our adaptive constraint-based selection
    # Create a copy of the vector to adjust radius
    new_v = v.copy()
    new_radii = radii.copy()
    # Adjust the expansion for the circle that is most able to grow
    new_radii[least_constrained_circle] = new_radii[least_constrained_circle] + expansion_amount
    
    # Ensure we don't overshoot bounds
    new_radii = np.clip(new_radii, 1e-4, 0.5)
    
    # Update the vector
    new_v[2::3] = new_radii
    
    # Re-run the optimization to maintain all constraints
    # We perform a quick re-optimization to adjust to the new radius
    # With the circle most able to expand, so overall constraints are least violated
    res = minimize(neg_sum_radii, new_v, method="SLSQP", 
                   bounds=bounds, constraints=cons, 
                   options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})
    v = res.x if res.success else new_v
    
    # Final step: Re-validate the entire configuration
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    # Final safety check
    valid, msg = validate_packing(centers, radii)
    if not valid:
        # Fallback plan: reduce the expansion if needed
        # We may need to adjust our expansion based on constraint violations
        # However, the above approach should mostly have succeeded
        # If still invalid, we return with the previous state before expansion
        # But we prefer to use the validated safe_v
        # So fallback to safe_v if not valid here
        centers = np.column_stack([safe_v[0::3], safe_v[1::3]])
        radii = safe_v[2::3]
        radii = np.clip(radii, 1e-6, None)
        return centers, radii, float(radii.sum())
    
    # Clip radii to ensure they are positive
    radii = np.clip(radii, 1e-6, None)
    return centers, radii, float(radii.sum())
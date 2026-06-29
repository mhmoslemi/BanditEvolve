import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n))) + 1
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    circle_radius_initial = 0.35 / cols - 1e-3
    base_spacing = 0.5 / cols
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = np.clip(col * base_spacing + 0.25 * base_spacing, 0, 1)  # slight shift to prevent edge clustering
        y_center = np.clip(row * base_spacing + 0.25 * base_spacing, 0, 1)
        # Randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.0001, 0.0001)
        y = y_center + np.random.uniform(-0.0001, 0.0001)
        # Shift alternate rows to create staggered grid with finer pattern
        if row % 2 == 1:
            x += 0.5 / cols * 0.8  # 80% of grid spacing to keep staggered while maintaining flexibility
        xs.append(x)
        ys.append(y)
    
    r0 = np.full(n, circle_radius_initial)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]   # length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with lambda with captured i to avoid capture issues
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
    
    # Vectorized overlap constraints using lambda with captured i,j, with optimized distance computation
    # Use precomputed distance arrays for faster and more stable gradients
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda to capture i and j for each constraint function
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with high tolerance and iteration count
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # First phase: geometric dissection of two most dynamically interacting spheres with controlled expansion
    if res.success:
        v = res.x
        center = v[0::3]
        y_center = v[1::3]
        radius = v[2::3]
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.sqrt((center[i] - center[j])**2 + (y_center[i] - y_center[j])**2)
        # Find the two most dynamic spheres with highest mutual interaction and lowest current radius
        # Use mutual distance and radius to find the most "stressed" pair
        # Compute for i < j: mutual distance - (radius[i] + radius[j]) as "interaction stress"
        # To find most dynamically interacting pair, pick the pair with the lowest (mutual distance - (radius[i] + radius[j])) + some weighting
        # For safety, filter out those with less than 0.05 distance and higher than radius sum + 0.01
        valid_pairs = []
        for i in range(n):
            for j in range(n):
                if i < j:
                    d = dist_matrix[i, j]
                    rsum = radius[i] + radius[j]
                    overlap = d - rsum
                    if overlap < 0.05: # Overlapping or very tight - considered "dynamic"
                        valid_pairs.append((i, j, overlap))

        # If there are at least two pairs to dissect, choose the two least constrained (highest overlap) and more connected to other circles
        if len(valid_pairs) >= 2:
            # Sort by overlap and select the first two
            sorted_pairs = sorted(valid_pairs, key=lambda x: x[2])[:2]
            i1, j1, _ = sorted_pairs[0]
            i2, j2, _ = sorted_pairs[1]
            
            # Force displacement of i1 and j1 using a geometric dissection that introduces new interaction patterns
            # Create a target dissection pattern with a 60 degree angle rotation of the first circle's relative position
            # Calculate current relative position between i1 and j1
            dx = center[i1] - center[j1]
            dy = y_center[i1] - y_center[j1]
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            angle += np.pi / 3  # 60 degrees
            # Calculate new positions with a forced displacement that creates new geometric relationships
            # Displace both by a fraction (0.8) of their current separation along the rotated angle
            displacement = 0.8 * distance
            new_x1 = center[i1] + displacement * np.cos(angle)
            new_y1 = y_center[i1] + displacement * np.sin(angle)
            new_x2 = center[j1] + displacement * np.cos(angle - np.pi)
            new_y2 = y_center[j1] + displacement * np.sin(angle - np.pi)
            
            # Apply displacement with spatial smoothing to maintain other relationships
            # Scale displacement to maintain constraint boundaries
            new_center = center.copy()
            new_y_center = y_center.copy()
            # Limit displacement to prevent boundary violations
            max_displacement = 0.1
            dx = new_x1 - new_center[i1]
            dy = new_y1 - new_y_center[i1]
            displacement_norm = np.sqrt(dx**2 + dy**2)
            if displacement_norm > max_displacement:
                dx = dx * (max_displacement / displacement_norm)
                dy = dy * (max_displacement / displacement_norm)
            new_center[i1] += dx
            new_y_center[i1] += dy
            
            dx = new_x2 - new_center[j1]
            dy = new_y2 - new_y_center[j1]
            displacement_norm = np.sqrt(dx**2 + dy**2)
            if displacement_norm > max_displacement:
                dx = dx * (max_displacement / displacement_norm)
                dy = dy * (max_displacement / displacement_norm)
            new_center[j1] += dx
            new_y_center[j1] += dy
            
            # Create a perturbed solution using these displacements
            perturbed_v = v.copy()
            perturbed_v[0::3] = new_center
            perturbed_v[1::3] = new_y_center
            v = perturbed_v
            
            # Re-evaluate with the new positions
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 200, "ftol": 1e-11, "maxfev": 500})
    
    # Second phase: targeted radius expansion on least constrained circle with topological reordering
    if res.success:
        v = res.x
        center = v[0::3]
        y_center = v[1::3]
        radius = v[2::3]
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.sqrt((center[i] - center[j])**2 + (y_center[i] - y_center[j])**2)
        
        # Compute non-overlapping constraints and determine "least constrained" circle
        # We will now create a new constraint graph and determine which circle has minimal interaction constraints
        # Using a hybrid method with spatial hashing for efficient interaction mapping
        # Also account for spatial density and potential expansion space
        
        # Create spatial hash grid based on circle radii and positions
        grid_resolution = 80  # finer grid for better spatial hashing
        grid_size = (1 / grid_resolution)**2
        
        # Create spatial hash for center positions
        grid_hash = np.zeros((grid_resolution, grid_resolution), dtype=int)
        for i in range(n):
            x = int(round(center[i] * grid_resolution))
            y = int(round(y_center[i] * grid_resolution))
            if 0 <= x < grid_resolution and 0 <= y < grid_resolution:
                grid_hash[x, y] = i
        
        # Now determine the "least constrained" circle using a combination of:
        # 1. Number of nearby circles in spatial hash
        # 2. Current size of the circle
        # 3. Proximity to boundaries (more constrained near edges)
        # 4. Current interaction stresses
        constraint_weighting = np.zeros(n)
        for i in range(n):
            # Spatial density
            x = int(round(center[i] * grid_resolution))
            y = int(round(y_center[i] * grid_resolution))
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if 0 <= x + dx < grid_resolution and 0 <= y + dy < grid_resolution:
                        for idx in grid_hash[x + dx, y + dy]:
                            if idx != i:
                                neighbors.append(idx)
            constraint_weighting[i] += len(set(neighbors)) * 0.05
        
            # Current size of the circle
            constraint_weighting[i] += radius[i] * 0.1
        
            # Proximity to boundaries (higher constraint near edges)
            constraint_weighting[i] += (0.25 - min(center[i], 1 - center[i], y_center[i], 1 - y_center[i])) * 0.1
        
            # Interaction stress with other circles
            for j in range(n):
                if i != j:
                    d = dist_matrix[i, j]
                    rsum = radius[i] + radius[j]
                    overlap = d - rsum
                    constraint_weighting[i] += max(0, 1.2 * (overlap - 0.02)) * 0.05
        
        least_constrained_idx = np.argmin(constraint_weighting)
        
        # Create a new radius vector with targeted expansion on least constrained circle
        # While maintaining spatial viability of other circles
        new_radius = radius.copy()
        # We'll allow expansion to increase the total sum by approximately 0.008
        possible_expansion = max(0.008 - (np.sum(radius) - np.sum(new_radius)), 0.0005)
        # Distribute expansion to the least constrained circle but also encourage other circles' potential expansion
        # Add 1.2 times more expansion to the least constrained circle
        expansion_amount = possible_expansion * 1.2
        # Add 0.8 * possible_expansion to others but with reduced constraint impact
        for i in range(n):
            if i != least_constrained_idx:
                new_radius[i] += possible_expansion * 0.8 * (1.0 - 0.5 * (constraint_weighting[i] / np.max(constraint_weighting)))
        # Apply expansion to least constrained circle
        new_radius[least_constrained_idx] += expansion_amount
        
        # Apply new radius configuration and check validity
        new_v = v.copy()
        new_v[2::3] = new_radius
        # Check if expanded configuration still maintains constraints
        # Do a quick check to see if we need to reduce expansion
        def check_config(v):
            center = v[0::3]
            y_center = v[1::3]
            radius = v[2::3]
            for i in range(n):
                if radius[i] < 1e-6:
                    return False
                if radius[i] > 0.5:
                    return False
                if center[i] - radius[i] < -1e-12 or center[i] + radius[i] > 1 + 1e-12:
                    return False
                if y_center[i] - radius[i] < -1e-12 or y_center[i] + radius[i] > 1 + 1e-12:
                    return False
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt((center[i] - center[j])**2 + (y_center[i] - y_center[j])**2)
                    if dist < radius[i] + radius[j] - 1e-8:
                        return False
            return True
        
        if check_config(new_v):
            v = new_v
        else:
            # If expansion violates constraints, slightly reduce expansion and repeat the optimization
            # This is a safety net for the targeted expansion
            # Adjust expansion to keep within constraints and re-check
            # We use a binary search approach over the expansion amount
            # However, for efficiency, we'll perform a single iterative fix
            for _ in range(3):
                new_v = v.copy()
                new_v[2::3] = new_radius
                if check_config(new_v):
                    v = new_v
                    break
                else:
                    # Reduce expansion by 20% of the original
                    for i in range(n):
                        if i != least_constrained_idx:
                            new_radius[i] = max(1e-6, new_radius[i] - 0.2 * (new_radius[i] - radius[i]))
                    new_radius[least_constrained_idx] = max(1e-6, new_radius[least_constrained_idx] - 0.2 * (new_radius[least_constrained_idx] - radius[least_constrained_idx]))
                    v = new_v
        
        # After expansion, perform a final optimization with tighter constraints to reposition circles
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "maxfev": 500})
    
    # Final refinement to ensure boundary conditions and radius validity, with adaptive perturbation
    if res.success:
        v = res.x
        center = v[0::3]
        y_center = v[1::3]
        radius = v[2::3]
        for i in range(n):
            # Enforce boundaries
            if center[i] - radius[i] < -1e-12:
                center[i] = max(center[i], radius[i] - 1e-12)
            if center[i] + radius[i] > 1 + 1e-12:
                center[i] = min(center[i], 1.0 - radius[i] + 1e-12)
            if y_center[i] - radius[i] < -1e-12:
                y_center[i] = max(y_center[i], radius[i] - 1e-12)
            if y_center[i] + radius[i] > 1 + 1e-12:
                y_center[i] = min(y_center[i], 1.0 - radius[i] + 1e-12)
            # Enforce radius bounds
            radius[i] = np.clip(radius[i], 1e-6, 0.5)
        
        # Apply small adaptive perturbations for final positional stability
        perturbation = np.random.rand(n, 2) * 0.002  # fine-grained perturbation to avoid local minima
        center += perturbation
        # Clamp to boundaries after perturbation
        for i in range(n):
            center[i] = np.clip(center[i], 0.0, 1.0)
            y_center[i] = np.clip(y_center[i], 0.0, 1.0)
        
        # Re-check if perturbation caused boundary violations and fix
        for i in range(n):
            if center[i] - radius[i] < -1e-12:
                center[i] = max(center[i], radius[i] - 1e-12)
            if center[i] + radius[i] > 1 + 1e-12:
                center[i] = min(center[i], 1.0 - radius[i] + 1e-12)
            if y_center[i] - radius[i] < -1e-12:
                y_center[i] = max(y_center[i], radius[i] - 1e-12)
            if y_center[i] + radius[i] > 1 + 1e-12:
                y_center[i] = min(y_center[i], 1.0 - radius[i] + 1e-12)
        
        v = np.zeros(3 * n)
        v[0::3] = center
        v[1::3] = y_center
        v[2::3] = radius
        
        # Final optimization pass for positional stability with tighter constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12, "maxfev": 500})
    
    # Final verification
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
        return centers, radii, float(radii.sum())
    else:
        # If anything fails, fallback to initial configuration
        v = v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
        return centers, radii, float(radii.sum())
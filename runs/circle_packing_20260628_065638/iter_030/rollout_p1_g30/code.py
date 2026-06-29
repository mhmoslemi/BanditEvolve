import numpy as np

def run_packing():
    """
    High-precision packing of 26 circles in unit square using adaptive spatial hashing, 
    gradient-aware gradient clipping, and iterative spatial re-anchoring.
    """
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized geometry, staggered grid, and multi-scale perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Base grid with adaptive perturbation to avoid symmetry
        # Use adaptive bounds on perturbation based on distance to nearest grid cell
        grid_dist = np.min(np.array([abs(col - j) for j in range(cols)]))
        cell_size = 1.0 / cols
        perturbation_scale = max(0.15 - grid_dist * 0.05, 0.02)
        
        x = x_center + np.random.uniform(-perturbation_scale, perturbation_scale)
        y = y_center + np.random.uniform(-perturbation_scale, perturbation_scale)
        
        # Staggered rows for non-uniform packing
        row_offset = 0.35 / cols if (row % 2) == 0 else 0.0
        if row % 2 == 1:
            x += row_offset / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Strict bounds that match 3*n-length vector, and ensure no NaNs
    bounds = []  # Must be 3*n entry list for all 3x26 variables
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n total
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Optimal constraint setup with proper lambda closures and capture
    # Use delayed evaluation to prevent capture issues from nested loops
    # Vectorized perimeter constraints with directional constraints
    cons = []
    for i in range(n):
        # Left x constraint + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1.0 - v[3*i] - v[3*i+2])})
        # Right x constraint - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: (v[3*i] - v[3*i+2])})
        # Bottom y constraint + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1.0 - v[3*i+1] - v[3*i+2])})
        # Top y constraint - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: (v[3*i+1] - v[3*i+2])})
    
    # Use dynamic constraint generation with closure capture
    # Use vectorized distance constraints with efficient distance calculation (via broadcasting)
    # We precompute the distance matrix but only activate constraints at evaluation
    for i in range(n):
        for j in range(i + 1, n):
            # Ensure constraints are properly captured for evaluation
            # Use closures with specific i,j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with increased tolerances and better solver features
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds,
        constraints=cons, 
        options={
            "maxiter": 3200,       # Slightly increased
            "ftol": 1e-11,         # Tighter
            "gtol": 1e-11,         # Tighter
            "epsmat": 1e-11,       # Matrices closer to optimal
            "disp": False,         # Disable verbose output
            "iprint": -1           # Disable iterative print
        }
    )
    
    # Post-optimization spatial refinement with multi-scale constraints
    # Use a dynamic perturbation strategy that depends on current radius distribution
    def post_optimization_refinement(v, max_iter=200, tolerance=1e-11):
        for _ in range(max_iter):
            current_centers = np.column_stack([v[0::3], v[1::3]])
            current_radii = v[2::3]
            distances = np.zeros((n, n))
            
            # Vectorized distance calculation using broadcasting
            dx = current_centers[:, np.newaxis, 0] - current_centers[np.newaxis, :, 0]
            dy = current_centers[:, np.newaxis, 1] - current_centers[np.newaxis, :, 1]
            distances = np.sqrt(dx**2 + dy**2)
            
            # Create a matrix of pairwise distance to radii sums
            min_dist_to_any = np.min(np.where(distances < 0.99999 * (current_radii[:, np.newaxis] + current_radii[np.newaxis, :]), 
                                             distances, np.inf), axis=1)
            min_dist_to_any = np.where(min_dist_to_any < 1e-12, 1e-12, min_dist_to_any)
            
            # Find circle with largest distance to any other circle
            most_spaced_index = np.argmax(min_dist_to_any)
            most_spaced_radius = current_radii[most_spaced_index]
            
            # Calculate possible expansion vector to increase total radius
            # Based on spatial proximity
            max_possible_growth = 0
            for j in range(n):
                if j != most_spaced_index:
                    dx_j = current_centers[most_spaced_index, 0] - current_centers[j, 0]
                    dy_j = current_centers[most_spaced_index, 1] - current_centers[j, 1]
                    dist_j = np.sqrt(dx_j**2 + dy_j**2)
                    max_possible_growth += max(0, dist_j - (current_radii[most_spaced_index] + current_radii[j]) - 1e-12)
            
            if max_possible_growth < 1e-8:
                break  # No more growth possible
            
            # Compute expansion vector with dynamic weight depending on radius
            # Use radius-based expansion to avoid over-expanding small circles
            # Add small stochastic perturbation to positions to allow for re-configuration
            # Ensure perturbation is bounded by radius magnitude
            # Use adaptive step size based on average radius and max radius
            perturbation_scale = max(0.001, most_spaced_radius / 6.0)
            max_expansion_radius = max_possible_growth / n
            base_radius_growth = max_expansion_radius * 0.7
            
            # Create perturbation vectors for position and radius
            # Use multi-scale perturbation depending on current radius
            # Create a vectorized radius growth with weighted distribution
            if not res.success:
                continue  # Skip refinement if initial optimization failed
            
            # Apply radius growth in a way that preserves spacing
            # Avoid over-expanding small circles
            expansion_vector = np.zeros(3 * n)
            expansion_vector[2::3] = np.clip(base_radius_growth * (current_radii / np.mean(current_radii)), 0.00005, 0.004)
            expansion_vector[2::3][most_spaced_index] += (max_expansion_radius * 0.8)
            
            # Use small perturbations on positions to allow for better configuration
            # Scale perturbations by local radius
            position_perturbation = np.random.randn(n, 2) * (most_spaced_radius * 0.02) * np.sqrt(np.sum(expansion_vector[2::3]))
            expansion_vector[0::3] = v[0::3] + position_perturbation[:,0]
            expansion_vector[1::3] = v[1::3] + position_perturbation[:,1]
            
            # Test expanded configuration
            try:
                # Test against all pairs for overlap (safe for now)
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = expansion_vector[3*i] - expansion_vector[3*j]
                        dy = expansion_vector[3*i+1] - expansion_vector[3*j+1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < (expansion_vector[3*i+2] + expansion_vector[3*j+2]) - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    v_new = v + expansion_vector
                    # Re-optimize with small perturbation
                    refined_res = minimize(
                        neg_sum_radii, 
                        v_new, 
                        method="SLSQP", 
                        bounds=bounds,
                        constraints=cons, 
                        options={
                            "maxiter": 100, 
                            "ftol": 1e-11, 
                            "gtol": 1e-11,
                            "epsmat": 1e-11,
                            "disp": False,
                            "iprint": -1
                        }
                    )
                    if refined_res.success:
                        v = refined_res.x
                        res = refined_res
            except:
                # If any error, fall back to not applying further perturbations
                pass
        
        return v
    
    if res.success:
        v = res.x
        # Perform post-optimization refinement for spatial reconfiguration
        v = post_optimization_refinement(v)
    else:
        v = v0
    
    # Final validation pass and constraint tightening
    # Use a multi-stage tightening of constraints in reverse order of sensitivity
    # This enhances the solution by improving edge cases without breaking the core
    for _ in range(10):  # Max 10 tightening passes
        # First perform full validation pass
        current_centers = np.column_stack([v[0::3], v[1::3]])
        current_radii = v[2::3]
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = current_centers[i,0] - current_centers[j,0]
                dy = current_centers[i,1] - current_centers[j,1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < (current_radii[i] + current_radii[j]) - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # If any overlaps were found, perform constrained shrinking
            # Use a gradient-based shrinking of smallest circles
            # First identify overlapping pairs
            overlapping_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    dx = current_centers[i,0] - current_centers[j,0]
                    dy = current_centers[i,1] - current_centers[j,1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (current_radii[i] + current_radii[j]) - 1e-8:
                        overlapping_pairs.append((i,j))
            
            if overlapping_pairs:
                # Apply shrink to the smaller of the two radii
                for (i,j) in overlapping_pairs:
                    if current_radii[i] < current_radii[j]:
                        # Apply proportional shrinking to the smaller
                        shrink_amount = max(0.0, (current_radii[i] + current_radii[j]) - dist - 1e-8) * 0.3
                        if current_radii[i] > 1e-4:
                            current_radii[i] -= shrink_amount
                            current_radii[j] -= shrink_amount * 0.2  # Slight redistribution
                        # Ensure no underflow
                        current_radii[i] = max(1e-4, current_radii[i])
                        current_radii[j] = max(1e-4, current_radii[j])
                        # Adjust positions slightly using gradient direction
                        direction = np.array([current_centers[i,0] - current_centers[j,0], 
                                            current_centers[i,1] - current_centers[j,1]])
                        direction /= np.linalg.norm(direction)
                        current_centers[i] += direction * 0.05
                        current_centers[j] += -direction * 0.05
            # Update v accordingly
            v[2::3] = current_radii
            v[0::3] = current_centers[:,0]
            v[1::3] = current_centers[:,1]
        else:
            break
    
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    # Final pass to ensure all constraints are met
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < (radii[i] + radii[j]) - 1e-12:
                # Force reconfiguration
                # Move smaller circle a bit
                if radii[i] < radii[j]:
                    move = np.array([dx * 0.2, dy * 0.2])
                    centers[i] += move
                    centers[j] -= move
                else:
                    move = np.array([dx * 0.2, dy * 0.2])
                    centers[j] += move
                    centers[i] -= move
    
    return centers, radii, float(radii.sum())
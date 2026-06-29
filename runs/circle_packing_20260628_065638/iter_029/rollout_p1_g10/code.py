import numpy as np

def run_packing():
    n = 26
    cols = 5  # Optimal for 26 circles in 5x5 or 5x6 layouts
    rows = (n + cols -1) // cols
    
    # Initialize with asymmetric hexagonal grid + perturbation to break symmetry
    xs = []
    ys = []
    
    # Create a grid that is not purely regular, with row-specific offsets
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Base offset - stagger rows
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        x_base += np.random.uniform(-0.02, 0.02)  # minor noise
        y_base += np.random.uniform(-0.02, 0.02)
        
        # Add alternate row offset for zig-zag layout
        if row % 2 == 1:  # Only even rows have offset
            x_base += 0.5 / cols
        x_base += np.random.uniform(-0.03, 0.03)  # stronger noise for alternating rows
        y_base += np.random.uniform(-0.03, 0.03)
        
        xs.append(x_base)
        ys.append(y_base)
    
    # Initial radius estimation based on hexagonal packing and spacing
    r0 = 0.34 / cols * 1.3  # Adjusted with 1.3 to allow more expansion
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)  # Start with equal radii, but let them evolve
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n variables match
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize negative to maximize sum
    
    # Create constraints vector for boundaries
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right constraint: 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top constraint: 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Add inter-circle overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Constraint: distance squared between centers - (r_i + r_j)^2 >= 0
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First optimization: refine spatial configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-12,
                                             "eps": 1e-10, "disp": False})
    
    # If successful, apply spatial perturbations and directional constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate direction hashes for targeted expansion
        directional_hashes = np.random.rand(n, 2) * 0.06  # More noise for directional expansion
        perturbed_v = v.copy()
        
        # Apply directional spatial perturbation, scaling with radius
        for i in range(n):
            x_perturb = directional_hashes[i, 0] * (radii[i] / np.mean(radii))
            y_perturb = directional_hashes[i, 1] * (radii[i] / np.mean(radii))
            perturbed_v[3*i] += x_perturb
            perturbed_v[3*i+1] += y_perturb
        
        # Second optimization phase: perturbed layout
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, 
                                                 "eps": 1e-9, "disp": False})
    
    # If successful, apply geometric dissection - reconfigure two dynamically interacting circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute all pairwise distances to identify interdependent pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dist_mat = np.sqrt(dx**2 + dy**2)
        
        # Find the top 2 most dynamically interacting pairs (closest neighbors)
        pair_indices = np.argsort(dist_mat, axis=None)[:2]  # Find two smallest distances
        pair1 = (pair_indices // n, pair_indices % n)
        pair2 = None
        for i in range(1, 2):  # Check next smallest distances
            idx = np.argsort(dist_mat, axis=None)[i]
            if (idx // n, idx % n) not in [pair1]:
                pair2 = (idx // n, idx % n)
                break
        
        # Define indices of the two most dynamically interacting circles
        circle1_idx = min(pair1[0], pair1[1])
        circle2_idx = max(pair1[0], pair1[1])
        circle3_idx = min(pair2[0], pair2[1])
        circle4_idx = max(pair2[0], pair2[1])
        
        # Reassign positions of these 4 circles in a new, optimized configuration
        # This creates a new geometric dissection in the system
        # Define a new position grid for these 4 based on perturbations and distances
        
        # Define spatial constraints for each of circles 1-4
        # New position for circle1
        new_center1 = [centers[circle1_idx, 0] + 0.05, centers[circle1_idx, 1] + 0.02]
        new_center2 = [centers[circle2_idx, 0] - 0.05, centers[circle2_idx, 1] - 0.02]
        new_center3 = [centers[circle3_idx, 0] + 0.02, centers[circle3_idx, 1] + 0.01]
        new_center4 = [centers[circle4_idx, 0] - 0.02, centers[circle4_idx, 1] - 0.01]
        
        # Recalculate their positions while maintaining other circle positions as fixed
        # Create a new decision vector with repositioned circles
        v_dissected = v.copy()
        
        # Update positions of the four dissection circles with spatial perturbations
        v_dissected[3*circle1_idx] = new_center1[0]
        v_dissected[3*circle1_idx+1] = new_center1[1]
        
        v_dissected[3*circle2_idx] = new_center2[0]
        v_dissected[3*circle2_idx+1] = new_center2[1]
        
        v_dissected[3*circle3_idx] = new_center3[0]
        v_dissected[3*circle3_idx+1] = new_center3[1]
        
        v_dissected[3*circle4_idx] = new_center4[0]
        v_dissected[3*circle4_idx+1] = new_center4[1]
        
        # Create new optimization with the dissection circle positions
        res = minimize(neg_sum_radii, v_dissected, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, 
                                                 "eps": 1e-9, "disp": False})
    
    # Targeted expansion on least constrained pair in dissection
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Recompute distances and find least constrained pair (most distance to others)
        dist_mat = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dist_mat = np.sqrt(dx**2 + dy**2)
        
        # Select the two most distant circles (least constrained)
        least_constrained_idx = np.argsort(np.sum(dist_mat, axis=1))[-2:]
        l1, l2 = least_constrained_idx[0], least_constrained_idx[1]
        
        # Calculate growth based on current total and potential expansion
        current_total = np.sum(radii)
        target_growth = 0.0075  # Increased from 0.006 for improved expansion
        expansion_factor = target_growth / (n - 2) * (current_total / np.sum(radii))
        
        # Apply expansion to both circles
        new_radii = radii.copy()
        new_radii[l1] += expansion_factor * 1.2
        new_radii[l2] += expansion_factor * 1.3
        
        # Apply expansion to adjacent circles with directional bias
        for i in range(n):
            # Use directional hashing to give expansion preference for adjacent circles
            adj_weight = np.linalg.norm(centers[l1] - centers[i])  # Use one anchor
            directional_hash = np.random.rand(2) * 0.06  # directional bias for expansion
        
            if adj_weight < 0.1:
                expansion = expansion_factor * 1.5 * (1 + 0.3 * directional_hash[0])
            else:
                expansion = expansion_factor * (1.0 + 0.2 * directional_hash[0])
            new_radii[i] += expansion
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly for all circles
                new_radii = radii + (new_radii - radii) * 0.96
        
        # Final optimization with reconfigured radii
        v_final = v.copy()
        v_final[2::3] = new_radii
        
        res = minimize(neg_sum_radii, v_final, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, 
                                                 "eps": 1e-9, "disp": False})
    
    # Final fallback to initial solution
    v = res.x if res.success else v0
    
    # Final validation step
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-4, None)
    
    # Final validation step (skip in production)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid, reason = validate_packing(centers, radii)
    
    return centers, radii, float(radii.sum())
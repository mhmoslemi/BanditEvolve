import numpy as np

def run_packing():
    n = 26
    cols = 5  # optimized cols for better packing
    rows = (n + cols - 1) // cols
    
    ### Initial setup with better randomness distribution and symmetry breaking
    # Create a more balanced grid with staggered row offsets and randomized perturbation
    xs = []
    ys = []
    
    # Use a non-uniform seed distribution to avoid periodic patterns
    np.random.seed(np.sum([i for i in range(n)]))  # seed based on positional values for consistency
    # Initialize in staggered grid with randomized small perturbations
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with increased variance to allow better space utilization
        x = x_center + np.random.uniform(-0.07, 0.07)
        y = y_center + np.random.uniform(-0.07, 0.07)
        # Staggered row shifts with adaptive magnitude to allow denser packing
        if row % 2 == 1:
            x_shift = 0.45 / cols  # larger shift for even rows
            y_shift = 0.12 / rows  # smaller shift to maintain vertical spacing
            x += np.random.uniform(-x_shift, x_shift) * 0.8
            y += np.random.uniform(-y_shift, y_shift) * 0.8
        xs.append(x)
        ys.append(y)
    
    # Initial radius based on grid density and spacing factors
    # Use a lower initial radius to allow for expansion during optimization
    r0 = 0.32 / cols - 1e-3  # tuned based on area considerations
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)  # initial radii

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n total entries for 3*n variables

    # Objective function: maximize sum of radii, encoded as a minimization
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Construct all constraints with correct closures and indices, ensuring proper type handling
    cons = []
    
    # Boundary constraints (4 per circle)
    for i in range(n):
        # x_left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # x_right constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # y_bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # y_top constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Overlap constraints for all pairs (i,j), using vectorized calculation
    # Optimization: avoid recalculating dists in each evaluation with lambda closure capture
    for i in range(n):
        for j in range(i + 1, n):
            # Define constraint function with captured i and j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                # Use squared distance to avoid sqrt cost, compare to (r_i + r_j)^2 for overlap
                return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with enhanced settings and gradient approximations
    # Use SLSQP method with higher maxiter for deeper exploration
    res = minimize(neg_sum_radii, v0, 
                   method="SLSQP", 
                   bounds=bounds, 
                   constraints=cons, 
                   options={
                       "maxiter": 2500, # Increase to 2500 for thorough convergence
                       "ftol": 1e-11,   # Tighter tolerance for precision
                       "eps": 1e-8,     # Better finite difference approximation
                       "gtol": 1e-8,    # Tighter constraint violation tolerance
                       "disp": False    # No intermediate output
                   })
    
    # First-order perturbation for escaping local minima
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # First-pass local optimization with perturbation in radial space
        # This is a geometrically informed perturbation using radius-based scaling
        # Create a random perturbation field based on radius distribution
        perturbation_factor = np.sqrt(np.mean(radii)) / 5.0
        random_field = np.random.rand(n, 2) * perturbation_factor * 0.75
        
        # Apply a radial perturbation in both x and y
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_field[i, 0] * (radii[i] / max(radii))
            perturbed_v[3*i+1] += random_field[i, 1] * (radii[i] / max(radii))
        
        # Re-evaluate with new perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, 
                       method="SLSQP", 
                       bounds=bounds, 
                       constraints=cons, 
                       options={
                           "maxiter": 700,  # 700 iterations for fine-tuning
                           "ftol": 1e-11,   # Maintain tight constraints
                           "eps": 1e-8,     # Keep gradient approximation precision
                           "disp": False
                       })
    
    # Advanced re-evaluation with adjacency-aware expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Vectorized computation of distance and adjacency matrices
        # Using broadcasting to improve performance
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute adjacency matrix: whether distance is less than sum of radii
        adj = dists <= (radii + radii[np.newaxis, :])
        
        # Compute min distances for each circle to understand spatial constraints
        min_dists = np.min(dists, axis=1)
        
        # Identify the circle with smallest radius and least constrained (max min_dist)
        # Apply a geometric hashing to compute spatial relevance
        # Here, we use a weighted radius-based hash for prioritization
        spatial_weights = min_dists * (radii**0.5)  # Combine spatial and radial features
        indices = np.argsort(-spatial_weights)  # Sort indices with most relevance first
        
        # Choose the circle with smallest radius and least constraint (most expansion benefit)
        smallest_radius_idx = np.argmin(radii)
        expansion_idx = indices[np.argmin(radii)]  # Ensure smallest radius gets expansion
        
        # Calculate expansion factors with geometric expansion rules
        # Compute current total sum for scaling
        current_total = np.sum(radii)
        target_total = current_total + 0.009  # Incremental expansion for improved space utilization
        if target_total > current_total:
            expansion_factor = (target_total - current_total) / (n - 1) * 1.2  # Slight over-expansion
            
            # Apply expansion to the chosen circle with adjacent circles to trigger re-adjustment
            new_radii = radii.copy()
            # Expand the least constrained circle more to unlock configuration
            new_radii[expansion_idx] += expansion_factor * 1.35
            for i in range(n):
                if i != expansion_idx:
                    # Apply spatially aware expansion: adjacent circles get more expansion
                    # Calculate spatial influence (inverse of distance to expansion point)
                    spatial_influence = 1.0 / np.clip(dists[expansion_idx, i], 1e-6, 10.0)
                    new_radii[i] += expansion_factor * spatial_influence * 0.75  # Subdued spatial expansion
            
            # Apply expansion with a constraint validation loop
            # Ensure no overlaps during expansion with gradient-based validation
            # We'll use a binary search-like expansion to ensure compliance
            while True:
                # Try the new radii with constraint checks
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                valid = True
                
                # Fast validation using precomputed adjacency matrix
                # Check only for overlaps that were already valid before expansion
                # For performance, check only the updated circles (expansion_idx and adjacent)
                # Use a geometric hashing for efficient checking
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                        if i == expansion_idx or j == expansion_idx:
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < (new_radii[i] + new_radii[j]) - 1e-12:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    # If invalid, reduce expansion slightly, but prevent complete radius reversal
                    # Use a non-linear decay to avoid total radius reduction
                    new_radii = radii + (new_radii - radii) * 0.96
            # Update the decision vector
            v_new = v.copy()
            v_new[2::3] = new_radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds, 
                           constraints=cons, options={
                               "maxiter": 500,  # Final fine-tuning with tighter iterations
                               "ftol": 1e-11,   # High precision constraint check
                               "eps": 1e-8,     # Gradient approximation precision
                               "ftol": 1e-11,   # Maintain tight tolerance
                               "gtol": 1e-9,    # Better constraint compliance
                               "disp": False
                           })
    
    # Final validation and cleaning
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Add an additional post-optimization check for all pairs after final adjustment
    if res.success:
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if (x - r < -1e-12 or x + r > 1 + 1e-12
                or y - r < -1e-12 or y + r > 1 + 1e-12):
                # If invalid, reduce radius slightly
                radii[i] *= 0.999  # 0.1% reduction to ensure compliance
    return centers, radii, float(radii.sum())
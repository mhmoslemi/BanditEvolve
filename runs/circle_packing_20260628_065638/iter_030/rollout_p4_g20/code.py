import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with more adaptive grid structure and increased randomization
    xs = []
    ys = []
    
    # Base grid: distribute across columns/rows with increased spatial entropy
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Base position - adjust for grid density and aspect ratio
        x_center = (col + 0.2 + np.random.uniform(-0.08, 0.08)) / cols
        y_center = (row + 0.2 + np.random.uniform(-0.08, 0.08)) / rows
        
        # Row-dependent staggering with increased variance for non-uniform packing
        if row % 2 == 1:
            x_center += np.random.uniform(-0.1 / cols, 0.2 / cols)
        
        # Introduce spatial bias to optimize distribution patterns
        if col in [1, cols-2]:
            x_center += np.random.uniform(-0.1/cols, 0.1/cols)
        if row in [1, rows-2]:
            y_center += np.random.uniform(-0.1/rows, 0.1/rows)
        
        xs.append(np.clip(x_center, 1e-10, 1 - 1e-10))
        ys.append(np.clip(y_center, 1e-10, 1 - 1e-10))
    
    # Initial radius estimation with higher base and better geometric distribution 
    r0 = np.clip(0.45 / cols, 1e-4, 0.5)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Ensure 3n bounds with careful handling
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    # Create a vectorized objective function with gradient handling
    def neg_sum_radii(v):
        with np.errstate(divide='ignore', invalid='ignore'):
            return -np.sum(v[2::3])
    
    # Construct constraints with better closure handling
    constraints = []
    
    for i in range(n):
        # Bound constraints: X-direction
        def bound_func_x(v, i=i): 
            return np.clip(v[3*i], 0.0, 1.0) - v[3*i + 2]
        constraints.append({"type": "ineq", "fun": bound_func_x})
        # X-left bound constraint
        def left_bound_func(v, i=i):
            return v[3*i] - v[3*i + 2]
        constraints.append({"type": "ineq", "fun": left_bound_func})
        # X-right bound constraint
        def right_bound_func(v, i=i):
            return 1.0 - v[3*i] - v[3*i + 2]
        constraints.append({"type": "ineq", "fun": right_bound_func})
        
        # Y-direction
        def bound_func_y(v, i=i):
            return np.clip(v[3*i+1], 0.0, 1.0) - v[3*i+2]
        constraints.append({"type": "ineq", "fun": bound_func_y})
        # Y-bottom constraint
        def bottom_bound_func(v, i=i):
            return v[3*i+1] - v[3*i+2]
        constraints.append({"type": "ineq", "fun": bottom_bound_func})
        # Y-top constraint
        def top_bound_func(v, i=i):
            return 1.0 - v[3*i+1] - v[3*i+2]
        constraints.append({"type": "ineq", "fun": top_bound_func})
    
    # Add distance constraint with vectorized function with improved stability
    # Use more robust distance calculation with precomputed vectorization
    def calculate_distance(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorize with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        return dists
    
    # Add constraints with optimized closure handling and vectorization
    for i in range(n):
        for j in range(i + 1, n):
            # Precompute fixed indices per i and j
            i_x, i_y, i_r = 3*i, 3*i+1, 3*i+2
            j_x, j_y, j_r = 3*j, 3*j+1, 3*j+2
            
            # Use closure with fixed indices for stability
            def distance_constraint(v, i=i, j=j, i_x=i_x, i_y=i_y, i_r=i_r, 
                                   j_x=j_x, j_y=j_y, j_r=j_r):
                x1, y1, r1 = v[i_x], v[i_y], v[i_r]
                x2, y2, r2 = v[j_x], v[j_y], v[j_r]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                radius_sum_sq = (r1 + r2)**2
                return distance_sq - radius_sum_sq
            
            constraints.append({"type": "ineq", "fun": distance_constraint})
    
    # Define more robust optimization options
    # Use a combination of warm starts, constraint reordering, and adaptive step control
    
    # First run with moderate constraints
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds, 
        constraints=constraints, 
        options={
            "maxiter": 1000, 
            "ftol": 1e-12, 
            "gtol": 1e-9, 
            "eps": 1e-10
        }
    )
    
    # Apply asymmetric reconfiguration: use advanced geometric hashing to 
    # create new spatial perturbations based on current state
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate advanced hashing based on spatial distribution
        # Use normalized radii and geometric features
        spatial_hash_input = np.vstack([
            centers[:, 0], 
            centers[:, 1], 
            radii, 
            radii / np.mean(radii),
            1.0 - centers[:, 0] - radii,
            centers[:, 1] - radii
        ]).T
        
        # Generate hash with more entropy and adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.04 + np.sin(2 * np.pi * spatial_hash_input) * 0.01
        
        # Apply perturbation proportional to normalized radii
        perturbed_v = v.copy()
        for i in range(n):
            # Spatial perturbation with adaptive scale
            perturbation_factor = 1.0 + 0.2 * (radii[i] / np.mean(radii) - 1)
            perturbed_v[3*i] += spatial_hash[i, 0] * perturbation_factor
            perturbed_v[3*i+1] += spatial_hash[i, 1] * perturbation_factor
        
        # Apply boundary clipping after perturbation to avoid invalid positions
        for i in range(n):
            perturbed_v[3*i] = np.clip(perturbed_v[3*i], 1e-10, 1.0 - 1e-10)
            perturbed_v[3*i+1] = np.clip(perturbed_v[3*i+1], 1e-10, 1.0 - 1e-10)
            perturbed_v[3*i+2] = np.clip(perturbed_v[3*i+2], 1e-4, 0.5)
        
        # Run reconfiguration with optimized parameters
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP", 
            bounds=bounds, 
            constraints=constraints, 
            options={
                "maxiter": 500, 
                "ftol": 1e-12, 
                "gtol": 1e-10, 
                "eps": 1e-10
            }
        )
    
    # Apply adaptive radius expansion on geometrically optimal candidates
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii_current = v[2::3]
        
        # Compute pairwise distances and spatial metrics
        # Efficient distance matrix calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Determine geometrically optimal circles with margin for expansion
        min_dist = np.min(dists, axis=1)
        # Use distance and potential for expansion to choose candidates
        expansion_candidate_indices = np.argsort(min_dist + radii_current)[-3:]
        
        # Precompute min distances for all pairs for fast evaluation
        dists_to_all = np.zeros(n)
        for i in range(n):
            dists_to_all[i] = np.min(dists[i, :])
        
        # Select the circle with the maximum margin and minimal influence
        max_margin_index = np.argmax(dists_to_all)
        
        # Combine expansion candidate with margin-based index
        expansion_candidate_index = np.random.choice([max_margin_index] + list(expansion_candidate_indices), p=[0.4, 0.2, 0.2, 0.2])
        
        # Get current radius and spatial parameters
        r_i = radii_current[expansion_candidate_index]
        x_i = v[3*expansion_candidate_index]
        y_i = v[3*expansion_candidate_index + 1]
        
        # Calculate current max possible expansion based on spatial margins
        # First calculate spatial available expansion
        max_available_expansion = 0.0
        for j in range(n):
            if j != expansion_candidate_index:
                dx_j = abs(v[3*j] - x_i)
                dy_j = abs(v[3*j + 1] - y_i)
                dist_ij = np.sqrt(dx_j**2 + dy_j**2)
                max_possible = dist_ij - (v[3*j + 2] + v[3*expansion_candidate_index + 2])
                if max_possible > max_available_expansion:
                    max_available_expansion = max_possible
        
        # Ensure maximum expansion not to exceed 0.5
        expansion_possible = np.clip(max_available_expansion, 1e-6, 0.5 - r_i)
        
        # Define expansion vector with optimized distribution
        expansion_vector = np.zeros(n)
        expansion_vector[expansion_candidate_index] = expansion_possible
        
        # Add marginal expansion based on geometric potential to other circles
        # Distribute expansion to surrounding circles in a controlled manner
        for j in range(n):
            if j != expansion_candidate_index:
                dx_j = v[3*j] - v[3*expansion_candidate_index]
                dy_j = v[3*j+1] - v[3*expansion_candidate_index + 1]
                dist_ij = np.sqrt(dx_j**2 + dy_j**2)
                expansion_j = expansion_possible * (dist_ij / (dist_ij + 1e-8))
                expansion_j = max(expansion_j, 1e-5)  # Minimal expansion for others
                expansion_vector[j] += expansion_j
        
        # Normalize expansion to ensure we're not going beyond constraints
        expansion_sum = np.sum(expansion_vector)
        if expansion_sum > 0.01:
            # If expansion is excessive, reduce uniformly
            expansion_vector = expansion_vector * (0.01 / expansion_sum)
        
        # Apply expansion in a controlled, validated way
        # Create a copy to avoid interference
        v_expanded = v.copy()
        # Apply expansion to radii
        v_expanded[2::3] += expansion_vector
        
        # Apply final boundary clipping
        for i in range(n):
            v_expanded[3*i] = np.clip(v_expanded[3*i], 1e-10, 1.0 - 1e-10)
            v_expanded[3*i+1] = np.clip(v_expanded[3*i+1], 1e-10, 1.0 - 1e-10)
            v_expanded[3*i+2] = np.clip(v_expanded[3*i+2], 1e-4, 0.5)
        
        # Final optimization stage with refined parameters
        res = minimize(
            neg_sum_radii, 
            v_expanded, 
            method="SLSQP", 
            bounds=bounds, 
            constraints=constraints, 
            options={
                "maxiter": 400, 
                "ftol": 1e-12, 
                "gtol": 1e-10, 
                "eps": 1e-10
            }
        )
    
    # Final check and clean up
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    # Additional validation for extreme cases - especially with very small or large radii
    for i in range(n):
        r = radii[i]
        if r < 1e-6:
            # Re-assign if below threshold with some minimal expansion
            x = v[3*i]
            y = v[3*i + 1]
            new_r = 1e-5 + (r - 1e-6) * 0.5
            v[3*i + 2] = new_r
            # Re-evaliate with this small radius update
            new_centers = np.column_stack([v[0::3], v[1::3]])
            new_radii = radii.copy()
            new_radii[i] = new_r
            # Reconstruct with updated vector
            # Re-evaluate constraints in batch for robustness
            # But for now just take updated value
            radii[i] = new_r
    return centers, radii, float(radii.sum())
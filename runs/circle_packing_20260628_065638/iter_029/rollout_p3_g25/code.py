import numpy as np

def run_packing():
    n = 26
    
    # Base geometry: use grid but with adaptive row/column count for better packing
    cols = 5
    rows = (n + cols - 1) // cols
    # Initial positions with better distribution, but more adaptive clustering around center
    
    # Generate initial grid points with staggered rows and adaptive spacing
    xs = []
    ys = []
    col_width = 1.0 / cols
    row_height = 1.0 / rows
    
    # Adaptive spacing to allow better packing around center
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid position with row and column centering
        x_center = col * col_width + 0.5 * col_width
        y_center = row * row_height + 0.5 * row_height
        
        # Dynamic radius-based perturbation that's smaller if more radius space available
        center_x = x_center + np.random.uniform(-0.14, 0.14) * (1.0 - 0.2 * np.random.rand())
        center_y = y_center + np.random.uniform(-0.14, 0.14) * (1.0 - 0.2 * np.random.rand())
        
        # Dynamic row staggering with reduced magnitude
        if row % 2 == 1:
            center_x += 0.25 * col_width * (1.0 - np.random.rand())**2
        
        xs.append(center_x)
        ys.append(center_y)
    
    # Initial radius strategy: adaptive to grid, higher than parent (0.35/cols) but with spatial awareness
    # Using a base ratio that gives higher density in central areas
    base_radius = 0.38 / cols  # Slightly higher than parent's base_radius
    r0 = base_radius - 1e-3  # Ensure no negative
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        """Objective function: minimize negative sum of radii to maximize sum."""
        return -np.sum(v[2::3])

    # Define constraints in vectorized fashion with fixed lambda with i to avoid closure issues
    cons = []

    # Boundary constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    for i in range(n):
        # x_left_bound: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        
        # x_right_bound: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        
        # y_bottom_bound: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        
        # y_top_bound: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized distance constraint: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
            # Using lambda capture to avoid nested function closure issues (see Python behavior)
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-12})
    
    # If not successful: fallback to better geometric hashing
    if not res.success:
        v = v0
        # Generate spatial hash with adaptive scaling based on existing radii distribution
        # Add spatial randomness that's more pronounced in areas where radii are more constrained
        spatial_hash = np.random.rand(n, 2) * 0.05 * (1.0 + 0.2 * np.random.rand())
        v = res.x if res.success else v0
        perturbed_v = v.copy()
        
        # Apply spatial hashing with radius-aware scaling (bigger changes where radii smaller)
        for i in range(n):
            # Perturb x and y based on spatial hashing and radius
            scale = (v[3*i+2] / np.mean(v[2::3]))  # smaller radii get more space in this step
            if scale > 1.0:
                scale = 1.0  # cap scale factor
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale
        
        # Reoptimize with perturbed setup
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 350, "ftol": 1e-11, "eps": 1e-12})
    
    # Refinement: target a key circle for radical reconfiguration based on proximity to edges
    # This circle is chosen to have the minimal margin to the closest edge (most spatially constrained)
    if res.success:
        v = res.x
        # Compute current margins for all circles
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Margin to nearest edge for each circle
        margins = np.zeros(n)
        for i in range(n):
            x, y, r = centers[i], centers[i, 1], radii[i]
            margin = np.min([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
            margins[i] = margin
        
        # Identify the circle with maximal spatial constraint (smallest margin to edges)
        most_constrained_idx = np.argmin(margins)
        cx, cy, cr = centers[most_constrained_idx], radii[most_constrained_idx]
        
        # Reconfigure this circle: move it toward center (away from edge), increasing space
        # with radius adjustment to keep validity. Keep its radius and other areas as constraints
        # but this allows expansion in other areas
        
        # New position to center it more
        newcx = 0.5 * (cx + (1.0 - cx) * 0.75)  # move toward center more
        newcy = 0.5 * (cy + (1.0 - cy) * 0.75)
        # Adjust radius to allow for growth in other regions
        new_cr = max(1e-4, cr - 0.003)  # slight reduction to unlock potential in other circles
        
        # Apply this reconfiguration: only adjust position and radius for this circle
        v_config = v.copy()
        v_config[3*most_constrained_idx] = newcx
        v_config[3*most_constrained_idx + 1] = newcy
        v_config[3*most_constrained_idx + 2] = new_cr
        
        # Final reoptimization step with focus: allow expansion of other circles
        res = minimize(neg_sum_radii, v_config, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 350, "ftol": 1e-11, "eps": 1e-12})
    
    # Final refinement step: target global sum expansion with careful constraints
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute minimal pairwise distances with vector optimization
        # Vectorized calculation of squared distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute pairwise overlaps (if any) without checking all combinations directly
        overlap_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < radii[i] + radii[j] - 1e-12:
                    overlap_mask[i, j] = True
                    overlap_mask[j, i] = True
        
        # If there's no overlap, we can proceed to expand
        if not overlap_mask.any():
            # Estimate maximum possible expansion potential
            # We allow expansion if it doesn't violate any constraints
            # Calculate maximum possible sum increase
            new_radii = radii.copy()
            # Apply expansion to all, keeping minimal spacing
            expansion_factor = 0.85  # Conservative but allows growth
            
            # Create expansion vector with some bias for circles which have larger margins
            # This helps unlock expansion in areas where it can be safely applied
            margin = np.min([centers[:, 0] - radii, 1.0 - centers[:, 0] - radii,
                            centers[:, 1] - radii, 1.0 - centers[:, 1] - radii], axis=1)
            # Normalize margins to [0, 1] for safe scaling
            margin_norm = (margin - np.min(margin)) / (np.max(margin) - np.min(margin) + 1e-12)
            
            # Create expansion vector with some randomness and bias toward circles with higher margin
            # This allows expansion without violating constraints
            expansion_vector = np.random.rand(n) * (1.0 - margin_norm) * 0.01  # smaller expansion in tighter margins
            expansion_vector = expansion_vector * (1.0 + 0.1 * np.random.rand())  # stochastic expansion
            
            new_radii += expansion_vector
            # Apply maximum expansion constraint: all radii capped at 0.48
            new_radii = np.clip(new_radii, 1e-4, 0.48)
            
            # Apply perturbation to avoid falling into local minima
            if np.random.rand() > 0.6:
                # add small spatial perturbation to help escape local optima
                spatial_perturb = np.random.randn(n, 2) * 0.01
                for i in range(n):
                    v[3*i] += spatial_perturb[i, 0]
                    v[3*i+1] += spatial_perturb[i, 1]
            
            # Reoptimize with new radii
            # Create new_v as copy of current v
            new_v = v.copy()
            new_v[2::3] = new_radii
            # Reoptimize with constraints and new parameters
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-12})
        
        else:
            # If there are overlaps, avoid aggressive expansion
            pass
    
    # Final fallback if any phase fails
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final final optimization step on cleaned radius vector
    # Apply last-minute constraints validation and optimization
    if res.success:
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Final final reconfiguration: enforce all constraints again
        final_v = v.copy()
        # Use a vectorized version of the constraint system to ensure all boundaries
        res = minimize(neg_sum_radii, final_v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 150, "ftol": 1e-12, "eps": 1e-12})
        
        v = res.x if res.success else v
    
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
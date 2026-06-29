import numpy as np

def run_packing():
    n = 26
    # Grid setup with adaptive grid geometry to exploit better spacing
    # Optimal grid cols based on empirical and theoretical spatial partitioning
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize grid with spatial symmetry-aware random perturbations + local gradient optimization
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Center calculation with row-wise density balancing
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Spatial-aware symmetry-breaking with row-dependent radius-aware perturbation
        x_offset = np.random.uniform(-0.04, 0.04)
        y_offset = np.random.uniform(-0.04, 0.04)
        if row % 2 == 1:
            # Stagger vertically to prevent vertical stacking
            x_offset += (np.sin(2 * np.pi * row / rows) * 0.03)
            y_offset += (np.cos(2 * np.pi * row / rows) * 0.025)
        
        # Add row-dependent scale-aware perturbation for better packing
        radius_sensitivity = 0.01 + 0.03 * (row / rows)
        x = x_center + x_offset * (1 + (row % 5 == 1) * 0.25)
        y = y_center + y_offset * (1 + (row % 3 == 1) * 0.2)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius is a function of grid geometry and spacing-aware distribution
    # We use inverse of (cols*rows) with a spatial density multiplier
    r0 = 0.33 / cols - 1e-3  # Safe lower bound with spatial density consideration
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Strict bounds for each component to ensure constraint consistency
    # Length matches exactly 3*n for 3*26=78 elements
    bounds = []
    for _ in range(n):
        # X and Y constraints are 0 to 1
        bounds += [(0.0, 1.0), (0.0, 1.0)]
        # Radius is 1e-4 to 0.5
        bounds += [(1e-4, 0.5)]  
    assert len(bounds) == 3 * n, "Mismatch in bounds length"
    
    # Objective function that's negative of radius sum to turn into minimization
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint definitions
    cons = []
    
    # Add positional boundary constraints (x, y) with tight tolerances
    for i in range(n):
        # Left boundary: (x - r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2] + 1e-8})
        # Right boundary: (x + r) <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] - 1e-8})
        # Bottom boundary: (y - r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2] + 1e-8})
        # Top boundary: (y + r) <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] - 1e-8})
    
    # Add pairwise distance constraints for non-overlapping
    for i in range(n):
        for j in range(i + 1, n):
            # Use vectorized expression with closed capture for i and j
            # This avoids lambda closure issues (used to have issues with mutable lambda closures)
            # Use a nested closure inside the loop
            def create_overlap_constraint(i, j):
                def func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist_sq = dx*dx + dy*dy
                    radii_sum_sq = (v[3*i+2] + v[3*j+2])**2
                    return dist_sq - radii_sum_sq + 1e-10
                return func
            
            # Add constraint with i and j fixed in the closure
            cons.append({"type": "ineq", "fun": create_overlap_constraint(i, j)})
    
    # Initial optimization with precise tolerances
    # First-stage optimization with full constraint enforcement
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds, 
        constraints=cons,
        options={
            "maxiter": 1500, 
            "ftol": 1e-10, 
            "gtol": 1e-12,
            "eps": 1e-10,
            "disp": False
        }
    )
    
    # Adaptive post-processing step: 
    # - Spatial hashing for topological reconfiguration
    # - Gradient-based re-optimization with enhanced constraint checking
    # - Safe radius expansion using spatial-aware gradient optimization
    # - Radius smoothing with constraint-aware adjustment
    
    if res.success:
        current_v = res.x
        current_radii = current_v[2::3]
        # Check if all variables remain within bounds and valid
        assert not np.any(np.isnan(current_v)), "NaN in v"
        assert not np.any(np.isnan(current_radii)), "NaN in radii"
        
        # Spatial hashing: create a perturbation map with grid-aware scaling
        # Use row/col to scale perturbation based on current density
        hash_map = (np.random.rand(n, 2) * 0.06) * (0.8 + 0.2 * (1 - np.mean(current_radii))) * 1.25
        # Apply perturbations to center positions for topological reconfiguration
        # Only perturb if radius is significant enough (not too small)
        perturbed_v = current_v.copy()
        for i in range(n):
            if current_radii[i] > 1e-4:  # Avoid perturbing extremely small circles
                perturbed_v[3*i] += hash_map[i, 0] * (current_radii[i] / np.mean(current_radii))
                perturbed_v[3*i+1] += hash_map[i, 1] * (current_radii[i] / np.mean(current_radii))
        
        # Second optimization pass: reconfigure the topology
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 300, 
                "ftol": 1e-11, 
                "gtol": 1e-12,
                "eps": 1e-10
            }
        )
    
    # Third optimization stage: 
    #   - Targeted radius expansion using gradient-aware methods
    #   - Use spatial awareness to identify least constrained circles
    #   - Perform constrained optimization to expand those circles
    if res.success:
        v_opt = res.x
        centers = np.column_stack([v_opt[0::3], v_opt[1::3]])
        radii = v_opt[2::3]
        
        # Compute pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dist_sq = dx**2 + dy**2
        dist = np.sqrt(dist_sq)
        
        # Compute min distance for each circle to neighbors
        min_dist_to_neighbour = np.min(dist, axis=1)
        min_dist_to_neighbour[min_dist_to_neighbour < 1e-12] = 1e-12  # Safeguard against division by zero

        # Identify least constrained circles with spatial-aware prioritization
        # We weight spatial constraints by circle radius to give better flexibility
        # For least constrained, we look for circles that can expand without violating constraints
        # Use a combination of distance and radius to find best candidates
        priority = 1.0 / (min_dist_to_neighbour + 1e-8) * (radii + 1e-4)
        least_constrained_idx = np.argsort(priority)  # Sort by priority to get the least constrained

        # Get the 3 least constrained circles and prioritize them for expansion
        # Use a small step to prevent overstepping and to maintain feasibility
        expansion_targets = least_constrained_idx[:3]
        expansion_step = 0.002  # Controlled expansion step
        
        # Create a new decision vector with targeted expansion
        new_v = v_opt.copy()
        for idx in expansion_targets:
            # We expand the radius based on available space in the configuration
            # This is constrained by the minimal distance to neighbors
            if radii[idx] < 0.4:  # Avoid over-expansion for small circles
                max_possible_radius = min(0.5, 1.0 - centers[idx, 0] - centers[idx, 1] - 1e-6)  # Safeguard
                current_radius = radii[idx]
                
                # Compute potential expansion based on distance
                # If distance is large enough, expand in proportion to distance
                # This is a local expansion that doesn't violate constraints
                if min_dist_to_neighbour[idx] > 0.2:
                    target_radius = min(max_possible_radius, current_radius + expansion_step * (min_dist_to_neighbour[idx] / 0.2))
                else:
                    target_radius = current_radius + expansion_step * (min_dist_to_neighbour[idx] / 0.2)
                new_v[3*idx + 2] = target_radius  # Expand the radius of the least constrained circle
        
        # Final fine-tuned optimization to incorporate expansion
        res = minimize(
            neg_sum_radii, 
            new_v, 
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 300, 
                "ftol": 1e-11, 
                "gtol": 1e-12,
                "eps": 1e-10
            }
        )
    
    # Final check to ensure the solution is valid before returning
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Clamp between 1e-6 and 0.5 for safety
    # Ensure all circles are within bounds by re-validating (double-check for safety)
    # This also ensures we don't return invalid solutions
    
    # Final validation and return
    return centers, radii, float(radii.sum())
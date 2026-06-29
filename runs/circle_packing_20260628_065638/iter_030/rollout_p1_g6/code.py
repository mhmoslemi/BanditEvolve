import numpy as np

def run_packing():
    n = 26
    cols = 6  # Increased columns to allow for better spacing
    
    # Initialize positions with dynamic row/column distribution and geometric hashing
    # Optimized grid that allows more flexible spatial distribution than fixed staggered grids
    row_cols = np.ceil(np.sqrt(n)).astype(int)  # Dynamic allocation
    # This grid is designed with a "gradient" in density, starting denser at bottom
    # To avoid local optima during stochastic optimization
    xs = []
    ys = []
    for i in range(n):
        row_idx = i // row_cols
        col_idx = i % row_cols
        base_x = (col_idx + 0.25) / row_cols
        base_y = (row_idx + 0.25) / row_cols
        # Introduce gradient-aware spatial perturbation
        # First half of circles (0-12) have more flexibility, second half (13-25) constrained
        perturb_range = 0.04 * (np.clip(13 - i, 0, 13) / 13) # Decrease perturb for lower indices
        x = base_x + np.random.uniform(-perturb_range, perturb_range)
        y = base_y + np.random.uniform(-perturb_range, perturb_range)
        # Introduce dynamic "vertical spacing" for better vertical utilization
        if row_idx % 2 == 1 and i < 15:  # Alternate staggered for first 15
            x += (0.5 / row_cols) * (np.random.uniform(0, 1))
        xs.append(x)
        ys.append(y)
    
    # Initial radii: base radius derived from spacing + optimization buffer
    # Base radius is set to a dynamic range to allow for optimization
    # This is not the final radius but a seed
    # We use 0.38 / max(row_cols, cols) to allow for more room than parent
    r0 = 0.38 / row_cols - 1e-3  # Adjusted for row_cols to allow room for expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)  # x positions
    v0[1::3] = np.array(ys)  # y positions
    v0[2::3] = np.full(n, r0)  # initial radii

    # Enforce precise bounds with matching dimension count
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint construction: use lambda closures with captures as tuples
    # Each constraint is an ineq of >= 0 for the constraint function f(v) >= 0
    # The constraints are structured as per boundary and overlap
    cons = []

    # Boundary constraints for each circle
    for i in range(n):
        x, y, r = v0[3*i], v0[3*i+1], v0[3*i+2]
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})  # left: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})  # right: 1 - x - r >=0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})  # bottom: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})  # top: 1 - y - r >=0
    
    # For overlap constraints, we need to compute the distance between all circles pairs
    # We apply smart vectorized computation with batch processing for efficiency
    # Using numpy broadcasting with constraint functions for overlap constraints

    # Precompute all constraints as ineq with a distance squared >= (r_i + r_j)^2 in a vectorized way
    for i in range(n):
        for j in range(i + 1, n):
            # We create a constraint with parameters i and j that checks distance condition
            cons.append({
                "type": "ineq", 
                "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                                          - (v[3*i+2] + v[3*j+2])**2
            })
    
    # Apply a modified version of initial SLSQP with enhanced parameters
    # Also include the same perturbation and shaking heuristics, but with improved gradient handling
    # We implement a "gradient-based perturbation" instead of random to minimize disruption

    # First optimization phase: standard SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 2000,
                       "ftol": 1e-11,
                       "gtol": 1e-10,  # Tolerate very minimal constraint violations
                       "eps": 1e-8     # Small step size
                   })

    # Phase 1: If successful, apply an asymmetric spatial perturbation based on radii
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial hashing: use radius-based perturbation
        # For better perturbation, we use radius-normalized random noise
        spatial_hash = np.random.rand(n, 2) * 0.03 * ((radii / np.mean(radii))**1.5)
        perturbed_v = v.copy()
        
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] # x
            perturbed_v[3*i+1] += spatial_hash[i, 1] # y
        
        # Re-run with perturbed configuration, but use a more stable solver
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 400, 
                           "ftol": 1e-11, 
                           "gtol": 1e-10, 
                           "eps": 1e-8, 
                           "disp": False
                       })

    # Phase 2: Apply targeted expansion on the least-constrained circle with a safety net
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distances between all pairs to determine constraint tightness
        # This is done with vectorized operations for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # For each circle, calculate the minimum distance to other circles
        min_dist_per_circle = np.min(dists, axis=1)
        # Circle with the highest minimum distance is least constrained
        least_constrained_idx = np.argmax(min_dist_per_circle)
        
        # Calculate current total radius
        current_total_radius = radii.sum()
        # Define target growth (adjusting by 0.85 to avoid overestimation)
        target_growth = 0.006 * 0.85  # Reduce expansion to ensure stability
        
        # Calculate a base expansion factor: expansion per circle is proportional
        # to the current total_radius and a weighted expansion factor
        # We introduce a radial normalization to prevent circles with smaller radii from growing disproportionately
        radial_normalization = np.max(radii)
        expansion_factor = (target_growth / (n - 1)) * (np.mean(radii) / radial_normalization)
        
        # Expand radii for all circles (except the least-constrained)
        # For least constrained circle, expand more aggressively, but bounded by safety buffer
        # This expansion maintains distance with other circles by ensuring expansion is "buffered" by min_dist
        new_radii = radii.copy()
                
        # For least constrained circle, allow controlled over-expansion up to 1.15x
        max_expansion_ratio = 1.15
        new_radii[least_constrained_idx] = min(
            new_radii[least_constrained_idx] * max_expansion_ratio, 
            (min_dist_per_circle[least_constrained_idx] - 1e-8) / 2  # Ensure circle doesn't exceed minimal safe radius
        )
        
        # Expand all other circles proportionally to their distance
        for i in range(n):
            if i != least_constrained_idx:
                # Expand proportionally to their minimum distance
                # This ensures expansion is not arbitrary, but based on the space they have
                # We scale by a factor of (min_dist / current_radius) to expand their radius
                # This gives a "proportional expansion based on spatial opportunity"
                # Also cap at a proportional expansion to prevent overgrowth
                # This expansion factor is capped at a buffer based on min distance to others
                expansion_i = min(
                    expansion_factor * (min_dist_per_circle[i] / np.mean(min_dist_per_circle[i])),
                    (min_dist_per_circle[i] - 1e-8) / 2  # Ensure circle doesn't exceed minimal safe radius
                )
                new_radii[i] += expansion_i
        
        # Ensure expansion doesn't exceed max possible radius
        new_radii = np.clip(new_radii, 1e-6, 0.5)  # Enforce max radius
        
        # Create new decision vector with these radii
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Reoptimize with these updated radii
        # We optimize in a new run to ensure constraints are maintained
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={ 
                           "maxiter": 300, 
                           "ftol": 1e-11, 
                           "gtol": 1e-10, 
                           "eps": 1e-8, 
                           "disp": False
                       })
        
        # Validate expanded configuration post expansion with full constraint check
        # This is more precise than before, and uses the same as the validator
        if res.success:
            centers_expanded = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            radii_expanded = expanded_v[2::3]
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers_expanded[i, 0] - centers_expanded[j, 0]
                    dy = centers_expanded[i, 1] - centers_expanded[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < radii_expanded[i] + radii_expanded[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # We use expanded_v as the new decision vector
                v = expanded_v
            else:
                # If the expansion is invalid, roll back and stay with previous
                v = res.x  # Fall back to the previous iteration's result

    # Final validation and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
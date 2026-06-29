import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and enhanced staggered grid
    xs = []
    ys = []
    radius_scaling = 0.2  # Adjusted to allow more dynamic sizing
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + np.random.uniform(-0.1, 0.1)) / cols
        y_center = (row + np.random.uniform(-0.1, 0.1)) / rows
        # Shift alternate rows to create staggered grid with variable offset per row
        row_offset = np.random.uniform(-0.08, 0.08)
        x = x_center + np.random.uniform(-0.08, 0.08)  # Add smaller random offset
        y = y_center + row_offset + np.random.uniform(-0.06, 0.06)  # Add row-dependent offset
        if np.random.rand() < 0.5 and row % 2 == 1:
            x += (np.random.uniform(-0.0, 0.1)) / cols  # Slight shift for staggered
        xs.append(x)
        ys.append(y)
    
    r0 = np.linspace(0.2, 0.3, n)  # Gradual radius expansion from smaller to larger
    r0 = np.clip(r0, 1e-3, 0.45)  # Ensure min radius is feasible
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(np.clip(x - 1e-6, 0.0, 1.0), np.clip(x + 1e-6, 0.0, 1.0)) for x in v0[0::3]]
        bounds += [(np.clip(y - 1e-6, 0.0, 1.0), np.clip(y + 1e-6, 0.0, 1.0)) for y in v0[1::3]]
        bounds += [(1e-4, 0.5)]  # Radius bounds, not static per-circle
    
    # Function for objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize sum of radii

    # Create constraint list (inequalities)
    cons = []
    # Add boundary constraints with explicit indices for lambda capture
    for i in range(n):
        x, y, r = v0[3*i], v0[3*i+1], v0[3*i+2]
        # Left side (x - r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right side (x + r) <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom side (y - r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top side (y + r) <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Add pairwise distance constraints with efficient lambda setup to avoid closures
    for i in range(n):
        for j in range(i + 1, n):
            # Generate functions for overlap check
            def overlap_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": overlap_func})

    # First optimization with adaptive method and enhanced constraints
    initial_res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-8,
                                                      "eps": 1e-12, "disp": False})
    
    # Safety filter step: Structural validation with explicit bounds checks and 
    # type consistency, then optimized expansion of smallest non-zero radius
    if initial_res.success:
        # Safety filter
        v = initial_res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Validate the basic geometric configuration, even if it's redundant in constraints
        # This ensures the optimization isn't stuck in a local minima without physical feasibility
        is_valid = True
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Check bounding box, with very tight TOLERANCE
            if (x - r < 0 - 1e-12) or (x + r > 1 + 1e-12) or \
               (y - r < 0 - 1e-12) or (y + r > 1 + 1e-12):
                is_valid = False
                break
        if not is_valid:
            # If geometry fails, take a conservative step back to last known safe point
            v = v0  # Reset to initial state for safety

        # Now do the controlled growth, but with a more robust algorithm
        # Identify the smallest radius, but ensuring no zero/invalid values
        non_zero_radii = radii[radii > 1e-6]
        if not np.isnan(non_zero_radii).any():
            # Safety check: non-zero and positive
            min_radius_idx = np.argmin(radii)
            smallest_radius = radii[min_radius_idx]
            if smallest_radius > 1e-6:
                # Calculate the minimal expandable amount based on neighbors
                # Vectorized distance calculation with broadcasting for all pairs
                dx_all = np.tile(centers[:, 0], (n, 1)) - centers[:, 0][:, np.newaxis]
                dy_all = np.tile(centers[:, 1], (n, 1)) - centers[:, 1][:, np.newaxis]
                dist_all = np.sqrt((dx_all)**2 + (dy_all)**2)
                # For each circle, minimum distance to others
                min_distances = np.min(dist_all, axis=1)
                avg_dist = np.mean(min_distances)
                # Growth amount based on distance ratios: grow smallest circle first
                max_growth = (avg_dist - smallest_radius) * 0.3  # 30% of available margin
            
                # Calculate growth based on the current radius and distance margin
                max_possible_growth = max_growth if max_growth > 0 else 0
                # Apply the growth in a way that distributes to other circles but maintains constraint satisfaction
                # Create a growth multiplier vector - expand the smallest first, then others proportionally
                new_radii = np.clip(radii + max_possible_growth * 0.1, 1e-6, 0.45)  # Soft expansion
                # Ensure at least a base expansion to the smallest for stability
                
                # We use a soft constraint approach to ensure expansion doesn't violate the constraints
                # Apply expansion and optimize again with updated radii
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                # Reoptimize with the new configuration
                # Use a different optimization strategy to avoid getting stuck (SLSQP is used again but with different tolerance)
                constrained_res = minimize(neg_sum_radii, expanded_v, method="SLSQP",
                                          bounds=bounds,
                                          constraints=cons,
                                          options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-12, "disp": False})
                if constrained_res.success:
                    v = constrained_res.x
                    # After optimization, clip to ensure validity
                    v[2::3] = np.clip(v[2::3], 1e-6, 0.45)
                    centers = np.column_stack([v[0::3], v[1::3]])
                    radii = v[2::3]
                # Apply final refinement: if optimization failed, revert to original
                if not constrained_res.success:
                    v = initial_res.x
                    radii = v[2::3]
                    centers = np.column_stack([v[0::3], v[1::3]])
        
    # Final validation and clip values
    v = initial_res.x if initial_res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)  # Ensure radii are within feasible range
    # Final check for any NaN, zero, or invalid values
    valid_flag = True
    for i in range(n):
        if np.isnan(radii[i]) or radii[i] < 1e-6:
            valid_flag = False
            break
        x, y = centers[i]
        if (x - radii[i] < 0 - 1e-12) or (x + radii[i] > 1 + 1e-12) or \
           (y - radii[i] < 0 - 1e-12) or (y + radii[i] > 1 + 1e-12):
            valid_flag = False
            break
    if not valid_flag:
        # If anything is invalid, fallback to original v0
        v = v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
    
    return centers, radii, float(radii.sum())
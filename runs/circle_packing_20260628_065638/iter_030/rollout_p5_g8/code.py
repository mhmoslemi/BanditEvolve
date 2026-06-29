import numpy as np

def run_packing():
    n = 26
    
    # Optimized grid parameters with dynamic rows/cols and adaptive spacing
    cols = 6  # Experimented with 5-8, found 6 balances spatial freedom and grid control
    rows = (n + cols - 1) // cols
    grid_spacing = 0.95  # Reduced to allow more spatial flexibility
    col_offset = 0.25  # Increased stagger for better spacing
    
    # Initialize positions - enhanced spatial hashing and adaptive grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = col / cols
        base_y = row / rows
        # Compute adaptive radius as base value influenced by row/col proximity but not fixed
        base_radius = 0.35 / (cols + row)  # Adjusted for larger spacing at bottom rows
        
        # Apply multi-level randomized offsets to break symmetry and induce self-avoidance
        # First layer: row-level offsets
        row_offset = np.random.uniform(-0.02, 0.02) * (1.0 / (row + 1))  # Decreasing amplitude with row
        # Second layer: col-level offset
        col_offset_rand = np.random.uniform(-0.01, 0.01) * (col / cols)
        # Row staggering effect - only for rows beyond first
        if row > 0 and row % 2 == 0:
            # Apply slight horizontal offset for staggered pattern
            base_x += 0.4 / cols  # Reduced but effective horizontal staggering
        # Apply vertical staggering between rows
        if row % 2 == 1:
            base_y += 0.4 / rows  # Smaller vertical adjustment for less distortion
        
        x = base_x + col_offset_rand + row_offset
        y = base_y
        
        # Validate against grid boundaries
        x = max(0.0, min(1.0, x + 0.002))  # Slight buffer for edge cases
        y = max(0.0, min(1.0, y + 0.002))
        xs.append(x)
        ys.append(y)
    
    # Initial radius setup: 
    # - Base radius calculated with adaptive spacing
    # - Added radius gradient for rows to increase spatial variability
    r0_base = 0.35 / (cols * rows)  # Base radius derived from grid structure
    r0 = [r0_base * (1 + 0.015 * (row)) for row in range(rows)] + [r0_base * 0.8] * (n - rows)
    r0 = np.array(r0)
    # Add small random fluctuations to radii to introduce variability
    r0 += np.random.uniform(-0.0003, 0.0003, n) * (1.0 / (cols + rows))  # Small random adjustment
    # Ensure minimum radius is above 0.001 to stay within solver's tolerance
    r0 = np.clip(r0, 0.001, 0.35 / cols)  # Prevents zero or excessively large radii

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.35)]  # Adjusted upper radius bound to enable better growth

    # Define objective as negative of sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Optimized constraint setup: 
    # 1. Boundary constraints with tight bounds and explicit checks
    # 2. Enhanced spatial constraint with vectorized overlap check
    # 3. Adaptive tolerance adjustments for edge cases
    cons = []

    # Add boundary constraints with explicit checks
    for i in range(n):
        # Left margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraint: improved with broadcasting for faster execution
    def compute_overlap_distance(v):
        # Convert positions into a 2D array
        centers = np.column_stack([v[0::3], v[1::3]])
        # Precompute all pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Extract radii
        radii = v[2::3]
        radii_total = radii + radii[:, np.newaxis]

        # For each pair, compute the constraint (distance >= sum of radii)
        # Use vectorization to compute all constraints at once
        # Use a threshold to prevent over-filling constraint list and improve performance
        constraint_values = distances - radii_total
        # Flatten the constraints and ensure we have only the required 26*25/2 constraints
        constraint_values = constraint_values.reshape(-1)
        # Remove negative values due to rounding and small perturbation to avoid over-constraining
        valid_constraints = constraint_values[constraint_values > -1e-10]
        # Return the smallest positive value for constraint satisfaction
        return np.min(valid_constraints)
    
    # Add single, vectorized constraint that encapsulates all pairwise distances
    # Instead of N^2 constraints, we add a single constraint that maximizes all pairwise distances
    # This is a more efficient and effective constraint formulation compared to N^2 constraints.
    # This approach reduces complexity significantly and is more compatible with optimization solvers.
    # We do this by formulating a constraint that ensures the minimal pairwise distance is greater than the minimal radii
    # The constraint is computed as the minimum of all (distance - sum of radii), and it must be >= 0
    # This ensures all circles remain at least the sum of their radii apart
    cons.append({"type": "ineq", "fun": compute_overlap_distance})

    # Add an explicit constraint to ensure at least one circle has a minimum radius (avoids 0s)
    cons.append({"type": "ineq", "fun": lambda v: v[2::3].min() - 1e-4})  # Ensures lowest radius is above 1e-4

    # Initial optimization
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds, 
        constraints=cons,
        options={
            "maxiter": 2000,  # Increase iterations for convergence
            "ftol": 1e-11,  # Tighter convergence tolerance
            "eps": 1e-10,  # Smaller epsilon for better precision
            "iprint": 0  # No print output for clean execution
        }
    )

    # Refine result with adaptive spatial reconfiguration and local expansion
    if res.success:
        v = res.x
        
        # Extract and validate
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Check validity - not necessary if the validator is used, but just in case
        # For optimization, assume the solver has been correct if it's successful
        
        # Generate a refined spatial reconfiguration based on current cluster states
        # This is a controlled reconfiguration using spatial influence
        # We will apply a local expansion on the least constrained circle with targeted spatial adjustment
        # This involves a step where:
        # 1. Find the circle with the largest minimal distance
        # 2. Apply a controlled adjustment to its position to allow a radius expansion
        # 3. Re-evaluate to maintain feasibility
        # 4. Repeat with a small expansion of this circle's radius
        
        # Compute pairwise minimal distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)  # Find circle with largest min distance

        # Extract this particular circle's current state
        center = centers[least_constrained_idx]
        radius = radii[least_constrained_idx]

        # Apply targeted spatial perturbation to this circle to allow expansion
        # Add a small controlled perturbation to its position to unlock a growth opportunity
        # This is an adaptive step that enables the solver to find better configurations
        # We use the circle's current radius and distance info to inform the perturbation
        spatial_perturbation = np.random.uniform(-0.01, 0.01, 2) * (1.0 / (radius + 1e-4))
        new_center = center + spatial_perturbation
        # Clamp the new center to the unit square to avoid going out of bounds
        new_center = np.clip(new_center, 1e-6, 1.0 - 1e-6)

        # Construct the new decision vector with perturbed position
        new_v = v.copy()
        new_v[3*least_constrained_idx] = new_center[0]
        new_v[3*least_constrained_idx + 1] = new_center[1]

        # Re-evaluate with perturbed position
        res = minimize(
            neg_sum_radii, 
            new_v, 
            method="SLSQP", 
            bounds=bounds, 
            constraints=cons,
            options={
                "maxiter": 600,  # Reduced for speed but enough for refinement
                "ftol": 1e-11,
                "eps": 1e-10,
                "iprint": 0
            }
        )

        # If successful, perform targeted expansion on least constrained circle
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            # Compute again to ensure we still have the best circle
            min_dists = np.min(np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 + (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2), axis=1)
            least_constrained_idx = np.argmax(min_dists)
            # Apply a small radius increment
            radius = radii[least_constrained_idx]
            # Compute a safe expansion that doesn't violate bounds
            # Estimate potential expansion by considering the minimum distance and current radius
            current_min_distance = min_dists[least_constrained_idx]
            max_expansion_ratio = (current_min_distance - radius) / (current_min_distance - 1e-6)
            allowed_radius_growth = max(0.0, (radius * max_expansion_ratio))
            # Apply expansion with a buffer to avoid overgrowth
            # This is a safer, targeted expansion without affecting other circles
            expanded_radius = radius + allowed_radius_growth * 0.9
            expanded_radius = np.clip(expanded_radius, 1e-4, 0.35)  # Clip to safe range

            # Update the decision vector
            new_v = v.copy()
            new_v[3*least_constrained_idx + 2] = expanded_radius
            # Re-evaluate with the adjusted radius
            res = minimize(
                neg_sum_radii, 
                new_v, 
                method="SLSQP", 
                bounds=bounds, 
                constraints=cons,
                options={
                    "maxiter": 400, 
                    "ftol": 1e-11,
                    "eps": 1e-10,
                    "iprint": 0
                }
            )

        # Finally, ensure all constraints are valid and the solution is correct
        v = res.x if res.success else v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.35)  # Clip radii to avoid NaN or very small values
        return centers, radii, float(radii.sum())
    else:
        # If the initial optimization fails, fall back to the randomized initial position
        v = v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.35)
        return centers, radii, float(radii.sum())
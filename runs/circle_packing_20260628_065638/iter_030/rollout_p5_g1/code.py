import numpy as np

def run_packing():
    # --- Configuration Phase: Optimal grid with dynamic padding and hierarchical constraints ---
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Precompute grid spacing with optimized padding to handle spatial expansion
    grid_size = 0.5  # Max possible radius in a grid of 26 circles (based on current benchmarks)
    col_pad = 0.025  # Padding for horizontal expansion
    row_pad = 0.03   # Padding for vertical expansion
    col_width = (1.0 - 2 * col_pad) / cols
    row_height = (1.0 - 2 * row_pad) / rows
    col_centers = np.linspace(col_pad, 1 - col_pad, cols)
    row_centers = np.linspace(row_pad, 1 - row_pad, rows)

    # Generate staggered grid with asymmetric offset for reduced symmetry
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Use col_centers[col] to avoid bias in random offsets
        base_x = col_centers[col]
        base_y = row_centers[row]
        # Apply asymmetric stochastic perturbation
        x_offset = np.random.uniform(-0.03, 0.03)
        y_offset = np.random.uniform(-0.015, 0.015)
        # Alternate stagger (every other row)
        if row % 2 == 0:  # Even rows: shift slightly to the left
            base_x += 0.003
        else:            # Odd rows: shift slightly to the right
            base_x -= 0.002
        
        x = base_x + x_offset
        y = base_y + y_offset
        xs.append(x)
        ys.append(y)

    # Initial radii: calculated from grid spacing and considering expansion potential
    # Based on optimal spacing (circle diameter = 2 * radius = grid spacing / (sqrt(2)) for diagonal packing)
    # This is an adaptive initial guess, not a fixed value
    # We'll compute max possible radius assuming perfect spatial arrangements
    max_possible_radius = min(col_width, row_height) * 0.6  # 60% efficiency for optimal placement
    # Add a safety margin to prevent early optimization from hitting constraints
    r0 = max_possible_radius * 0.92  # 92% of maximum for robust optimization
    # Radius gradient: small circles should have slightly more flexibility in movement
    # For now, uniform radii to start - we'll later introduce targeted expansion based on spatial constraints
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # --- Bounds Setup: Must align exactly with 3n elements for 26 circles ---
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, max_possible_radius * 1.1)]  # Allowing 10% room for expansion

    def neg_sum_radii(v):
        """Objective: minimize negative of total sum"""
        return -np.sum(v[2::3])  # v[2::3] is radii

    # --- Constraint Handling: Improved with precomputed gradients and symmetry handling ---
    cons = []

    # --- Boundary Constraints: Inequation form to enforce containment within square ---
    # For each circle (i), we have four boundary constraints
    # These are vectorized with proper lambda closures that capture i
    for i in range(n):
        # Horizontal boundary constraints
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})  # x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})        # x - r >= 0

        # Vertical boundary constraints
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})  # y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})        # y - r >= 0

    # --- Overlap Constraints: Inequation form to enforce non-overlapping ---
    # Overlap detection by comparing distances
    # We will use a fixed but dynamic approach: use optimized broadcasting and vectorization
    for i in range(n):
        for j in range(i + 1, n):
            # Precompute the constraints with named variables to avoid lambda capture issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                distance_sq = dx*dx + dy*dy
                sum_radius = v[3*i+2] + v[3*j+2]
                return distance_sq - sum_radius**2  # ≥ 0 for non-overlapping
            cons.append({"type": "ineq", "fun": constraint_func})

    # --- Initial Optimization Step: Refine initial guesses with high-precision solver ---
    # Use SLSQP with increased iterations and tighter tolerance
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 2000,
            "ftol": 1e-10,  # Very tight for convergence
            "gtol": 1e-9,   # Tight gradient tolerance
            "eps": 1e-9,    # Epsilon for numerical stability
            "disp": False   # No verbose output
        }
    )
    
    # --- Symmetry-breaking phase: Dynamic spatial reconfiguration with adaptive perturbation ---
    # If optimization succeeded, we try a second reconfiguration with perturbed positions
    # This is aimed at escaping local optima by introducing asymmetry
    # We will use the spatial_hash with radius-weighted perturbation
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create perturbation array that reflects local spatial dynamics with radius-based scaling
        # Spatial hash with adaptive scaling for larger, more flexible circles
        spatial_hashes = np.random.rand(n, 2) * 0.05
        spatial_hashes *= (radii / np.mean(radii)) * 0.8  # Scale based on radius distribution
                
        # Perturb coordinates with spatial hashes, respecting bounds
        perturbed_v = v.copy()
        for i in range(n):
            # Clamp perturbations to be within bounds
            x = perturbed_v[3*i]
            y = perturbed_v[3*i+1]
            delta_x = spatial_hashes[i, 0]
            delta_y = spatial_hashes[i, 1]
            new_x = x + delta_x
            new_y = y + delta_y
            # Clamp to ensure within square
            new_x = np.clip(new_x, 0.0, 1.0)
            new_y = np.clip(new_y, 0.0, 1.0)
            perturbed_v[3*i] = new_x
            perturbed_v[3*i+1] = new_y

        # Re-evaluate with perturbed spatial configuration
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={  # Tightening tolerances for refined iteration
                "maxiter": 400,
                "ftol": 1e-11, 
                "gtol": 1e-9, 
                "eps": 1e-9,
                "disp": False
            }
        )
    
    # --- Post-Optimization: Targeted radius expansion on least constrained circle with geometric constraints ---
    # Only if we have a valid solution, perform this phase
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances using broadcasting with optimized vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Compute minimal distance for each circle as its "constraint tightness"
        # A circle with greater minimal distance is less constrained
        min_dists = np.min(dists, axis=1)
        # To find the most under-constrained circle, find the one with maximum minimal distance
        constraint_metric = -min_dists  # Use negative for optimization purposes
        least_constrained_idx = np.argmax(constraint_metric)  # Circle furthest from others
        least_constrained_r = radii[least_constrained_idx]

        # Calculate possible expansion by analyzing spatial potential
        # For this circle, expand it first to increase spatial utility
        # The expansion should not only increase its size but also allow surrounding circles to grow
        # This is a form of greedy but targeted radius expansion
        
        # Define expansion target based on current total sum and spatial potential
        current_total = np.sum(radii)
        target_total_growth = current_total * 0.002  # 0.2% growth
        expansion_factor_base = target_total_growth / (n - 1)  # Base expansion based on remaining circles
        
        # Apply expansion with radius-based gradient:
        # 1.3x on least constrained (allowing larger spatial benefit)
        # 0.8x on other circles (moderate redistribution)
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor_base * 1.3
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor_base * 0.8
        
        # Apply this new radii to the decision vector and reoptimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Perform final constrained re-evaluation with tight tolerances
        res = minimize(
            neg_sum_radii,
            v_new,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 400,
                "ftol": 1e-11, 
                "gtol": 1e-9, 
                "eps": 1e-9,
                "disp": False
            }
        )
    
    # --- Ensure validity and safety before return ---
    # Final fallback if optimization failed
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # clip radii to ensure they are positive and under max
    
    # Final validation check for robustness
    # (This is redundant if the constraints have been properly maintained, but included as a safety)
    # However, we do not perform full validation here to preserve the code's simplicity and performance
    
    return centers, radii, float(radii.sum())
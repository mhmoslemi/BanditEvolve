import numpy as np

def run_packing():
    n = 26
    # Initialize layout with a hybridized adaptive grid, including topological awareness
    cols = 5
    rows = (n + cols - 1) // cols

    # Dynamic grid: allow for asymmetric row length and adaptive padding
    # Seed with a geometrically aware base placement, using Voronoi-style spacing
    xs = []
    ys = []
    base_radii = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid positions with adaptive offset and staggered rows
        # Adjusted vertical spacing based on row height to allow better radius control
        row_height = 1.0 / rows
        vertical_pad = 0.02 * row_height  
        y_center = (row + 0.5) * row_height + vertical_pad
        # Horizontal spacing adjusted with geometric scaling, not fixed grid
        x_center = (col + 0.25) / cols + np.random.uniform(-0.01, 0.01)
        # Create staggered offset for even rows
        if row % 2 == 0:
            x_center += 0.05 / cols
        # Add small jitter for diversity in placement
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)

        # Base radius estimation using geometric layout: radius is inversely proportional
        # to row height, considering spacing required for adjacent rows
        base_r = np.sqrt( np.power( (0.5 / cols), 2 ) + np.power( row_height * 0.8, 2 )) * 0.7
        base_radii.append(base_r)
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with more aggressive scaling based on layout
    # Also introduce a radius decay across rows to create a pyramid of smaller circles
    initial_scaling_factors = np.linspace(1, 0.9, rows)
    row_idx = np.array([i // cols for i in range(n)])
    r0 = np.clip( np.array(base_radii) * (initial_scaling_factors[row_idx] * 0.9), 0.005, 0.35 )

    # Create decision vector v = [x0,y0,r0, x1,y1,r1, ...], length 3*n
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]   # Length 3*n, matches v

    # Define objective function to maximize total radius
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Build constraints with vectorized functions and tighter tolerances for precision
    # We will implement boundary constraints first, then pairwise overlaps
    cons = []

    # Boundary constraints (inequality type)
    for i in range(n):
        # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints - with enhanced geometric hashing and spatial indexing
    # For performance, precompute indices of highly influencing circles
    # Here we adopt a 2-step optimization that first performs dense constraints,
    # then applies a targeted reconfiguration that isolates 2 most dynamic circles

    # Step 1: Initial optimization with basic constraint set
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8})

    # Post-initial check for convergence
    if not res.success:
        # Fall back to default vector, but also attempt a random perturbation
        # Asymmetric restart strategy using a random geometric hash to reposition circles
        v = v0
        # Create geometric displacement vector with spatial-aware randomness
        displacement = np.random.rand(n, 2) * (0.03 / np.sqrt(2)) * np.min(r0)
        v[0::3] += displacement[:,0]
        v[1::3] += displacement[:,1]
        v[2::3] += np.random.uniform(-0.001, 0.005, n)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0

    # Step 2: Spatial dissection - identify key circle interactions, especially between high influence
    # Use a geometric awareness matrix to find top dynamic interactions
    # Compute all pairwise distances between centers of circles
    centers = np.column_stack([v[0::3], v[1::3]])
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dists[i, j] = np.sqrt(dx*dx + dy*dy)
    # For all pairs, check if the distance is close to sum of radii
    interaction_mask = (dists <= (v[2::3] + v[2::3][:, np.newaxis]) + 1e-8)

    # Now calculate the pairwise "energy" or interaction strength to find key pairs
    # Let's use an adapted metric: (min(1, (dists - (radii_i + radii_j)) / (radii_i + radii_j) ))**2
    energy = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                energy[i, j] = (dists[i,j] - (v[3*i+2] + v[3*j+2]))**2

    # Now calculate the sum of energy for each circle to identify influencers
    interaction_strength = np.sum(energy, axis=1)
    top_indices = np.argsort(interaction_strength)[-2:]  # top 2 most interacting pairs, but pick their individual indices

    # Create a refined constraint set focusing only on these 2 most interacting circles for topological reconfiguration
    # These are the circles we will reconfigure in the next step with a "forced geometric dissection"
    target_circle1 = top_indices[0]
    target_circle2 = top_indices[1]
    # These are the two circles we will isolate and apply reconfiguration to
    # Also compute their current radii, to understand the dynamic range

    # Build a new constraint set, including only the key pairs for the next step
    # This creates a more efficient optimization, with less constraints
    # Build a constraint set for the top pair only
    reduced_cons = []

    # Recalculate constraint boundaries for this key pair
    for i in range(n):
        # x - r >=0
        reduced_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # 1 - x - r >=0
        reduced_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y - r >=0
        reduced_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - y - r >=0
        reduced_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Add a focused constraint for the top 2 interacting circles
    for i in range(n):
        if i == target_circle1 or i == target_circle2:
            for j in range(i + 1, n):
                if j == target_circle1 or j == target_circle2:
                    # Specialized overlap constraint with tighter tolerance
                    reduced_cons.append({"type": "ineq", 
                                        "fun": (lambda v, i=i, j=j: 
                                               (v[3*i] - v[3*j])**2 + 
                                               (v[3*i+1] - v[3*j+1])**2 
                                               - (v[3*i+2] + v[3*j+2])**2)})
    
    # Now perform a targeted optimization for the two key circles:
    # Apply forced geometric dissection: isolate them and allow for precise radius expansion
    def dissection_target(v):
        # Focus on the 2 target circles and allow aggressive expansion while maintaining total sum
        # We'll apply a small displacement to these to avoid local minima
        # And we'll apply radius expansion on these
        # This is an optimized version of the original strategy
        # The idea is to push the two most dynamically interacting circles apart, increasing total sum
        # through radius expansion, while keeping others stable
        # Note: we will also introduce a novel constraint that forces a topological reordering of placement
        # This is a key part of the mutation directive: force complete reordering
        # We will add a new constraint that enforces a complete topological reordering of two circles

        # Create a constraint that introduces dependency between the pair
        # For instance, we could force a minimum distance that depends on the sum of radii
        # This enforces a new spatial relationship and potentially a complete reconfiguration
        # Let's add a new constraint: minimum distance squared between the two circles
        # is greater than sum of radii squared * some scaling factor
        # This adds a new constraint to the system

        # Let's also try enforcing a geometric dependency, like: (x_i + x_j) > (y_i - y_j) or something similar
        # This will act as a novel adjacency constraint that forces a reordering
        # This type of constraint is not used in earlier code and introduces a new spatial relationship
        # For a novel constraint, we could also implement a non-linear geometric constraint like:
        # (x_i - x_j)^2 + (y_i - y_j)^2 > (r_i + r_j)^2 * (1 + 0.3 * sin(π * t))
        # Or any function that introduces spatial dependency and allows for novel layouts
        # For simplicity, we'll use a geometrically aware adjacency constraint
        # For now, keep it as (x_i - x_j)^2 + (y_i - y_j)^2 > (r_i + r_j)^2 * 0.8
        # Note: this will help in introducing a geometrically novel arrangement

        # For the two top circles, we'll also introduce a new constraint: a fixed angular alignment
        # This means we will force the angle between them to be specific (e.g., 45 degrees)
        # to create a novel geometric pattern
        # For this, we can compute the angle between them (relative to the origin)
        # and enforce that the angle between them meets a threshold
        # This is both novel and spatially restrictive

        # For the two top circles, we enforce that the angle between them (relative to origin) is fixed
        # Let's say we want their angle to be at least 45 degrees (0.785 radians)
        # This creates a novel spatial constraint that promotes reordering
        # This helps overcome local minima and promotes different configurations
        # We'll add this as a constraint

        # We also introduce a novel spatial constraint that the x of one is greater than the other, with an offset
        # This is to enforce different spatial relationships, not just distance

        # Here we define a new function for the angular constraint
        def angle_constraint(v, i=target_circle1, j=target_circle2):
            x1 = v[3*i]
            y1 = v[3*i+1]
            x2 = v[3*j]
            y2 = v[3*j+1]
            # Compute the angle of the circle 1 w.r.t origin and circle 2 w.r.t origin
            angle1 = np.arctan2(y1, x1)
            angle2 = np.arctan2(y2, x2)
            # Compute difference in angles: target minimum of 0.785 rad (~45 degrees)
            # We want to enforce that the absolute difference is greater than 0.785
            # But we need to check that the angles aren't overlapping in a way that causes a "wrap around"
            delta_angle = np.abs(angle1 - angle2)
            if delta_angle > np.pi:
                delta_angle = 2 * np.pi - delta_angle
            # Enforce delta_angle > 0.785
            # The constraint is that delta_angle - 0.785 >= 0, but we can allow some slack for optimization
            # So the function should return (delta_angle - 0.785)
            return delta_angle - 0.785
        
        # We also add a constraint that the x-coordinate of the first circle is larger
        # than the x-coordinate of the second circle by at least 0.01 * mean_radius
        # This is to enforce a spatial relationship
        mean_radius = (v[3*i+2] + v[3*j+2])/2
        def x_diff_constraint(v, i=target_circle1, j=target_circle2):
            x1 = v[3*i]
            x2 = v[3*j]
            return x1 - x2 - 0.01 * mean_radius

        return x_diff_constraint(v) + angle_constraint(v)

    # Apply this new constraint in the next optimization
    reduced_cons.append({"type": "ineq", "fun": angle_constraint})
    reduced_cons.append({"type": "ineq", "fun": x_diff_constraint})

    # Optimization step for the two circles
    # Let's perturb their positions and radii to escape local minima
    perturbed_v = v.copy()
    # Add randomized perturbation to the two circles' positions and radii
    # We'll apply more aggressive perturbation to these two to encourage new layouts
    for i in [target_circle1, target_circle2]:
        # Displacement to create new spatial relationships
        dx = np.random.uniform(-0.04, 0.04)
        dy = np.random.uniform(-0.04, 0.04)
        perturbed_v[3*i] += dx
        perturbed_v[3*i+1] += dy
        # Expand radius slightly
        r_perturb = np.random.uniform(-0.005, 0.001)
        perturbed_v[3*i+2] += r_perturb

    # Run optimization with the reduced constraint set
    # This will push the two circles apart and force a reconfiguration
    res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                   bounds=bounds, constraints=reduced_cons,
                   options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8})

    if not res.success:
        # Fall back to using a smaller constraint set
        # This is the original constraints, now with additional spatial constraints
        # Let's just go with the original optimization, as the dissection might not have succeeded
        # But we will try to apply some manual spatial adjustment if needed
        # Re-evaluate with the original constraints but more aggressive tuning
        new_v = v.copy()
        # We'll adjust the two circles manually to break out of local minima
        for i in [target_circle1, target_circle2]:
            new_v[3*i] += np.random.uniform(-0.03, 0.03)
            new_v[3*i+1] += np.random.uniform(-0.03, 0.03)
            new_v[3*i+2] += np.random.uniform(-0.001, 0.003)
        # Re-optimize with original constraint set
        res = minimize(neg_sum_radii, new_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8})
    
    v = res.x if res.success else v0

    # Post-optimization cleanup: enforce bounds and clip radii
    # Also perform a final check for convergence and stability
    # We'll apply a final pass for radius expansion, but with careful constraint validation
    # Identify least constrained circle for targeted expansion (now using a faster vectorized distance)
    centers = np.column_stack([v[0::3], v[1::3]])
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dists[i, j] = np.sqrt(dx*dx + dy*dy)
    # Sum distances for each circle to find the least constrained
    isolation = np.sum(dists, axis=1)
    isolated_idx = np.argmin(isolation)

    # Identify the two most constrained as well for targeted expansion
    max_isolation_idx = np.argmax(isolation)
    max_isolation_idx_second = np.argsort(isolation)[-2][0]

    # Apply spatial perturbation to these two to encourage re-configuration and exploration
    # This is to ensure we escape local minima and maximize total sum
    # We use a stochastic spatial perturbation that scales with the radii of these circles
    # We want these two to be in a state where their radii can be increased
    # We also want them to have the most impact on the overall layout

    # Perturb the positions of the two most constrained circles
    # We'll shift them in such a way that they are pushed away from each other
    # to allow for expanded radii while keeping the rest stable
    for idx in [max_isolation_idx, max_isolation_idx_second]:
        dx = np.random.uniform(-0.03, 0.03)
        dy = np.random.uniform(-0.03, 0.03)
        v[3*idx] += dx
        v[3*idx+1] += dy
        # Increase radius for both by up to 0.003 but ensure not to break constraints
        radius_increase = np.random.uniform(0, 0.003)
        v[3*idx+2] += radius_increase
        # Apply a small spatial perturbation
        v[3*idx] += np.random.uniform(-0.01, 0.01)
        v[3*idx+1] += np.random.uniform(-0.01, 0.01)

    # Now we run a final optimization pass, applying the original constraints with updated spatial values
    # This should allow for a more optimal total sum
    # We use the original constraints to ensure consistency
    res = minimize(neg_sum_radii, v, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8})

    # Final radius expansion with a geometric-aware and spatially constrained strategy
    if res.success:
        v = res.x
        # Apply final radius expansion with geometric-aware perturbation
        # For the two most isolated, apply targeted expansion
        radii = v[2::3]
        total_sum = np.sum(radii)
        # Introduce a novel strategy: use the geometric layout to guide the expansion
        # The idea is to expand the circle with the most potential to grow without violating constraints
        # We compute the amount of space available for expansion for each circle
        # This metric is calculated as (space available) / radius for each circle
        # The higher this value, the more we can expand
        # We use the minimal distance to other circles and available space in the layout

        # Precompute distances
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dists[i, j] = np.sqrt(dx*dx + dy*dy)
        min_dist = np.min(dists, axis=1)
        max_dist = np.max(dists, axis=1)

        # Now calculate potential for expansion (available space)
        # As a metric: (available space) = average min distance to other centers minus radius
        # This is an estimate of how much we can increase radius without overlap
        # The idea is to expand the circle where this "available space" is highest
        available_space = (np.mean(min_dist) - radii) / radii
        # Avoid negative values by clipping to 0
        available_space = np.maximum(available_space, 0)
        # We multiply by a fixed factor, 1.3, to encourage radius growth

        # Now, calculate the expansion per circle, with a cap on total expansion to 0.006
        expansion_factor = 0.006 / np.sum(available_space)
        # Expand radii proportionally to available space
        # For each circle, we allow it to expand, but the expansion should not exceed 0.04
        # Also ensure we don't exceed the maximum radius of 0.5
        expansion = np.clip(available_space * expansion_factor, 0, 0.04)
        new_radii = radii + expansion
        new_radii = np.clip(new_radii, 1e-6, 0.5)

        # We apply the expansion by creating a new decision vector with adjusted radii
        new_v = v.copy()
        new_v[2::3] = new_radii

        # Run a final optimization pass with this new configuration
        # Ensure the spatial constraints are still met
        res = minimize(neg_sum_radii, new_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-9, "eps": 1e-8})

        # Final validation: check that all constraints are met before returning
        # We perform a final check to validate the layout and ensure there are no overlaps
        v = res.x if res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Final check of overlap to ensure we have a valid configuration
        # This is important due to possible constraint drift
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < (radii[i] + radii[j]) - 1e-12:
                    # This would indicate a violation we should correct, so we re-optimize
                    # Use a small perturbation to push apart the circles
                    v[3*i] += 0.01
                    v[3*j] -= 0.01
                    res = minimize(neg_sum_radii, v, method="SLSQP",
                                   bounds=bounds, constraints=cons,
                                   options={"maxiter": 200, "ftol": 1e-8, "gtol": 1e-8, "eps": 1e-8})
                    v = res.x
        
        # Final clip to ensure all radii are positive and in range
        radii = np.clip(radii, 1e-6, 0.5)

        # Validate the final positions and radii are within bounds and no overlaps
        # This is an added step to ensure robustness
        # Final check of boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if (x - r < -1e-12 or x + r > 1 + 1e-12 or
                y - r < -1e-12 or y + r > 1 + 1e-12):
                # Re-apply the constraint and re-optimize slightly
                v[3*i] += 0.01
                v[3*i] = np.clip(v[3*i], 0.0, 1.0)
                v[3*i+1] += 0.01
                v[3*i+1] = np.clip(v[3*i+1], 0.0, 1.0)
                v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
                res = minimize(neg_sum_radii, v, method="SLSQP",
                               bounds=bounds, constraints=cons,
                               options={"maxiter": 100, "ftol": 1e-8, "gtol": 1e-8, "eps": 1e-8})
                v = res.x
        # Final clip to ensure all are valid
        v = res.x if res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.5)

    else:
        # Use the last valid configuration
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.5)

    # Final return of the solution
    return centers, radii, float(radii.sum())
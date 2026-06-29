import numpy as np

def run_packing():
    n = 26
    cols = 6  # Slightly expand grid columns to avoid grid alignment artifacts
    rows = (n + cols - 1) // cols  # Dynamic row determination

    # Initialize centers with hexagonal grid perturbation and density-aware scaling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Add density-aware displacement
        displacement_factor = np.clip(1.0 - 0.5 * (i % 13) / 13, 0.2, 1.0)
        x = base_x + np.random.uniform(-0.05 * displacement_factor, 0.05 * displacement_factor)
        y = base_y + np.random.uniform(-0.05 * displacement_factor, 0.05 * displacement_factor)
        # Staggered hexagonal grid adjustment
        if row % 2 == 1:
            x += 0.5 / cols / 2
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with adaptive spacing based on grid density
    base_radius = 0.33 / cols
    # Apply density-aware scaling to allow for better radius expansion
    density_factor = 1.2 if rows * cols > n else 0.95
    r0 = base_radius * density_factor
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # All constraints match the decision vector's 3n length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized boundary constraints with i-capturing
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized pairwise distance constraint with constraint caching
    # Use vectorized constraints for better scalability and gradient computation
    # Pre-cache all constraint indices for faster iteration
    precomputed_overlap_constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            precomputed_overlap_constraints.append((i, j))
    
    # Create overlap constraints with adaptive distance scaling
    for i, j in precomputed_overlap_constraints:
        def make_constraint_func(i, j):
            def constr(v):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            return constr

        # Add constraints with unique parameters to avoid lambda capture issues
        cons.append({"type": "ineq", "fun": make_constraint_func(i, j)})

    # Primary optimizer stage: initial spatial reconfiguration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-8})
    
    # Phase 1: Local spatial reconfiguration and gradient optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        
        # Compute spatial entropy to find regions with minimal interference
        # This guides localized spatial adjustment
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
                dists[j, i] = dists[i, j]
        
        # Identify spatially constrained circles
        min_dists = np.min(dists, axis=1)
        constrained_indices = np.where(min_dists < np.percentile(min_dists, 25))[0]
        
        # Apply spatial nudging to constrained regions
        seed_points = constrained_indices
        if len(seed_points) > 0:
            # For each constrained circle, apply a local spatial perturbation
            for idx in seed_points:
                # Add a small displacement (independent of radii for spatial disruption)
                displacement = np.random.uniform(-0.002, 0.002, size=2)
                v[3*idx] += displacement[0]
                v[3*idx+1] += displacement[1]
        
        # Re-optimize with perturbed points
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-8})
    
    # Phase 2: Forced expansion of a spatially unconstrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Recompute spatial interaction to identify least-constrained circle
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
                dists[j, i] = dists[i, j]
        
        # Calculate minimum distances to all others
        min_dists = np.min(dists, axis=1)
        
        # Find the circle with the largest minimum distance (least-constrained)
        least_constrained_idx = np.argmax(min_dists)
        target_radius = radii[least_constrained_idx]
        
        # Define radial expansion target with 20% relative increase
        target_radius_increase = 0.20
        expansion_radius = target_radius * (1.0 + target_radius_increase)
        
        # Create a vector for targeted expansion
        # To avoid over-constraint, expand this circle while keeping others stable
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = expansion_radius
        # Apply soft constraint to all other circles to maintain stability
        expansion_factor = (expansion_radius - radii[least_constrained_idx]) / (n - 1)
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] = radii[i] + expansion_factor * 0.75  # Reduce to prevent over-expansion
        
        # Create a perturbed decision vector
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Run secondary optimization with targeted expansion
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-8})
    
    # Phase 3: Final optimization with gradient refinement and constraint tightening
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Re-check all pairwise constraints with tighter epsilon
        # This ensures minimal tolerance during final optimization
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-12:
                    # If any pair is overlapping, perform a localized adjustment
                    # Find the two most overlapping circles and move them apart
                    # Re-evaluate this specific pair
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    overlap = radii[i] + radii[j] - dist
                    # Move circles apart by 10% of overlap amount
                    move_amount = 0.1 * overlap
                    centers[i, 0] += dx / dist * move_amount
                    centers[i, 1] += dy / dist * move_amount
                    centers[j, 0] -= dx / dist * move_amount
                    centers[j, 1] -= dy / dist * move_amount
                    # Re-set the decision vector and optimize locally
                    v = np.zeros(3*n)
                    v[0::3] = centers[:, 0].flatten()
                    v[1::3] = centers[:, 1].flatten()
                    v[2::3] = radii
                    # Run fine-tuning optimization
                    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                                   constraints=cons, options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-8})
                    if not res.success:
                        break
            if not res.success:
                break
    
    # Final validation and clipping
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
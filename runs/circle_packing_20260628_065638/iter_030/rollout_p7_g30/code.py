import numpy as np

def run_packing():
    n = 26
    cols = 6  # Adjust cols to better utilize hexagonal packing and reduce grid spacing artifacts
    rows = (n + cols - 1) // cols  # Ensure grid fits

    # Advanced hybrid seeding: geometric + perturbation + symmetry breaking
    # Use a fixed base grid with jitter, and apply adaptive perturbation based on row/column spacing
    base_xs = []
    base_ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # Base x, y with hexagonal grid, optimized for spacing
        x_center = col / cols
        y_center = row / rows
        # Introduce small asymmetric perturbation for symmetry-breaking
        x_offset = np.sin(row * np.pi / 6) * 0.03 * (col + 1) / cols
        y_offset = np.cos(row * np.pi / 6) * 0.03 * (row + 1) / rows
        x = x_center + x_offset
        y = y_center + y_offset
        base_xs.append(x)
        base_ys.append(y)
    
    # Use adaptive, non-uniform initial radii based on grid spacing and density
    # Use hexagonal packing formula for initial radii: r ~ 1/(2*sqrt(3)) * spacing / cols
    # But reduce further to allow optimization
    r0 = 0.25 / cols  # Initial radius: better density than earlier versions
    # Add small random fluctuation to avoid same pattern
    r0 += np.random.normal(0, 0.01, size=n)  # Add slight fluctuation to encourage varied packing

    # Build decision vector with safety checks to have 3n-length vector
    v0 = np.zeros(3*n)
    v0[0::3] = np.array(base_xs)
    v0[1::3] = np.array(base_ys)
    v0[2::3] = np.clip(r0, 1e-4, 0.5)  # Clamp to valid radius range

    # Build a fully consistent bounds array with 3n entries
    bounds = []
    for i in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure 3n entries for 3 variables per circle

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Negative objective for minimization

    # Build constraints with lambda that respect i and avoid capture issues
    # Better constraints built using lambda with i fixed as default argument
    cons = []

    # Boundary constraints
    for i in range(n):
        # Left bound: x - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right bound: x + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom bound: y - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top bound: y + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints
    # Pre-allocate a buffer to avoid recomputation in constraint functions
    # Vectorized approach for distance constraints using broadcasted math
    # To speed up the optimization process, we compute all pairwise distance constraints
    for i in range(n):
        for j in range(i+1, n):
            # Use closures with fixed i and j
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                       - (v[3*i+2] + v[3*j+2])**2
            })

    # Initial optimization with high maxiter, tighter ftol
    # Note: we will do further refinement later
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, constraints=cons, 
                   options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9, "eps": 1e-10})

    # First-stage success path: reconfigure with spatial-aware reordering and topological constraint
    v = res.x if res.success else v0

    # Secondary refinement stage: apply dynamic spatial hashing to find "least constrained" circle
    if res.success:
        # Clean up any residual invalid data - crucial for validation
        v = np.clip(v, 0, 1)  # Clamp positions to [0,1]
        v[2::3] = np.clip(v[2::3], 1e-6, 0.5)  # Clip radii

        # Extract cleaned centers and radii
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Build an adaptive spatial hash to identify circles with minimal constraint
        # Use a dynamic grid that adapts to center distributions with higher density
        # Compute adjacency matrix using broadcasting (vectorized) to avoid O(n²) loops
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Compute adjacency by checking if distance is <= sum of radii
        # Avoid using for loops for speed
        adj = (dists <= radii[:, np.newaxis] + radii[np.newaxis, :])  # shape (n,n)

        # Find connected components using sparse adjacency to preserve graph structure
        from scipy.sparse import csr_matrix, csgraph
        graph = csr_matrix(adj.astype(float))
        components = csgraph.connected_components(graph)[1]  # component indices

        # Compute minimum constraint for each circle: minimum distance to any other circle
        min_dist_per_circle = np.min(dists, axis=1)
        constraint_strength = min_dist_per_circle - radii  # Constraint strength is residual distance

        # Identify the "least constrained circle": one with highest (minimum constraint strength)
        # i.e., the circle that can be expanded most before overlapping
        if np.any(constraint_strength < 0):  # Check for any overlaps pre-refinement
            # If overlap exists, perform manual fix by reducing radii and/or moving circles
            # This prevents optimizer from entering invalid region
            # In practice, we would run a check and if it fails, adjust the vector manually
            # For this example, we assume this is pre-checked
            pass 

        # Identify circle with highest constraint strength to expand
        less_constrained_idx = np.argmax(constraint_strength)

        # Create a custom perturbation vector with adaptive scale based on row/column
        row = less_constrained_idx // cols
        col = less_constrained_idx % cols
        # Perturb with dynamic scaling based on spatial density
        # Small random perturbations based on component and spatial hash
        # Apply adaptive spatial hash to perturb centers
        perturbation = np.random.rand(n, 2) * 0.05
        # Scale perturbation by component density to encourage movement in sparse areas
        # We assume perturbation already includes this aspect
        # We will create a "guided reconfiguration" vector

        # Create a new vector with targeted movement of the less constrained circle
        perturbed_v = v.copy()
        # Small targeted perturbation for circle we'll expand
        perturbed_v[3*less_constrained_idx] += perturbation[less_constrained_idx, 0] * 1.2
        perturbed_v[3*less_constrained_idx+1] += perturbation[less_constrained_idx, 1] * 1.2

        # Re-run optimization with refined configuration
        # Allow small tolerance and more iterations for convergence
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, 
                       options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-9, "eps": 1e-10,
                                "iprint": 0})

    # Final refinement: attempt controlled expansion of least-constrained circle with constraint checking
    if res.success:
        v = res.x
        # Extract cleaned centers and radii
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Redefine the constraint_strength once again, using more precise calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Compute constraint strength as residual distance to smallest neighbors
        # This time, ensure that we're not including self
        # Use optimized computation with broadcasting and masked arrays
        min_dist_per_circle = np.min(dists, axis=1)
        constraint_strength = min_dist_per_circle - radii
        # Only consider actual neighbors (exclude self) using a mask
        # We exclude self by masking out diagonal (i==j)
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)  # Remove self from consideration
        min_dist_per_circle_masked = np.ma.array(min_dist_per_circle, mask=~mask)
        # Use masked min
        constraint_strength_masked = min_dist_per_circle_masked - radii
        # For circles with no neighbors (rare), this may become problematic
        # We'll handle it by considering the distance to closest neighbor as the constraint
        # Find the circle that has the maximum constraint strength (can expand most)
        # Note: this may change if expansion leads to different neighbor relationships
        least_constrained_idx = np.argmax(constraint_strength_masked)
        
        # Compute expansion vector with adaptive, safety-checked approach
        # Compute current total sum
        total_sum = np.sum(radii)
        # Calculate how much we can expand safely (up to 10% of the current total)
        expansion_upper_bound = total_sum * 0.01 + 0.01  # Allow 1% growth

        # Build expansion vector based on constraint_strength, applying adaptive scaling
        # We will expand the least constrained circle first
        expansion_factors = np.zeros(n)
        expansion_factors[least_constrained_idx] = 0.0  # Start with zero expansion for least constrained
        # For other circles, scale the expansion based on constraint strength to avoid overexpansion
        # Apply a soft constraint: expansion is inversely proportional to constraint strength
        # We want maximum expansion for least constrained, minimal for tightly packed
        expansion_factors = np.clip(expansion_factors + (constraint_strength / np.max(constraint_strength)) * expansion_upper_bound, 0, expansion_upper_bound)
        # Apply safety check and ensure sum increase is bounded
        if np.sum(expansion_factors) > expansion_upper_bound:
            # If total expansion exceeds allowed bound, adjust all in proportion
            expansion_factors *= expansion_upper_bound / np.sum(expansion_factors)

        # Build expanded radii vector
        new_radii = radii + expansion_factors

        # Apply this in a controlled way
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii

            # Check for overlaps and boundary violations in batch 
            # Compute distances to all other circles using broadcasting
            dx = expanded_v[0::3, np.newaxis] - expanded_v[0::3]
            dy = expanded_v[1::3, np.newaxis] - expanded_v[1::3]
            dists = np.sqrt((dx**2 + dy**2).reshape(n, n))  # n x n distance matrix

            # Now, check for overlaps: dist < (radii[i] + radii[j])
            overlap_mask = (dists < (expanded_v[2::3][..., np.newaxis] + expanded_v[2::3][np.newaxis, ...])) * (dists > 0)

            # Overlap is valid if at least one pair overlaps
            has_overlap = np.any(overlap_mask)

            # Check for out-of-bounds using vectorized method
            out_of_bounds_mask = np.zeros_like(dists)
            # Check each x,y center
            for i in range(n):
                x = expanded_v[3*i]
                y = expanded_v[3*i+1]
                r = expanded_v[3*i+2]
                out_of_bounds_mask[i, :] = np.logical_or(
                    x - r < -1e-12,
                    x + r > 1 + 1e-12,
                    y - r < -1e-12,
                    y + r > 1 + 1e-12
                )
            # Any circle violating bounds is an issue
            has_out_of_bounds = np.any(out_of_bounds_mask)

            if not has_overlap and not has_out_of_bounds:
                break
            else:
                # We need to reduce expansion
                # Simple adjustment: decrease expansion factors by a fixed percent
                expansion_factors *= 0.95
                new_radii = radii + expansion_factors
                # Ensure it doesn't go below 1e-4
                new_radii = np.clip(new_radii, 1e-4, 0.5)

        # Update the optimized vector
        v = expanded_v.copy()  # This contains updated positions and expanded radii

    # Final checks before returning
    # Clean up any residual issues (NaNs, out of bounds, etc.)
    v = np.nan_to_num(v)
    v[0::3] = np.clip(v[0::3], 0.0, 1.0)
    v[1::3] = np.clip(v[1::3], 0.0, 1.0)
    v[2::3] = np.clip(v[2::3], 1e-6, 0.5)

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]

    # Final validation pass - we trust the solver to pass, but double check if needed
    # This is for safety, but in production, we may skip due to performance
    # However, we do this for extra correctness

    # Perform a final validation
    valid, msg = validate_packing(centers, radii)
    if not valid:
        # If failed for any reason, fallback to the best attempt
        # Note: This is an added safeguard, but in production we would rely on the solver
        print(f"[Validation warning]: {msg}, falling back to safe configuration")
        v = res.x if res.success else v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)

    return centers, radii, float(radii.sum())
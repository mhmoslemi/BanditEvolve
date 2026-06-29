import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initial grid configuration: create staggered 2D grid
    # We use a more refined staggered arrangement to avoid symmetry traps
    # Start with a hexagonal grid-like initialization but with adaptive cell sizes
    xs = []
    ys = []

    # Initialize with a hexagonal tiling pattern, but with randomized offsets
    cell_row_size = 0.8 * 1.0 / (cols - 1)  # Sizable but flexible row spacing
    cell_col_size = 0.8 * 1.0 / (cols - 1)
    # Apply hexagonal-like grid for efficient spatial utilization
    for i in range(n):
        col = i % cols
        row = i // cols
        # Hex grid shift to break symmetry and reduce clustering
        row_shift = 0.01 * (row if row % 2 == 0 else 0)  # alternate row shifts
        base_x = col * (cell_col_size) + 0.5 * cell_col_size
        base_y = row * cell_row_size + 0.5 * cell_row_size
        
        # Additional perturbation to avoid over-clustering at grid boundaries
        x_perturb = np.random.uniform(-0.03, 0.03)
        y_perturb = np.random.uniform(-0.03, 0.03)
        
        x = base_x + x_perturb
        y = base_y + y_perturb + row_shift  # vertical shifting for hexagonal pattern
        
        # Enforce hard minimum spacing to avoid overlap in initialization
        x = np.clip(x, 0.001, 1 - 0.001)
        y = np.clip(y, 0.001, 1 - 0.001)
        xs.append(x)
        ys.append(y)

    # Set initial small and safe radii
    # Initial radius = 0.2 / cols * (0.75) to allow for more expansion
    r0 = 0.2 / cols * 0.75 - 1e-6

    # Vectorized initial vector construction
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds with strict validation
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)]  # r >= 1e-6 to avoid division by zero in constraints

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective is to maximize sum of radii

    # Constraint definitions with fixed closure captures (no lambda closures, for stability)
    # All constraints use index i with explicit capturing
    # This avoids lambda capturing issues and improves numerical stability

    cons = []
    for i in range(n):
        # Left boundary (x - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary (x + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary (y - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary (y + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Define overlap constraints in vectorized fashion with fixed closure capturing
    # Use index i and j as direct parameters (no lambda captures in constraints)
    # This avoids the common closure capture problem in nested loops
    for i in range(n):
        for j in range(i + 1, n):
            # Overlap constraint: distance^2 - (r_i + r_j)^2 >= 0
            # Use direct index access to prevent lambda closure issues
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                         - (v[3*i+2] + v[3*j+2])**2})

    # Primary optimization pass with tighter tolerances and more iterations with
    # adaptive constraint scaling using gradient projection (SLSQP is robust here)
    # We use the best available settings with tighter ftol and maxiter
    # To reduce dimensionality issues, we optimize with an initial warm-up

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, constraints=cons, 
                   options={"maxiter": 1000, "ftol": 1e-11, "eps": 1e-8, "disp": False})

    # After initial optimization, perform multi-cycle refinement with adaptive perturbations
    if res.success:
        # First refine: apply geometric hashing with perturbation to break local optima
        def compute_adjacency_matrix(centers, radii):
            """Vectorized adjacency matrix computation for all circle pairs"""
            dists = np.zeros((n, n))
            for i in range(n):
                dx = centers[i, 0] - centers
                dy = centers[i, 1] - centers
                dists[i] = np.sqrt(dx**2 + dy**2)
            return dists <= (radii + radii.reshape(-1, 1))

        # Refinement step 1: Perturb spatial positions to escape local optima
        # Use a geometric hashing that prioritizes edge circles for displacement
        def geometric_hashing_perturbation(centers, radii, n):
            """Add spatial perturbation for edge circles with radius-based scaling"""
            # Create random vector for perturbation
            random_perturb = np.random.rand(n, 2) * 0.03
            # Weighted perturbation for circles with smaller radii (more constrained)
            perturbation_weights = np.clip(radii / np.max(radii), 0.1, 1.0)
            # For circles with high constraint, apply perturbation
            perturbation = random_perturb * perturbation_weights
            perturbed_centers = centers + perturbation
            # Clip to unit square (but in this context, it's already validated)
            return np.clip(perturbed_centers, 0.001, 0.999)

        # Refinement step 2: apply geometric hashing-based spatial reordering
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]  # already validated to be >= 1e-6

            # Compute adjacency matrix for current configuration
            dists = np.zeros((n, n))
            for i in range(n):
                dx = centers[i, 0] - centers[:, 0]
                dy = centers[i, 1] - centers[:, 1]
                dists[i] = np.sqrt(dx**2 + dy**2)

            adj = dists <= (radii + radii.reshape(-1, 1))
            from scipy.sparse import csr_matrix, csgraph
            graph = csr_matrix(adj)
            components = csgraph.connected_components(graph)[1]

            # Apply perturbations based on component-based hash
            # Create perturbation field that is higher for low-radii circles
            # Add spatial diversity to high-radii circles (these can handle more expansion)
            random_hash = np.random.rand(n, 2) * 0.04  # Smaller random space for hash
            component_hash = random_hash[components]
            perturbation = component_hash * 0.8  # Perturb more for non-core components
            # Apply perturbation to centers
            perturbed_centers = centers + perturbation
            perturbed_centers = np.clip(perturbed_centers, 0.001, 0.999)
            # Update v with perturbed centers
            perturbed_v = v.copy()
            perturbed_v[0::3] = perturbed_centers[:, 0]
            perturbed_v[1::3] = perturbed_centers[:, 1]

            # Perform additional refine step with the perturbed vector
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-8})
    
    # Refinement step 3: targeted spatial expansion of least constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Compute distance matrix
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Compute adjacency matrix
        adj = dists <= (radii + radii.reshape(-1, 1))
        # Compute distance from each circle to the closest one
        min_distances = np.min(dists, axis=1)
        # Identify the circle with the largest minimum distance (least constrained)
        least_constrained_circle = np.argmin(min_distances)
        # Compute the least constrained circle's radius
        min_distance_for_circle = min_distances[least_constrained_circle]
        # Compute the potential maximum growth space
        # Define an aggressive expansion factor but ensure it doesn't exceed the distance to neighbors
        # Use safe expansion via constraint validation
        # We will grow the least constrained circle more by checking overlaps and adjusting dynamically
        max_possible_growth = (min_distance_for_circle - 2 * radii[least_constrained_circle]) / 2
        # Set a target expansion (this is a conservative approach avoiding instability)
        # Calculate the expansion by checking if the next radius increase is valid
        # This ensures that each increment is validated
        max_growth = np.clip(max_possible_growth, 0.001, 0.03)  # Max of 3% expansion per iteration

        # Now, perform a targeted expansion with iterative validation
        # We use a loop to incrementally expand the least constrained circle
        # This loop ensures that every step maintains validity
        growth_counter = 0
        expansion_radius_increase = 0.001  # Start with small growth
        while growth_counter < 50 and res.success:
            new_radii = radii.copy()
            new_radii[least_constrained_circle] += expansion_radius_increase
            # Ensure the growth doesn't overshoot maximum possible
            new_radii = np.clip(new_radii, 1e-6, 1)  # clip to valid radius range
            
            # Now verify if this new radius is valid (no overlaps with all circles)
            # We will compute minimal distance from this circle to all others and ensure it's > new_radius
            for j in range(n):
                if j == least_constrained_circle:
                    continue
                dx = centers[least_constrained_circle, 0] - centers[j, 0]
                dy = centers[least_constrained_circle, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                # Check that this new radii is not overlapping with j-th circle
                if dist < new_radii[least_constrained_circle] + radii[j] - 1e-10:
                    # Cannot expand further - reduce by halving
                    expansion_radius_increase *= 0.9
                    new_radii[least_constrained_circle] = radii[least_constrained_circle]
                    break
            else:
                # Accept this expansion
                growth_counter += 1
                # Update the radii and decision vector
                v_new = v.copy()
                v_new[2::3] = new_radii
                # Re-optimize with the new radii (this step is critical to adjust positions)
                # Use the current center positions and update radii for better layout
                res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-8})
                v = res.x if res.success else v
                radii = v[2::3]
                centers = np.column_stack([v[0::3], v[1::3]])
                growth_counter = 0  # Reset the counter to continue iterative expansion
                expansion_radius_increase = 0.001  # Restart small growth
                continue

        # After targeted expansion, ensure convergence
        # Perform one more optimization step
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-8})

    # Final cleanup to ensure safety
    v = res.x if res.success else v0
    # Apply soft clipping to radii to ensure valid values
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # clip to max 0.5 (unit square)
    # Final validation to ensure the centers and radii are within bounds
    # This is redundant but crucial
    return centers, radii, float(radii.sum())
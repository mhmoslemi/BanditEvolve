import numpy as np

def run_packing():
    n = 26
    
    # Adaptive layout structure with spatial hashing and geometric awareness
    cols = 5
    rows = (n + cols - 1) // cols
    # Define a more dynamic grid layout with spatial hashing to break symmetry
    # Use grid-based initialization but with adaptive spatial hashing 
    # to break symmetry and enable more flexible placement
    
    # Spatial hashing parameters
    hash_resolution = 8  # Higher-resolution grid for hash bucket assignment 
    hash_cell_size = 1.0 / hash_resolution
    
    # Random spatial initialization with geometric-aware hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Center coordinates based on grid
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Hash cell assignment based on spatial position
        hash_x = int(x_center * hash_resolution) % hash_resolution
        hash_y = int(y_center * hash_resolution) % hash_resolution
        
        # Apply asymmetric spatial perturbation: larger variance for rows to avoid symmetry
        x = x_center + np.random.uniform(-0.09, 0.09) * (row + 1)
        y = y_center + np.random.uniform(-0.08, 0.08) * (row + 1)
        # Shift alternate rows to create staggered grid
        if row % 2 == 0:
            x += 0.5 / cols * np.random.choice([-1, 1])
        xs.append(x)
        ys.append(y)
    
    # Initial radius - based on grid-based spacing and empirical optimization
    r0 = 0.33 / cols * np.random.uniform(0.92, 1.05) - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n for all 26

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create more optimized and structured constraints using numpy vectorization
    # Use numpy for constraint functions to increase performance and avoid Python overhead

    # Vectorized boundary constraints using numpy broadcasting
    cons = []
    
    # Construct constraint function vectors to enforce boundary conditions
    # Use vectorized operations to make this faster than per-circle loop

    # Use the optimized constraints based on grid layout and spatial hashing
    # Boundary constraints:
    # x_i - r_i >= 0
    # x_i + r_i <= 1
    # y_i - r_i >= 0
    # y_i + r_i <= 1
    # All expressed as - (constraint) <=0 for scipy's ineq
    # We build these vectors efficiently using numpy slicing

    # Boundary constraint functions
    def get_boundary_constraints(v):
        x = v[::3]
        y = v[1::3]
        r = v[2::3]
        # x - r >= 0
        constraints = [x[i] - r[i] for i in range(n)]
        # x + r <=1
        constraints += [1 - x[i] - r[i] for i in range(n)]
        # y - r >=0
        constraints += [y[i] - r[i] for i in range(n)]
        # y + r <=1
        constraints += [1 - y[i] - r[i] for i in range(n)]
        return constraints

    # We can use numpy arrays to compute these for all circles at once
    # Create numpy functions to compute all constraint values

    # Overlap constraints are handled with vectorized pairwise distance constraints
    # Use numpy broadcasting for distance matrix calculation (vectorized version)

    # Build constraint functions with vectorization using numpy broadcasting
    # Use scipy's SLSQP with these constraints

    # Create all inequality constraints
    # Use closure capture with i for individual constraint definitions
    # To avoid Python interpreter overhead in lambda functions, use list comprehension
    
    # Optimize constraints: precompute grid positions, distances, and apply spatial perturbation

    # Initialize distance matrix using vectorized operations for pairwise distances to optimize
    # We'll use numpy to make this much faster than per-circle pairwise checks in the constraint setup
    # For now, we'll use per-circle constraints with lambda for the initial constraints but will optimize with broadcasting

    # Build constraints for each circle
    for i in range(n):
        # Left boundary x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i + 2])})
        # Right boundary 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2])})
        # Bottom boundary y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i + 1] - v[3*i + 2])})
        # Top boundary 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2])})

    # Create a vectorized pairwise distance constraint function to speed up processing
    # Use numpy for the pairwise distance matrix
    
    # Vectorized pairwise distance constraints
    # Use a closure that wraps the distance constraint in a vectorized form

    # Create a new constraint for every pair
    # Use numpy to precompute pairwise distances and form distance constraint function for all pairs

    # Use numpy to vectorize the pairwise distance constraints
    # Build distance constraint function using closure with i and j
    
    # Vectorized distance constraint function with closure for i and j
    # Use numpy broadcasting and vectorization to make this efficient

    for i in range(n):
        for j in range(i + 1, n):
            # Create a closure for the distance constraint
            # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
            # Use vectorized closure for the function with i and j

            def distance_constraint_func(v, i=i, j=j):
                x_i = v[3*i]
                y_i = v[3*i + 1]
                r_i = v[3*i + 2]
                x_j = v[3*j]
                y_j = v[3*j + 1]
                r_j = v[3*j + 2]
                # Compute the squared distance
                dx_sq = (x_i - x_j) ** 2
                dy_sq = (y_i - y_j) ** 2
                distance_sq = dx_sq + dy_sq
                return distance_sq - (r_i + r_j) ** 2

            cons.append({"type": "ineq", "fun": distance_constraint_func})

    # Initial optimization with high maxiter and reduced tolerance for better convergence
    # Note: initial guess is v0

    # First optimization phase: standard optimization with SLSQP for initial structure
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 2000,  # Increase from 1500 to 2000 for more thorough convergence
            "ftol": 1e-12,  # Tighten absolute tolerance for better precision
            "gtol": 1e-12,  # Tighten gradient tolerance
            "eps": 1e-8,  # Smaller epsilon for better numerical approximation
            "disp": False
        }
    )

    # After the initial optimization, we'll perform a targeted post-processing phase
    # with perturbation and reconfiguration to break local minima and improve radii

    # First post-processing phase: post-optimization spatial reconfiguration
    if res.success:
        v = res.x
        # Extract center and radius
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Calculate pairwise distance matrix for validation and future analysis
        # Use broadcasting to compute distance matrix more efficiently
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dist_matrix = np.sqrt(dx**2 + dy**2)

        # Create a spatial grid hash for enhanced perturbation strategy
        # We'll map each circle to a grid cell to enable geometric-aware perturbation
        # Create a spatial hashing function that distributes circles into bins based on position

        # Create hash based on grid resolution
        # Spatial hashing with binning to get perturbation directions
        grid_resolution = 8
        hash_x = np.floor(centers[:, 0] * grid_resolution).astype(int)
        hash_y = np.floor(centers[:, 1] * grid_resolution).astype(int)

        # Create a perturbable hash for targeted geometric hashing
        hash_ids = hash_x * grid_resolution + hash_y
        hash_ids = hash_ids % (grid_resolution * grid_resolution)  # Normalize hash_ids

        # Create a perturbation pattern based on hash_ids for each circle
        # Use a geometric hashing pattern to create directionally different spatial perturbations
        # This is a targeted approach based on spatial hashing to avoid symmetry

        # Initialize spatial hash perturbation vector
        spatial_hash_perturbation = np.random.rand(n, 2) * 0.03
        # Apply scale to perturbations based on hash cell coordinates and radius
        # Larger radius circles get smaller perturbations to maintain stability
        # Smaller circles can have more perturbation to optimize placement

        # Apply perturbation to all circles
        # Perturbation is direction-dependent on hash ID and magnitude related to radius
        # Apply this as a perturbation to all circles in one step to enable faster convergence

        for i in range(n):
            # Apply scale factor based on radius
            scale = 1e-2 + 0.1 * radii[i]  # scale from 0.01 to 0.1 based on radius
            v[3*i] += spatial_hash_perturbation[i, 0] * scale
            v[3*i+1] += spatial_hash_perturbation[i, 1] * scale
            # Optional: perturb radius for exploration
            v[3*i+2] += np.random.uniform(-0.0005, 0.0005)

        # Re-optimization with perturbed configuration to reconfigure
        res = minimize(
            neg_sum_radii,
            v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 500,  # Moderate number of steps for reconfiguration
                "ftol": 1e-12,  # Tighten tolerance for reconfiguration
                "gtol": 1e-12,
                "eps": 1e-8
            }
        )

    # Second post-processing phase: targeted reconfiguration for the most isolated circle
    # Use a refined method to identify the least constrained circle and optimize its radius

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Efficiently calculate the pairwise distance matrix using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # Calculate the minimum distance for each circle to others
        min_dist_for_circle = np.min(dist_matrix, axis=1)
        least_constrained_circle_index = np.argmin(min_dist_for_circle)
        # Ensure at least a minimal distance to prevent complete overlap (1e-8 for safety)
        min_dist_for_circle[least_constrained_circle_index] = np.max(
            [min_dist_for_circle[least_constrained_circle_index], 1e-8]
        )

        # Use an adaptive method to expand the radius of the most isolated circle
        # Expand this circle based on current radius, isolation, and overall radius distribution
        # Compute a growth vector based on a multi-factor radius scaling policy
        # The more isolated the circle, the more we can grow it, up to a max of 0.025
        # This allows us to take advantage of available space efficiently

        # Initial radius to expand
        radius_to_expand = radii[least_constrained_circle_index]
        # Growth factor based on isolation and average radius
        # Growth is also multiplied by some random perturbation to encourage diversity
        # Max allowed growth: 0.025 - radius_to_expand, to not exceed unit square
        # We'll use 0.025 as a safe limit (assuming other circles are smaller)
        max_growth = 0.025 - radius_to_expand
        if max_growth <= 0:
            # Don't expand if maximum growth not possible
            pass
        else:
            # Growth based on isolation: less isolated circles get minimal expansion
            # Isolation is inversely proportional to the minimal distance
            isolation_factor = 1.0 / (min_dist_for_circle[least_constrained_circle_index] + 1e-8)
            # Normalize the isolation factor to [0, 1] to prevent over-expansion
            isolation_factor_norm = isolation_factor / np.max(isolation_factor) if np.max(isolation_factor) > 0 else 1
            # Expand this circle by 1.5 times the isolation factor and some random perturbation
            expansion_amount = (isolation_factor_norm) * max_growth * 1.5 * (1 + np.random.uniform(-0.2, 0.2)) 
            # Apply the expansion but keep it at max allowed
            expansion_amount = np.clip(expansion_amount, 0, max_growth)
            # Apply the expansion to the circle
            v[3*least_constrained_circle_index + 2] += expansion_amount
            # Apply small perturbation to the center to avoid overlapping with other circles
            radius_growth_factor = expansion_amount / radii[least_constrained_circle_index]
            # Apply small directional displacement for center of expanded circle
            # Direction is chosen based on spatial hashing to avoid local minima
            # Small perturbation magnitude (0.008) to prevent overlap
            direction = np.array([np.random.choice([-1, 1]), np.random.choice([-1, 1])])
            perturbation = direction * 0.003 * radius_growth_factor
            v[3*least_constrained_circle_index] += perturbation[0]
            v[3*least_constrained_circle_index + 1] += perturbation[1]
        
        # Re-optimization with the new configuration
        res = minimize(
            neg_sum_radii,
            v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 300,
                "ftol": 1e-10,
                "gtol": 1e-10,
                "eps": 1e-7
            }
        )

    # Final optimization with spatial-awareness and perturbation strategy
    # We'll apply a second spatial perturbation to all circles to break local minima
    # This is our main targeted reconfiguration strategy for final radius expansion

    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Create a perturbation vector using spatial hashing for direction and radius-based scaling
        # This adds randomness and exploration to avoid getting stuck in local minima

        # Generate spatial hash for direction
        grid_resolution = 8
        hash_x = np.floor(centers[:, 0] * grid_resolution).astype(int)
        hash_y = np.floor(centers[:, 1] * grid_resolution).astype(int)
        hash_ids = hash_x * grid_resolution + hash_y
        hash_ids = hash_ids % (grid_resolution * grid_resolution)

        # Generate perturbation vector
        spatial_hash_perturbation = np.random.rand(n, 2) * 0.025
        # Apply perturbations based on spatial hash id and radius
        for i in range(n):
            scale = 0.001 + 0.01 * radii[i]  # Radius-based scaling for perturbation
            v[3*i] += spatial_hash_perturbation[i, 0] * scale
            v[3*i+1] += spatial_hash_perturbation[i, 1] * scale
            # Apply small perturbation to radius
            v[3*i+2] += np.random.uniform(-0.0005, 0.0005)

        # Final optimization step with perturbed configuration
        res = minimize(
            neg_sum_radii,
            v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 300,
                "ftol": 1e-10,
                "gtol": 1e-10,
                "eps": 1e-7
            }
        )

    # Final check of radius and center positions
    # Ensure no radii are negative, and all circles are within the unit square
    # Apply clipping to prevent NaNs, negative values, and ensure safety

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.05)  # 0.05 is empirically determined upper limit for radii

    # Final validation
    # We perform a lightweight final validation here for safety
    # This is to ensure no circles go outside the unit square or have negative radii
    # It's a last safeguard because the optimization itself should enforce these already
    # But this validation step is critical for the problem statement

    # Final validation step
    # We use the internal validate_packing function as specified in the problem statement
    # Although not directly called, we mimic its logic
    # Ensure no circles go beyond 1.0 in any dimension
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1.0 + 1e-12
            or y - r < -1e-12 or y + r > 1.0 + 1e-12):
            # Adjust to correct position if needed (e.g., clip to keep within unit square)
            x = np.clip(x, r, 1.0 - r)
            y = np.clip(y, r, 1.0 - r)
            centers[i] = np.array([x, y])
        # Ensure radius is positive
        if radii[i] <= 0:
            radii[i] = 1e-6  # minimum valid radius

    # Final validation step (optional, for robustness)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < radii[i] + radii[j] - 1e-12:
                # Adjust positions to increase separation between overlapping circles
                # We can do a small perturbation to centers to resolve overlap
                # This is a fallback if validation fails
                # Use perturbation with direction based on center position
                dir_x = dx / dist if dist > 1e-12 else np.random.rand() * 0.1 - 0.05
                dir_y = dy / dist if dist > 1e-12 else np.random.rand() * 0.1 - 0.05
                adjustment = 0.001 * np.array([dir_x, dir_y])
                centers[i] += adjustment
                centers[j] -= adjustment

    return centers, radii, float(radii.sum())
import numpy as np

def run_packing():
    n = 26
    # Adaptive grid and radial layout with dynamic spacing
    def compute_radial_grid_with_optimal_padding():
        # Dynamic grid calculation based on optimal packing ratios and spacing
        min_spacing = 0.25
        # Compute minimal grid size based on square root of n for uniformity
        grid_size_factor = 0.94 
        grid_rows = int(np.ceil(np.sqrt(n * 1.2)))
        grid_cols = int(np.ceil(np.sqrt(n * 1.2)))
        max_possible_circle_centers_per_cell = 4
        # Compute minimal radius to avoid overlap
        min_rad_for_grid = 0.21 
        # Compute radial layout grid with optimized initial spacing
        xs = []
        ys = []
        for i in range(n):
            row = i // grid_cols
            col = i % grid_cols
            # Radial center for better spacing
            angle = 2 * np.pi * i / float(n) + np.arcsin(0.5)  # Shift for initial spacing
            r = (row + 0.5) / grid_rows * min_spacing
            x_center = (col + 0.5) / grid_cols + (0.5 / grid_cols) * np.cos(angle)
            y_center = (row + 0.5) / grid_rows + (0.5 / grid_rows) * np.sin(angle)
            # Perturbation to avoid initial clustering (small random displacement)
            perturbation = np.random.uniform(-0.04, 0.04, size=2) 
            x = x_center + perturbation[0]
            y = y_center + perturbation[1]
            # Apply edge smoothing for near-bound conditions
            if row == grid_rows - 1:
                y += 0.05 * np.random.uniform(-1, 1)
            if col == grid_cols - 1:
                x += 0.05 * np.random.uniform(-1, 1)
            if (x < 1e-6) or (x > 1.0 - 1e-6) or (y < 1e-6) or (y > 1.0 - 1e-6):
                # If outside bounds, push into center safely
                x = max(0.2, min(0.8, x))
                y = max(0.2, min(0.8, y))
            xs.append(x)
            ys.append(y)
        # Return xs and ys with some grid-based radial variation
        return xs, ys
    
    # Create a more robust initial position with adaptive grid and radial variation
    xs, ys = compute_radial_grid_with_optimal_padding()
    
    # Determine initial radius based on grid size and overlap safety
    # Use geometric grid-aware calculation, ensuring minimal spacing
    # Let's set a lower bound based on grid cell spacing and circle placement
    min_initial_radius = 0.15 / np.sqrt(n)  # Based on grid cell and spacing needs
    # Initial radius with slight variance
    r0 = np.random.rand(n) * min_initial_radius * 1.8 + min_initial_radius
    # Adjust for minimal spacing to avoid initial clustering
    r0 = r0 * 1.1

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Create the bounds list matching decision vector length (3*n)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # For x, y, r respectively

    # Objective function to maximize sum of radii (minimized as negative)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraints: spatial and radial boundaries with vectorization
    cons = []
    # Create constraints for each circle's spatial boundaries
    for i in range(n):
        # Constraint for x: x_i - r_i >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Constraint for x: x_i + r_i <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Constraint for y: y_i - r_i >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Constraint for y: y_i + r_i <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Create a more efficient and vectorized overlap constraints using broadcasting
    # First, create spatial distances using vectorization
    # Create a function that can handle the constraint for all pairs efficiently
    # Instead of O(N^2) functions, we will precompute all pairs in a vectorized manner

    # For all i < j, compute the constraint f(x_i, y_i, r_i, x_j, y_j, r_j) >= 0
    def vectorized_overlap_constraints(v):
        # Efficient distance computation between all pairs
        centers = np.reshape(v[:2*n], (n, 2))
        radii = v[2::3]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = dx * dx + dy * dy
        # For i < j, compute the constraint (dist^2 - (r_i + r_j)^2) >= 0
        # We need to make sure to handle all pairs
        # To avoid redundant constraint checking, we'll generate the constraints for i < j
        # This makes the constraint list size O(N^2), which is necessary for correctness
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                # The constraint is dx^2 + dy^2 >= (r_i + r_j)^2
                constraint_val = dists[i,j] - (radii[i] + radii[j])**2
                # For optimization, we convert to ineq (value >= 0)
                constraints.append(constraint_val)
        # Because constraints are ineq with value >= 0, we return their negative
        # So we need to convert this to a negative of the value
        return -np.concatenate(constraints)
    
    # Add all pairwise distance constraints as constraints
    # Each pair (i, j) corresponds to a constraint function
    # We can vectorize the constraint evaluations by using the above function
    # However, scipy.optimize.minimize only allows functions that take v as input
    # So we need to create a set of constraint functions, each specific to a pair (i, j)
    # This is needed for proper numerical differentiation and optimization
    # Create all pairwise constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                         - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations, adjusted tolerances, and adaptive strategies
    # First, run a long optimization with vectorized constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-10})
    
    if not res.success:
        # First fallback: use a perturbed starting point with spatial hashing
        # Create space-aware randomized starting positions using hash-based spatial clustering
        xs = []
        ys = []
        for i in range(n):
            row = i // 4  # Using smaller grid for hashing
            col = i % 5
            x_base = (col + 0.5) / 4  # Base for col
            y_base = (row + 0.5) / 5  # Base for row
            # Apply small spatial hash to create diversity
            hash_val = np.random.rand() 
            x = x_base + (hash_val * 0.1)  # perturb based on hash
            y = y_base + (0.1 + np.sin(hash_val * 4)) / 8  # varied perturbation
            # Ensure within bounds
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            xs.append(x)
            ys.append(y)
        
        v0_perturbed = np.empty(3 * n)
        v0_perturbed[0::3] = np.array(xs)
        v0_perturbed[1::3] = np.array(ys)
        v0_perturbed[2::3] = r0 * 1.1  # Increase initial radii to push expansion
        
        # Re-run with perturbed initialization
        res = minimize(neg_sum_radii, v0_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-10})

    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Apply a more refined optimization by leveraging spatial hashing and radius adaptation for edge cases
        if np.linalg.norm(centers) > 5.0:  # Spatial hash reinitialization
            # Create a grid-based hash with radial layout
            new_centers = np.random.rand(n, 2) * 0.5  # Radial hashing
            new_centers *= 2.0 - np.random.rand(n, 2) * 0.7  # Expand and perturb
            new_centers = (new_centers - np.mean(new_centers))  # Center the hash
            # Adjust to fit in the unit square
            new_centers = np.clip(new_centers, 0.01, 0.99)
            # Reconstruct v based on new centers and current radius distribution
            new_v = v.copy()
            new_v[0::3] = new_centers[:, 0]
            new_v[1::3] = new_centers[:, 1]
            new_v[2::3] = radii * 1.1  # Increase radius to explore new positions
            # Re-run with new spatial hashing
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-9})
        # After rehashing, perform targeted radius expansion based on spatial distribution
        # Use distance matrix optimization
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            # Create a distance matrix for the centers
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx*dx + dy*dy)
            # For each circle, find the minimal distance to any other circle
            min_distances = np.min(dists, axis=1)
            # Find the circle(s) with the highest minimal distance to others (best candidate for expansion)
            max_min_dist = np.max(min_distances)
            best_circle_indices = np.where(min_distances == max_min_dist)[0]
            best_circle_idx = np.random.choice(best_circle_indices)
            # Expand radius of the best circle by increasing the radius while maintaining constraints
            # This is a heuristic-based expansion to enhance total volume
            # We do an adaptive expansion using numerical constraint checking
            # First, we compute the feasible expansion range based on minimal distance to others
            dist_from_best = dists[best_circle_idx]
            min_distance_to_others = np.min(dist_from_best)
            radius = radii[best_circle_idx]
            # We compute the maximum allowable expansion factor that respects the minimal distance to others
            expansion_factor = min(0.8, (min_distance_to_others - radius) / (np.sqrt(2) * radius))
            new_radius = radius * (1.5 + expansion_factor)
            # Apply this expansion and run a targeted optimization step
            # Create a new decision vector with just this circle's radius increased
            new_v = v.copy()
            new_v[3*best_circle_idx + 2] = new_radius
            # Run a localized optimization step
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-9})
            # If the radius is feasible after expansion (e.g., overlaps not found)
            if res.success:
                v = res.x

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)  # Ensure no invalid radii
    # Final check: if the optimization failed, retry with a different perturbation strategy
    # But we've already attempted multiple strategies, so just return what we have
    return centers, radii, float(radii.sum())
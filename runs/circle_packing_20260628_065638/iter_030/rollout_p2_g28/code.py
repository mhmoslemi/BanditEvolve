import numpy as np

def run_packing():
    n = 26

    # Phase 1: Optimal grid initialization with adaptive cell count based on proximity constraints
    # Adaptive grid approach: 5x5 grid with 1 row padding, + 4 extra columns for asymmetric packing
    cols = 5  # Base columns
    rows = (n + cols - 1) // cols

    # Initialize with structured grid + soft randomized disturbance
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base center positions with adaptive padding for spacing
        base_x = (col + 0.25) / (cols + 0.25)  # Adjusted for more space
        base_y = (row + 0.25) / (rows + 0.25)
        # Add stochastic perturbation, larger for edge circles to reduce cluster density
        # For the last row or column (i.e., high index or row/col at edges), add more randomness
        # This creates a "shattered" grid where edge circles are more flexible
        perturbation = 0
        if (row == rows - 1) or (col == cols - 1):
            perturbation = np.random.uniform(-0.15, 0.15)
        x = base_x + np.random.uniform(-0.075 + perturbation, 0.075 + perturbation)
        if (row % 2 == 0):  # Alternate row bias to create non-collinear staggering
            x += 0.2
        y = base_y + np.random.uniform(-0.075, 0.075)
        xs.append(x)
        ys.append(y)
    
    # Initial radius guess: optimize based on cell spacing and overlap avoidance
    # Base radius as a function of cell size and edge buffer
    # Start with radii that are 60% of the minimal cell size to allow expansion
    base_cell_width = 1.0 / cols
    base_cell_height = 1.0 / rows
    r0 = 0.35 * min(base_cell_width, base_cell_height) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    # Phase 2: Refinement with optimized constraint formulation and gradient handling
    # Using vectorized constraint formulation and optimized computation
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Phase 2.1: Precompute constraint indices for performance and readability
    
    # Optimized constraint functions with closure capture through lambda with explicit variables
    # Use direct indexing instead of parameter capture to avoid potential lambda binding issues
    # Also, precompute the 3*26 indices once
    indices = np.arange(3 * n)
    
    # Constraint functions for boundary (inequality constraints as per problem statement)
    # Using vector function for distance constraint between each pair
    # Use direct references to variables in function definitions
    
    # Define constraint functions for boundaries (inequality constraints)
    cons = []
    for i in range(n):
        idx = 3 * i
        # Left - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, idx: v[idx] - v[idx + 2]})
        # Right + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, idx: 1.0 - v[idx] - v[idx + 2]})
        # Bottom - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, idx: v[idx + 1] - v[idx + 2]})
        # Top + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, idx: 1.0 - v[idx + 1] - v[idx + 2]})
    
    # Phase 2.2: Optimize pairwise distance constraints
    # Vectorized pairwise distance with broadcasting as a precomputed matrix and constraint functions
    # To prevent the constraint list from exploding beyond memory, precompute only the first 300
    # pairs and use a more scalable approach by reusing constraint templates
    
    # Instead of generating all pairs, we create constraint functions and add them in a controlled way
    # This helps manage the constraint count and memory usage, which is critical for optimization
    for i in range(n):
        for j in range(i + 1, n):
            # Ensure we add only the first 300 pairs for efficiency, as adding more can slow the solver
            # We select first 300 pairs of (i,j) where i < j, prioritizing those with least mutual distance in the initial v0
            # We add the first 200 pairs here, others are deferred for final-phase perturbation optimization
            if not (i >= 50 or j >= 50):
                # Distance constraint: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
                def distance_cons_func(v, i, j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                cons.append({"type": "ineq", "fun": lambda v, i, j: distance_cons_func(v, i, j)})
    
    # Phase 3: Initial optimization with high precision settings
    # Optimized settings include more iterations, lower tolerances, and method selection
    
    # Initial run
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-9})
    
    # Phase 3.1: Check for success, if not, fall back to initial
    if not res.success:
        # Add some manual adjustments to the optimization parameters to escape a local minimum
        # Adjusted parameters: try L-BFGS-B with more adaptive bounds if not converged
        def lbfgs_neg(v):
            return -np.sum(v[2::3])
        
        # Try lbfgs-b with tighter bounds when SLSQP fails
        try:
            res = minimize(lbfgs_neg, v0, method='L-BFGS-B', bounds=bounds,
                           constraints=cons, options={"maxiter": 700, "ftol": 1e-10})
        except:
            res = minimize(lbfgs_neg, v0, method='Nelder-Mead', bounds=bounds,
                           constraints=cons, options={"maxiter": 1000})
    
    # Phase 4: Spatial constraint reconfiguration with adaptive radius expansion
    # We now isolate two most interacting circles, reconfigure their positions, and apply targeted radius expansion
    
    if res.success:
        v = res.x
        # Extract center positions and radii for analysis
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
    
        # Step 1: Calculate pairwise distances
        # Vectorized distance matrix with broadcasting for efficient computation
        dx = centers[np.newaxis, :, 0] - centers[:, np.newaxis, 0]
        dy = centers[np.newaxis, :, 1] - centers[:, np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Step 2: Identify the two most interacting circles
        interaction_matrix = np.sum(dists, axis=1)  # Sum of distances per circle
        top_pair_indices = np.argpartition(interaction_matrix, -2)[-2:]  # Get indices of top 2 circles
        # For robustness, ensure distinct unique pairs
        top_pair_indices = np.unique(top_pair_indices)
    
        # Step 3: Force reconfiguration of the two interacting circles
        # This involves spatial constraint perturbation and spatial dissection
        # Spatial constraint perturbation: random perturbation in both x and y
        # Spatial dissection: spatially displace to break up overlap
        # We will create a new configuration for the top pair and optimize again
        # To avoid overfitting, we will apply a randomized constraint displacement but maintain feasibility
    
        # Generate new perturbations for the top_pair_indices for spatial dissection
        random_displacement = np.random.uniform(-0.08, 0.08, size=(2, 2))  # (dx, dy)
        new_centers = centers.copy()
        for idx in top_pair_indices:
            new_centers[idx] += random_displacement[(idx - top_pair_indices[0]) % 2, :]
        
        # Update the centers for the optimized vector
        new_centers = np.clip(new_centers, [0.0, 0.0], [1.0, 1.0])
        # Recalculate the vector by placing new centers in the solution vector
        new_v = v.copy()
        new_v[0::3] = new_centers[:, 0]
        new_v[1::3] = new_centers[:, 1]
    
        # Re-optimize the configuration after dissection
        # To prevent overfitting, keep the radius constraints fixed for these two
        # We'll only reoptimize the positional variables for these top two circles
        # Keep radii same, but allow optimization of their positions only
        
        # Create a new constraint list where we freeze radii of top_pair_indices
        # We'll do this by applying constraints only to their positions
        new_cons = []  # Reset constraints for new v
        
        # Add boundary constraints again for all, but now with new_v
        for i in range(n):
            idx = 3 * i
            new_cons.append({"type": "ineq", "fun": lambda v, idx: v[idx] - v[idx + 2]})
            new_cons.append({"type": "ineq", "fun": lambda v, idx: 1.0 - v[idx] - v[idx + 2]})
            new_cons.append({"type": "ineq", "fun": lambda v, idx: v[idx + 1] - v[idx + 2]})
            new_cons.append({"type": "ineq", "fun": lambda v, idx: 1.0 - v[idx + 1] - v[idx + 2]})
        
        # Add the distance constraints again, but now the top two pairs are already perturbed
        for i in range(n):
            for j in range(i + 1, n):
                # We only add the distance constraints that involve the top_pair_indices now
                # This is a controlled approach that ensures we only reconfigure the interacting circles
                if i in top_pair_indices or j in top_pair_indices:
                    def distance_constr_func(v, i, j):
                        dx = v[3*i] - v[3*j]
                        dy = v[3*i+1] - v[3*j+1]
                        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                    new_cons.append({"type": "ineq", "fun": lambda v, i, j: distance_constr_func(v, i, j)})
        
        # Now, run optimization with constrained spatial reconfiguration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-10})
    
    # Phase 5: Targeted radius expansion on the least constrained circle
    # Re-evaluate based on new configuration and optimize in a way that increases radius
    # We identify the least constrained circle and expand its radius
    
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Recalculate distances matrix
        dx = centers[np.newaxis, :, 0] - centers[:, np.newaxis, 0]
        dy = centers[np.newaxis, :, 1] - centers[:, np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Step 6: Find the least constrained circle by maxmin distance
        # We want the circle with the maximum minimum pairwise distance to neighbors
        min_dist_per_circle = np.min(dists, axis=1)
        isole_idx = np.argmax(min_dist_per_circle)  # Most isolated
            
        # Step 7: Calculate and apply growth to the least constrained circle
        total_sum = np.sum(radii)
        expansion_amount = 0.02  # Target growth value
        target_total_sum = total_sum + expansion_amount
        # To avoid constraint violation, distribute the expansion evenly
        # But prioritize the isolated circle with extra
        expansion_per_circle = (target_total_sum - total_sum) / (n - 1)
        # Give more to the isolated circle to push the boundary
        isolation_factor = 2.0  # 1.2 x more growth
        expanded_radii = radii + expansion_per_circle
        expanded_radii[isole_idx] += expansion_per_circle * isolation_factor
        
        # Create expanded v vector where we apply new radii
        expanded_v = v.copy()
        expanded_v[2::3] = expanded_radii
        
        # Re-evaluate with new radii
        # Since we've already reconfigured the spatial positions, this is only about radii now
        # We keep the same centers, but apply the expanded radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-10})
    
    # Phase 6: Final check for success and fallback to initial v0 if needed
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    # Perform one final manual check after optimization
    # This ensures that the radii can be expanded if the solver returns early or underconstrained
    # This is also a safeguard for cases where the solver returns with a lower total sum due to convergence
    if np.sum(radii) < 1.3:  # Below 1.3 is below our typical optimal range, we perform additional expansion
        # Forced expansion phase: manually check if it's feasible
        # Find the least constrained by distance, and expand its radius by 0.01
        dx = centers[np.newaxis, :, 0] - centers[:, np.newaxis, 0]
        dy = centers[np.newaxis, :, 1] - centers[:, np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)

        min_dist_per_circle = np.min(dists, axis=1)
        isole_idx = np.argmax(min_dist_per_circle)
        max_radius_candidate = np.min([1, np.min(dists[isole_idx, :]) / 2])
        # Expand the radius to the maximum possible based on distance constraints
        max_possible_radius = np.amin([1.0 - centers[isole_idx, 0] - centers[isole_idx, 1],  # Top-right margin
                                      centers[isole_idx, 0] - 0,  # Left margin
                                      1.0 - centers[isole_idx, 1],  # Top margin
                                      1.0 - centers[isole_idx, 0],  # Right margin
                                      (np.min(dists[isole_idx, :]) / 2) - 4e-3])  # Max radius based on min neighbor distance
        
        # If the current radius is not maximum possible, expand
        if max_possible_radius > v[3*isole_idx+2]:
            new_radius = max_possible_radius
        else:
            new_radius = v[3*isole_idx+2]
        expanded_v = v.copy()
        expanded_v[3*isole_idx+2] = new_radius
        expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
        # Re-evaluate manually to ensure no overlap, and force the constraint
        valid = True
        for i in range(n):
            for j in range(i+1,n):
                dx = expanded_centers[i,0] - expanded_centers[j,0]
                dy = expanded_centers[i,1] - expanded_centers[j,1]
                if np.sqrt(dx*dx + dy*dy) < (expanded_v[3*i+2] + expanded_v[3*j+2]) - 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            v = expanded_v
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())
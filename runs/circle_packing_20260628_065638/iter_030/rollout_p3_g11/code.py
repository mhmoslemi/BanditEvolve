import numpy as np

def run_packing():
    n = 26
    cols = 5  # Fixed to 5 for grid-like balance, better for dense packing
    rows = (n + cols - 1) // cols
    num_circular_layers = 5  # To add spatial diversity by layering
    
    # Optimize the initial grid layout with spatial hashing, layered distribution and dynamic seeding
    def generate_smart_base_grid():
        xs = []
        ys = []
        seeds = np.random.rand(n, 2) * 0.1  # Introduce spatially unique seed offsets
        for i in range(n):
            layer = i // 5  # Layering by 5 for better radial expansion dynamics
            col = i % cols
            base_x = (col + 0.7) / cols * 1.2  # Slightly expanded x
            base_y = (layer + 0.3) / rows * 1.2  # Slightly expanded y and layered
            x = base_x + seeds[i, 0] * 0.1  # Small seed perturbations for diversity
            y = base_y + seeds[i, 1] * 0.1
            # Alternate row staggering - shift every even index row by a fraction
            if (layer % 2 == 0 and (col < cols - 1) and col >= 2):
                x += np.random.uniform(-0.1, 0.1)
            else:
                # For dense layers, apply subtle vertical staggering to avoid clumping
                y += np.random.uniform(-0.05, 0.05)
            xs.append(np.clip(x, 0, 1))
            ys.append(np.clip(y, 0, 1))
        return np.array(xs), np.array(ys)
    
    # Initial positions with enhanced seeding and density layering
    xs, ys = generate_smart_base_grid()
    r0 = 0.34 / cols  # Increased from initial to allow expansion
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)
    
    # Ensure that the bounds list matches the decision vector length
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries

    def neg_sum_radii(v):
        """Objective function to minimize negative of sum of radii"""
        return -np.sum(v[2::3])

    # Construct constraints more robustly: use captured i and dynamic constraint functions
    # Use lambda with fixed closure, avoiding common Python closure capture issues
    def make_bound_constraint(i):
        def constraint(v):
            return v[3*i] - v[3*i + 2]
        return constraint

    def make_side_bound_constraint(i):
        def constraint(v):
            return 1.0 - v[3*i] - v[3*i + 2]
        return constraint

    def make_vert_bound_constraint(i):
        def constraint(v):
            return v[3*i + 1] - v[3*i + 2]
        return constraint

    def make_top_bound_constraint(i):
        def constraint(v):
            return 1.0 - v[3*i + 1] - v[3*i + 2]
        return constraint

    # Vectorize the constraints for each circle
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": make_bound_constraint(i)})
        cons.append({"type": "ineq", "fun": make_side_bound_constraint(i)})
        cons.append({"type": "ineq", "fun": make_vert_bound_constraint(i)})
        cons.append({"type": "ineq", "fun": make_top_bound_constraint(i)})

    # Overlap constraints: vectorized and optimized with geometric hashing and early pruning
    # Precompute for each pair only if they are in close proximity
    # We add a spatial prefilter with Voronoi-like distance thresholds to avoid constraint explosion
    # We use a grid-based filter: only compute overlaps if their centroids are within 1.5 * mean_radius
    # The mean_radius is computed with some buffer to allow safe constraints
    # This is a smart approximation: avoids computing pairwise distances for all combinations
    # It reduces the number of overlap constraints significantly

    # First, compute the initial mean_radius for overlap filtering
    mean_radius_initial = np.mean(r0)

    # Create an optimized set of overlapping constraints with spatial prefilter
    # We precompute a spatial graph
    from scipy.spatial.distance import cdist
    # Initial setup for filtering
    overlap_constraints = []

    def get_overlap_constraint(i, j):
        # Function to compute squared distance minus sum of radius squared
        def constraint(v):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
        return constraint

    # Build a list of (i,j) pairs where the initial grid suggests overlap
    # We calculate the pairwise distances for the initial grid and precompute
    # this with cdist to avoid full O(n²) computation
    # Only create constraints for distances that are likely to be overlapping
    # We precompute the initial pairwise distance
    # We only trigger constraints for pairs within 1.5 times mean_radius
    # to avoid overloading the solver with redundant constraints

    # Precompute initial centers
    initial_centers = np.column_stack([xs, ys])
    initial_distances = cdist(initial_centers, initial_centers)
    for i in range(n):
        for j in range(i + 1, n):
            if initial_distances[i, j] < 1.5 * mean_radius_initial:
                # Only add the constraint if the initial distance is less than 1.5 * mean_radius
                # This is a heuristic to prevent excessive constraints
                # It's not perfect, but it's necessary to handle large n scenarios
                # Also, add a small buffer to the constraint to allow optimization without tight bound
                cons.append({"type": "ineq", "fun": get_overlap_constraint(i, j)})
    
    # First optimization phase with higher tolerance and more aggressive constraint handling
    # Add constraint tolerance parameter to handle numerical error
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3500, "ftol": 1e-11, 
                                             "acceptable_tol": 1e-11, 
                                             "eps": 1e-8,
                                             "disp": False})
    
    # Phase 2: dynamic perturbation and reconfiguration via stochastic spatial hashing with adaptive scaling
    # This phase is triggered only if initial optimization is successful
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        mean_radius = np.mean(radii)
        median_radius = np.median(radii)
        
        # Generate a dynamic spatial perturbation map with adaptive scaling based on radius
        # This introduces a spatially adaptive mutation to allow for localized optimization
        spatial_hash = np.random.rand(n, 2) * 0.06  # Small scale to avoid chaos
        perturbed_v = v.copy()
        for i in range(n):
            # Apply perturbation proportional to radius to allow larger movement for smaller radii
            # Also, ensure that perturbation is capped at 0.1 to avoid going out of bounds
            max_perturb = np.clip((v[3*i + 2] / mean_radius) * 0.06, 0.02, 0.1)  # Adaptive radius-based scaling
            perturbed_v[3*i] += spatial_hash[i, 0] * max_perturb
            perturbed_v[3*i+1] += spatial_hash[i, 1] * max_perturb

        # Re-evaluate with stochastic perturbation and same constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, 
                       options={"maxiter": 3500, "ftol": 1e-12, "eps": 1e-8, "disp": False})
        
        # Phase 3: Adaptive expansion on the least constrained circle with soft, gradient-informed expansion
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            # Optimized vectorized distance computation with broadcasting
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            # Use masked array to avoid zero-division in min distances (i.e., same circle distance)
            dists = np.ma.masked_equal(dists, 0)
            # Use broadcasting to compute minimum distance to all other circles
            min_dists = np.min(dists, axis=1)
            # Identify the least constrained (i.e., isolated) circle
            # Use a combination of isolation and spatial expansion potential
            # Also, introduce a soft isolation index with tolerance
            # We define a "weight" for isolation as 1 - (dists / max_dist).mean()
            # We compute isolation score for each circle
            max_dist = np.max(dists)
            isolation_weights = np.zeros(n)
            for i in range(n):
                # Isolation weight is a combination of min distance and spatial distribution
                dist_sum = np.sum(dists[i, :])
                avg_dist = dist_sum / (n-1)  # Average of distances to all but self
                isolation_weights[i] = (avg_dist / max_dist) * 0.8 + np.sum(min_dists) * 0.2  # Adjusted for dynamics
            # Find the index of the least constrained circle
            least_constrained_idx = np.argsort(isolation_weights)[::-1][-1]  
            # Introduce a targeted expansion plan with gradient-based expansion and dynamic radius constraints
            # We compute the total sum of radii
            current_sum = np.sum(radii)
            # We aim for an increase of 0.009 with dynamic growth, using radius-based expansion
            desired_growth = 0.009  # Slightly larger ambition
            expansion_vector = np.zeros_like(radii)
            # We distribute the expansion with a base of 0.005 and a dynamic scaling factor (radii mean + buffer)
            expansion_factor = 0.005 * (np.mean(radii) + 0.01)  # Ensure expansion is feasible with buffer
            # Allocate expansion to non-isolated circles but with a preference to the isolated one
            for i in range(n):
                if i != least_constrained_idx:
                    # Allocate expansion based on inverse of distance to the isolated point
                    isolation_distance = dists[i, least_constrained_idx]
                    weight = 1 / (isolation_distance + 1e-8) if isolation_distance > 0 else 0
                    # Normalize the weight
                    normalizer = np.sum(1 / (dists[i, :n != i] + 1e-8) if dists[i, :n != i] > 0 else 0)
                    # But for non-isolated, use a base allocation
                    expansion_vector[i] = expansion_factor * (1.0 + (1.0 / (normalizer + 1)))
                else:
                    expansion_vector[i] = expansion_factor * 1.3  # Give the isolated one more expansion
            # We create a new radii vector that is the original + expansion
            # To preserve feasibility and avoid constraint violations, we use a constraint-driven approach
            # We apply a greedy iterative expansion with constraint checking
            new_radii = radii + expansion_vector
            new_radii = np.clip(new_radii, 1e-6, 0.5)  # Keep within bounds
            
            # Apply this new radii to the decision vector
            # We must recheck all constraints to validate feasibility
            # To be efficient, we precompute distance matrix for this configuration
            new_centers = np.column_stack([v[0::3], v[1::3]])
            new_radii = v[2::3] + expansion_vector
            
            # We re-validate by recomputing distances and ensuring no overlap
            # This is a fallback mechanism to enforce validation if constraint optimization fails
            # We perform this check first to avoid entering invalid configurations
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                # We update the decision vector with new radii and re-apply optimization
                # This creates a new decision vector with expanded radii
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                # Now, optimize under the same constraints
                res_expanded = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                                       constraints=cons, 
                                       options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-8, "disp": False})
                
                # Final check for success
                if res_expanded.success:
                    res = res_expanded
                    v = res.x
                    # Final clip and validation for safety
                    radii = np.clip(v[2::3], 1e-6, 0.5)
                    # Final validation for safety
                    final_validation = True
                    for i in range(n):
                        for j in range(i + 1, n):
                            dx = v[3*i] - v[3*j]
                            dy = v[3*i+1] - v[3*j+1]
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < v[3*i+2] + v[3*j+2] - 1e-12:
                                final_validation = False
                                break
                        if not final_validation:
                            break
                    if not final_validation:
                        # Fallback: reduce expansion slightly
                        reduction = 0.3
                        for i in range(n):
                            if i != least_constrained_idx:
                                v[3*i + 2] -= (v[3*i + 2] - radii[i]) * reduction
                        # Re-run final validation
                        center_array = np.column_stack([v[0::3], v[1::3]])
                        for i in range(n):
                            for j in range(i + 1, n):
                                dx = center_array[i, 0] - center_array[j, 0]
                                dy = center_array[i, 1] - center_array[j, 1]
                                dist = np.sqrt(dx**2 + dy**2)
                                if dist < v[3*i+2] + v[3*j+2] - 1e-12:
                                    final_validation = False
                                    break
                            if not final_validation:
                                break
                        if not final_validation:
                            # Final fallback to initial configuration
                            v = res.x
                            radii = v[2::3]
                            centers = np.column_stack([v[0::3], v[1::3]])
                    else:
                        v = res.x
                        centers = np.column_stack([v[0::3], v[1::3]])
                        radii = np.clip(v[2::3], 1e-6, 0.5)
                else:
                    # Fallback to the most recently successful result
                    v = res.x
                    centers = np.column_stack([v[0::3], v[1::3]])
                    radii = np.clip(v[2::3], 1e-6, 0.5)
            else:
                # Fallback to the most recently successful result
                v = res.x
                centers = np.column_stack([v[0::3], v[1::3]])
                radii = np.clip(v[2::3], 1e-6, 0.5)
    else:
        # Use the initial v0 in case of initial failure
        v = v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.5)
    
    # Final validation check
    # This is a defensive layer to avoid invalid results
    for i in range(n):
        if radii[i] < 0:
            return None, None, 0.0
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            # If outside bounds, return initial state
            return np.column_stack([v0[0::3], v0[1::3]]), v0[2::3], float(v0[2::3].sum())
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            if np.sqrt(dx**2 + dy**2) < radii[i] + radii[j] - 1e-12:
                # If circle over-lap, return initial state
                return np.column_stack([v0[0::3], v0[1::3]]), v0[2::3], float(v0[2::3].sum())
    
    return centers, radii, float(radii.sum())
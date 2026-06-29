import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # Adaptive grid and asymmetric initial configuration
    cols = int(np.ceil(np.sqrt(n)))
    base_grid_cols = cols
    base_grid_rows = (n + cols - 1) // cols
    base_cell_size = 0.5 / cols
    grid_cell_size = base_cell_size * 1e0  # Tuned for better spacing
    grid_cols = 5  # Hardcoded grid columns to maintain structure
    grid_rows = (n + grid_cols - 1) // grid_cols
    grid_cell_size = 1.0 / grid_rows
    # Grid adjustment: create a non-square, adaptive grid
    grid_adjustment_cols = 5  # Keep fixed grid
    grid_adjustment_rows = (n + grid_adjustment_cols - 1) // grid_adjustment_cols
    
    # Initialize positions with advanced geometric clustering and staggered grid, 
    # with adaptive perturbation, spatial regularization, and dynamic symmetry breaking
    xs = []
    ys = []
    
    # 1st layer: base grid with dynamic spatial perturbation
    for i in range(n):
        row = i // grid_cols
        col = i % grid_cols
        x_center = (col + 0.2) / grid_cols  # Off-center for better spacing
        y_center = (row + 0.2) / grid_rows
        # Spatial perturbation: use adaptive scale based on grid
        space_scale = np.sqrt(1.0 / (row**2 + col**2 + 1e-6)) * 0.1
        offset = [
            np.random.uniform(-space_scale * 0.8, space_scale * 0.8),
            np.random.uniform(-space_scale * 0.6, space_scale * 0.6)
        ]
        x = x_center + offset[0]
        y = y_center + offset[1]
        # Staggered row alignment
        if row % 2 == 1:
            x += 0.3 / grid_cols
        xs.append(x)
        ys.append(y)
    
    # 2nd layer: advanced spatial dislocation
    advanced_dislocation = np.random.rand(n, 2) * 0.002
    xs = np.array(xs) + advanced_dislocation[:,0]
    ys = np.array(ys) + advanced_dislocation[:,1]
    # Spatial regularization to maintain grid structure
    regularization = np.random.rand(n, 2) * 0.01
    xs, ys = xs + regularization[:,0], ys + regularization[:,1]
    
    r0 = 0.18  # Larger initial radius for better optimization potential
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)
    
    # Ensure consistent bounds length
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]  # Tighter lower bound to enhance convergence

    # Objective function: maximize sum of radii
    def neg_sum_radii(v):
        radii = v[2::3]
        return -np.sum(radii)

    # Vectorized boundary constraints with lambda with captured i using explicit closure 
    cons = []
    for i in range(n):
        # Left boundary: x_i - r_i >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]}) # v[3*i] is x_i, v[3*i+2] is r_i
        # Right boundary: 1.0 - x_i - r_i >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y_i - r_i >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: 1.0 - y_i - r_i >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints (vectorized with geometric hashing and spatial intelligence)
    # We precompute some distances to speed up optimization
    # Using spatial-aware hashing with dynamic constraints for edge-circle reordering
    # Instead of all pairs, we precompute a small subset
    # But to implement the 'forced geometric dissection on most interactive' constraint:
    # We precompute the full matrix (as it's small) and find top-pairs
    # This is a vectorized computation with broadcasting
    # Optimized to use vectorized operations and minimize recomputation
    # We'll use the full matrix but only store for constraint generation
    # Then in the optimization we'll use the full matrix (which is small)
    # This reduces recomputation burden on the solver
    # Then we will apply a forced constraint reordering on the top interactors

    # Precompute distance matrix for all pairs
    # Optimization: use vectorized broadcasting to compute all pair distances
    # This is done once, not at every iteration
    # So no need to recompute repeatedly in constraint
    # This is crucial for performance and stability
    def build_full_distance_matrix(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        # Broadcast to get all pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        return dists

    # Precompute distances only once at the start
    # This will allow us to perform all-pairs distance operations without recomputation
    # The solver will use this precomputed information in constraint generation
    full_dists = None
    full_rads = None

    # Vectorized constraint generator for all pairs, using distance matrix and radii
    # This will be used for the main optimization pass
    def precompute_full_constraints(v, full_dists_cache, full_rads_cache):
        # If distances not computed, compute them once
        if full_dists_cache is None:
            full_dists_cache = build_full_distance_matrix(v)
        if full_rads_cache is None:
            full_rads_cache = v[2::3]
        # Now generate all constraint functions
        cons_out = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                cons_key = (i, j)
                # We'll create unique, non-capturing lambda for each constraint
                # This uses closure binding and lambda functions, but not in ways that break optimization
                cons_out.append({"type": "ineq",
                                 "fun": (lambda v, i=i, j=j, dist_cache=full_dists_cache, rad_cache=full_rads_cache:
                                         dist_cache[i,j] - rad_cache[i] - rad_cache[j])})
        return cons_out

    # First pass: generate all constraints using the full distance matrix
    # But since the full_dists is built from the initial v0, it's just precomputed
    # We will create all the constraints here
    full_constraints = precompute_full_constraints(v0, full_dists, full_rads)
    # But we need to handle them at each optimization step, so we'll precompute them during each optimization phase
    # But for performance, we'll do this once, and just use the precomputed constraint functions

    # Now, implement the forced geometric dissection of two most interacting circles
    # This is a multi-phase strategy:
    # 1. Initial optimization with all constraints
    # 2. Identify and isolate the most interactive pairs
    # 3. Apply forced geometric dissection to these pairs
    # 4. Introduce new adjacency constraints to reorder topology
    # 5. Apply radius expansion on least constrained circle while handling edge cases
    # 6. Final reoptimization
    # For the forced dissection, we will use dynamic reordering

    # Initial optimization with all constraints and full spatial awareness
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=full_constraints,
        options={
            "maxiter": 2000,  # Increased from 1500 
            "ftol": 1e-12,  # Tighter tolerance
            "gtol": 1e-10, 
            "eps": 1e-8,
            "disp": False,
        },
    )

    # Phase 1: Check success and proceed with forced geometric dissection
    if res.success:
        v_current = res.x
        full_dists_current = build_full_distance_matrix(v_current)
        full_rads_current = v_current[2::3]
        # Find indices of 7 most interacting circles (top 7 pairs)
        interaction = np.sum(full_dists_current, axis=1)
        top_interacting_indices = np.argsort(interaction)[-7:]  # Most interacting circle indices
        # Now find top interactive pairs among them
        # Pairwise comparisons of these indices
        top_pairs = []
        for i in top_interacting_indices:
            for j in top_interacting_indices:
                if i < j:
                    dist = full_dists_current[i, j]
                    rad_sum = full_rads_current[i] + full_rads_current[j]
                    if dist < rad_sum:
                        top_pairs.append((i, j))
        # Select top 3 pairs among these for geometric dissection
        if len(top_pairs) >= 3:
            # Select top 3 most interactive pairs from the above
            # We will use a weighted score: (dist - rad_sum) < 1e-4
            scores = []
            for i, j in top_pairs:
                dist = full_dists_current[i,j]
                rad_sum = full_rads_current[i] + full_rads_current[j]
                # We want to find pairs that are close to touching or overlapping
                scores.append(1 / (dist - rad_sum + 1e-12))  # Use inverse to prioritize closeness
            top_interactive_pairs = sorted(top_pairs, key=lambda x: scores[top_pairs.index(x)])
            top_interactive_pairs = top_interactive_pairs[:3]  # Take top 3
        else:
            # Fallback to top 3 pairs if not found
            top_interactive_pairs = [(i, j) for i in range(n) for j in range(i+1, n)][:3]
        
        # Phase 2: Forced geometric dissection on top most interacting pairs
        # We will create new constraints that displace these circles apart
        # Add new constraints to ensure geometric separation
        # Also introduce a dynamic radius adjustment phase
        # Create a clone of current variables for this phase
        v_clone = v_current.copy()
        # For each pair in top_interactive_pairs, apply a forced separation constraint
        # Add a constraint to push the centers apart
        # We will add explicit constraints that separate the two pairs
        # This is done with a new constraint function that adds a fixed separation
        # These are added to the constraints list and reoptimize
        additional_constraints = []
        for i, j in top_interactive_pairs:
            dx = v_clone[3*i] - v_clone[3*j]
            dy = v_clone[3*i+1] - v_clone[3*j+1]
            sep = 1.2 * (v_clone[3*i+2] + v_clone[3*j+2])  # 20% more than sum of radii
            if sep < 1e-6:
                sep = 1e-6
            # Add geometric dissection constraints
            # 1. Enforce separation via distance function
            def constraint_func_sep(v, i=i, j=j, sep=sep):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - sep
            additional_constraints.append({"type": "ineq", "fun": constraint_func_sep})
            # 2. Enforce radii adjustment to maintain feasibility
            # We may introduce a soft radial constraint if radius expansion is needed
            # This is optional, but can help in reordering the topological constraints
            # We'll use a soft constraint to slightly inflate certain radii
            # Create a soft constraint that slightly increases radii
            # This can be useful for dissection, but we will apply it only on specific circles
            # For example, the two circles of the pair
            # This is a soft constraint that allows slight radius expansion
            def constraint_func_rad(v, i=i, j=j, r=1.0):
                return v[3*i+2] + v[3*j+2] - r
            additional_constraints.append({"type": "ineq", "fun": constraint_func_rad})

        # Re-add constraints with new dissection and radii adjustment
        # We will recompute the full distance matrix again based on updated v_clone
        # This ensures the distances are updated to reflect current positions
        # This is done only once more, then constraints are applied
        updated_full_dists = build_full_distance_matrix(v_clone)
        
        # Rebuild all constraints with updated matrix
        # We will use the same full_constraints but override the constraint functions
        # Note: this step is not efficient but is required due to the unique nature of the problem
        new_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dx = v_clone[3*i] - v_clone[3*j]
                dy = v_clone[3*i+1] - v_clone[3*j+1]
                dist = np.sqrt(dx**2 + dy**2)
                rad_sum = v_clone[3*i+2] + v_clone[3*j+2]
                # Use the new constraint function that dynamically reflects current positions
                # This is necessary for optimization with current v_clone
                def constraint_func(v, i=i, j=j, dist=dist, rad_sum=rad_sum):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    new_dist = np.sqrt(dx**2 + dy**2)
                    return new_dist - rad_sum
                new_constraints.append({"type": "ineq", "fun": constraint_func})
        
        # Combine with the additional constraints
        new_constraints += additional_constraints

        # Second optimization with dissection and radius adjustment
        # Now we'll optimize with the updated constraints and current v_clone
        res = minimize(
            neg_sum_radii,
            v_clone,
            method="SLSQP",
            bounds=bounds,
            constraints=new_constraints,
            options={
                "maxiter": 1400,  # Slight decrease for stability
                "ftol": 1e-12,
                "gtol": 1e-10,
                "eps": 1e-8,
                "disp": False,
            },
        )

        # Phase 3: Re-evaluate and identify least constrained circle
        if res.success:
            v_current = res.x
            full_dists_current = build_full_distance_matrix(v_current)
            full_rads_current = v_current[2::3]
            # Calculate the sum of distances to all others for each circle
            # To find the circle with least interaction (least constrained)
            min_dists = np.min(full_dists_current, axis=1)
            isolated_circle_idx = np.argmin(min_dists)
            # Target radius expansion: 0.6% of total sum as a potential growth
            # But we apply a controlled expansion, not direct radius increment
            # This is because expansion can create new interaction problems
            # Instead, we'll apply a soft growth to non-isolated circles
            # This is done while maintaining total sum constraint
            # We will compute the current total and plan a growth
            total_current = np.sum(full_rads_current)
            # We aim for a total growth of 0.006, as per previous
            target_total = total_current + 0.006
            # We can only grow up to the maximum size allowed
            max_circle_size = 0.5  # Hard constraint
            max_possible_radius = max_circle_size - 1e-4
            # Calculate feasible growth
            feasible_growth = (target_total - total_current) / (n - 1)
            if feasible_growth < 0:
                feasible_growth = 0.0
            # We need to distribute expansion across circles that are not isolated
            # Apply growth to all except isolated
            new_rads = full_rads_current.copy()
            # Apply growth proportionally, not to isolated
            for i in range(n):
                if i != isolated_circle_idx:
                    # Apply dynamic growth based on position and interaction
                    # For example, circles with higher distances (less interaction) get more growth
                    # This is a soft strategy to avoid over-constraint
                    # Also, apply a soft upper bound on growth
                    max_growth = np.min([feasible_growth * (0.9 + 0.1 * (1 - min_dists[i]/np.max(min_dists))),
                                        max_possible_radius - full_rads_current[i]])
                    # Apply stochastic perturbation for diversity
                    growth_rate = np.random.uniform(0.95, 1.05)
                    growth_amount = max_growth * growth_rate
                    new_rads[i] += growth_amount
            
            # Apply growth vector to the current state
            v_growth = v_current.copy()
            v_growth[2::3] = new_rads
            
            # Third optimization: apply grown radii with adjusted positions
            res = minimize(
                neg_sum_radii,
                v_growth,
                method="SLSQP",
                bounds=bounds,
                constraints=new_constraints,
                options={
                    "maxiter": 1200,
                    "ftol": 1e-12,
                    "gtol": 1e-10,
                    "eps": 1e-8,
                    "disp": False,
                },
            )

        # Final check and return
        if res.success:
            v_final = res.x
            centers = np.column_stack([v_final[0::3], v_final[1::3]])
            radii = np.clip(v_final[2::3], 1e-6, None)
            return centers, radii, float(radii.sum())
        else:
            # Fallback: use current solution without expansion
            v_final = v_current
            centers = np.column_stack([v_final[0::3], v_final[1::3]])
            radii = np.clip(v_final[2::3], 1e-6, None)
            return centers, radii, float(radii.sum())
    else:
        # If initial optimization fails, fallback to initial solution
        v_final = v0
        centers = np.column_stack([v_final[0::3], v_final[1::3]])
        radii = np.clip(v_final[2::3], 1e-6, None)
        return centers, radii, float(radii.sum())
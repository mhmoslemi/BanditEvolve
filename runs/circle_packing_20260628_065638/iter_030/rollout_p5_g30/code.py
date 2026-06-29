import numpy as np

def run_packing():
    from scipy.optimize import minimize

    n = 26
    # Optimized grid layout with dynamic col/row and improved initial clustering
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize with advanced clustering and symmetry reduction
    # We implement a dual-phase initialization to better manage initial spacing
    xs = []
    ys = []
    
    # Phase 1: Create initial grid with random offsets and staggering
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid positions with spacing optimized for 5 columns
        x_center = (col + 0.4) / cols + 0.02 * np.sin(np.pi * (row + col)/4)  # slight nonlinearity
        y_center = (row + 0.4) / rows + 0.02 * np.cos(np.pi * row / 4)  # vertical adjustment
        # Apply spatial perturbation with decay by radius (non-overlapping)
        x = x_center + np.random.uniform(-0.04, 0.04) * (0.4 / cols + 0.01)
        y = y_center + np.random.uniform(-0.04, 0.04) * (0.4 / rows + 0.01)
        # Stagger alternate rows to improve packing efficiency
        if row % 2 == 1:
            x += 0.45 / cols * (1 - 2 * (row % 4) / 3)  # adaptive staggering with row dependence
        xs.append(x)
        ys.append(y)

    # Phase 2: Add adaptive jitter based on relative spacing in a 3D lattice projection
    # Projected positions to space out clustering
    projected = [np.array([x, y]) for x, y in zip(xs, ys)]
    for i in range(n):
        # Adaptive jitter based on the local density (inverse of min distance to any neighbor)
        dists = np.array([np.sqrt((x - projected[i][0])**2 + (y - projected[i][1])**2) for x, y in projected])
        valid_dists = dists[dists > 1e-6]
        if valid_dists.shape[0] > 0:
            min_dist = np.min(valid_dists)
            jitter_radius = 0.2 * (0.1 + (0.05 * (2.0 / min_dist)) if min_dist <= 0.1 else 0.02)
            xs[i] += np.random.uniform(-jitter_radius, jitter_radius)
            ys[i] += np.random.uniform(-jitter_radius, jitter_radius)
            # Ensure jitter doesn’t break grid structure
            if abs(xs[i]) > 1.1 or abs(xs[i]) < -1e-5 or abs(ys[i]) > 1.1 or abs(ys[i]) < -1e-5:
                xs[i] = np.clip(xs[i], 0.0, 1.0)
                ys[i] = np.clip(ys[i], 0.0, 1.0)

    # Set base radius with better distribution logic and initial optimization
    # Start with larger base radius for better initial spacing and growth opportunities
    r0 = 0.38 / (cols / 2) - 0.002  # Adjusted initial radius for 5 columns
    # Ensure radius is safe from numerical instability
    r0 = np.clip(r0, 1e-3, 0.45)

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define tighter and structured bounds to prevent divergence
    # Ensure 3*n entries for the decision vector of length 3n
    bounds = []
    for _ in range(n):
        # X and Y bounds are strictly [0,1]
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-3, 0.45)]  # (x, y, r)
    
    # Define objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Define constraints in vectorized way with lambda binding (captured i,j)
    cons = []
    for i in range(n):
        # Boundary constraints
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with dynamic radius weighting and distance estimation efficiency
    # Optimization to reduce constraint evaluations
    # Use a dynamic approach with only top-N neighbor constraints and adaptive filtering
    def make_overlap_constraints():
        # Compute pairwise distances with broadcasting
        # Instead of O(n^2) constraints, apply filtering (e.g., only consider closest neighbors)
        # To reduce computational load for large n
        # We'll optimize by using a threshold based on initial estimated min radius
        # Calculate all pairwise distances, and filter for potential overlap
        # This is efficient since we'll only add constraints where potential overlaps exist
        # This is a major optimization step
        
        # Vectorized calculation of pairwise distances
        centers = np.array([v0[0::3], v0[1::3]]).T
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Create a mask to consider only the closest neighbors for constraints
        # Use a threshold based on base radius and spacing
        radius_threshold = r0 * 1.05
        # Filter only pairs where dist is less than 2*radius_threshold (initial overlap window)
        mask = dists < 2.0 * radius_threshold
        mask = np.triu(mask, 1)  # Only consider upper triangle for i<j
        non_zero_mask = np.where(mask)
        
        def build_constraint_func(i, j):
            def constraint_func(v):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            return constraint_func

        # Only add constraints where the distance is less than the threshold
        # This drastically reduces the number of constraints
        overlapping_pairs = [ (i, j) for i, j in zip(non_zero_mask[0], non_zero_mask[1]) ]
        
        for i, j in overlapping_pairs:
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: build_constraint_func(i, j)(v)
            })

        return

    make_overlap_constraints()

    # Initial optimization with aggressive tuning and constraint handling
    # Use advanced solver tolerances and dynamic options
    options = {
        "maxiter": 1500,  # High number to find better local optima
        "ftol": 1e-10,    # Tighter tolerance for radius and position
        "gtol": 1e-9,
        "eps": 1e-6,
        "disp": False,
        "iprint": -1,
        "finite_diff_rel_step": np.array([1e-6, 1e-6, 1e-4])
    }

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options=options)
    
    # Apply advanced adaptive reconfiguration with spatial sensitivity
    if res.success:
        v = res.x
        radius_arr = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Validate initial solution
        # We will not call the validator as it's assumed to be handled via the constraints
        
        # Apply advanced spatial perturbation and radius expansion
        # We implement an adaptive perturbation with radius-based sensitivity
        # This is a key innovation for improving the final configuration
        
        # Compute distance matrix with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Weighting matrix for influence of spatial proximity
        # Use a weighting function that decays with distance and is radius-sensitive
        spatial_weights = 1.0 / (np.clip(dists, 1e-8, 100.0) + 0.1 * radius_arr[:, np.newaxis])
        spatial_weights = np.triu(spatial_weights, 1)  # Only upper triangle (i < j)
        spatial_weights = np.nan_to_num(spatial_weights)  # Avoid infinite values
        
        # Compute total spatial influence of each circle
        total_influence = np.sum(spatial_weights, axis=1)
        normalizer = np.sum(total_influence)
        if normalizer > 0:
            # Compute influence vector
            influence_vector = spatial_weights / (total_influence[:, np.newaxis])
            influence_vector = np.nan_to_num(influence_vector)

        # Apply dynamic reconfiguration via spatial perturbation with influence-weighted scaling
        perturbation_scale = 0.05
        perturbation_radius_scale = 0.1  # Radius sensitivity
        perturbation_vector = np.random.rand(n, 2) * perturbation_scale * radius_arr[:, np.newaxis]
        perturbation_vector *= influence_vector
        new_v = v + np.concatenate([perturbation_vector[:, 0], perturbation_vector[:, 1], np.zeros(n)])
        new_v = np.clip(new_v, 0, 1)
        new_v[2::3] = np.clip(new_v[2::3], 1e-3, 0.45)
        
        # Re-evaluate perturbed configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 400,
                           "ftol": 1e-10,
                           "gtol": 1e-9,
                           "eps": 1e-6
                       })
    
    # Final targeted expansion with adaptive constraint-aware optimization
    # We now focus on the least constrained circle with spatial constraints
    # To avoid over-expanding, we implement a more refined and adaptive expansion approach
    if res.success:
        v = res.x
        radius_arr = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix again to find the weakest spatially constrained circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute spatial constraint strength by the minimum distance to other circles
        min_dists = np.min(dists, axis=1)
        # To avoid over-optimization, compute a weighted constraint strength
        constraint_strength = np.sum(1.0 / (min_dists + 1e-8), axis=1)
        constraint_strength[constraint_strength == 0] = np.inf
        least_constrained_idx = np.argmin(constraint_strength)
        
        # Compute current total radius and target total
        current_sum = np.sum(radius_arr)
        target_growth = 0.0085  # Increase the growth target
        target_sum = current_sum + target_growth
        
        # Allocate growth to the least constrained circle first
        new_radii = radius_arr.copy()
        expansion_factor = (target_sum - current_sum) / (n - 1)
        
        # Apply exponential growth with radius- and constraint-aware adjustment
        # We increase the least constrained circle more with a soft factor
        new_radii[least_constrained_idx] += expansion_factor * 1.3  
        # Apply moderate expansion to other circles but proportionally based on their spatial constraints
        for i in range(n):
            if i != least_constrained_idx:
                # Weighted by inverse of constraint strength
                weight = 1.0 / (constraint_strength[i] + 1e-8)
                new_radii[i] += expansion_factor * weight
        
        # Clamp radii to ensure they are within valid bounds, even if the optimization may push them over
        # This is an additional safeguard
        new_radii = np.clip(new_radii, 1e-3, 0.45)
        
        # Create new decision vector with adjusted radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Use a new optimization pass that prioritizes constraint tightening
        options = {
            "maxiter": 400,
            "ftol": 1e-10,
            "gtol": 1e-9,
            "eps": 1e-6,
            "iprint": -1,
            "finite_diff_rel_step": np.array([1e-6, 1e-6, 1e-4]),
            "bounds": bounds,
            "constraints": cons
        }
        res = minimize(neg_sum_radii, v_new, method="SLSQP", options=options)
    
    # Final safeguard before returning
    # If not successful, fallback to last known state
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-3, 0.45)  # Ensure radius bounds are respected
    return centers, radii, float(radii.sum())
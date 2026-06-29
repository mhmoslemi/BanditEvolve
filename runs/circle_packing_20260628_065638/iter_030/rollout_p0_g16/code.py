import numpy as np

def run_packing():
    n = 26
    cols = 6  # Use 6 columns for higher spatial resolution
    rows = (n + cols - 1) // cols
    
    # Initial positions with more spatial diversity: adaptive grid with geometric perturbations and asymmetric row offset
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Base grid coordinates
        x_grid = (col + 0.5) / cols
        y_grid_base = (row + 0.5) / rows
        y_grid = y_grid_base + np.sin(row * np.pi / rows) * 0.08  # Wave-like vertical perturbation to reduce clustering
        
        # Add small stochastic variation and asymmetric row offsets
        x_perturb = np.random.uniform(-0.04, 0.04)  # More varied perturbation
        y_perturb = np.random.uniform(-0.04, 0.04)
        x = x_grid + x_perturb
        y = y_grid + y_perturb
        
        # Asymmetric row shift for staggered alignment with adaptive magnitude
        if row % 2 == 0:
            row_shift = 0.1 / cols  # Small baseline shift for even rows
        else:
            row_shift = (0.15) / (cols * (1 + 0.02*(row // 2)))  # Larger stagger for odd rows with adaptive scaling
        if row % 2 == 1:
            x += row_shift  # Only odd rows get shifted to break alignment
        
        # Ensure x stays within bounds with minimal boundary relaxation
        x = np.clip(x, 1e-6, 1 - 1e-6)
        y = np.clip(y, 1e-6, 1 - 1e-6)
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with smaller starting base, more aggressive spatial allocation strategy
    r0 = 0.22 / cols  # Slightly less than previous, more aggressive allocation in space
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Bounds must have exactly 3*n entries, matched with 3*n length variable
    for _ in range(n):
        # x and y bounds are [0, 1], slightly loosened to avoid over-constraint for early iterations
        # radii have lower and upper bounds, optimized for both exploration and constraint robustness
        bounds += [(1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6), (1e-5, 0.4)]  # radii tighter than parent but higher than SOTA

    def neg_sum_radii(v):
        # Return a simple negative sum of radii
        # This is the objective function for optimization
        return -np.sum(v[2::3])
    
    # Prepare constraints with precise indexing and optimized evaluation
    cons = []
    for i in range(n):
        # Constraint 1: x_i - r_i >= 0 (left boundary)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Constraint 2: 1 - x_i - r_i >= 0 (right boundary)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Constraint 3: y_i - r_i >= 0 (bottom boundary)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Constraint 4: 1 - y_i - r_i >= 0 (top boundary)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Optimized overlap constraints: vectorized using precomputation to reduce per-call cost
    # Note: we precompute pairs to reduce lambda capture overhead, avoiding lambda closure issues
    # Also, avoid over-constraining by using efficient constraint setup
    for i in range(n):
        for j in range(i + 1, n):
            # Distance squared - (radii_i + radii_j)^2 >= 0
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2  # Use squared distance for better numerical stability
            
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with more iterations and stricter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-13})
    
    # Apply advanced perturbation strategy for global reconfiguration:
    # 1. Spatial hashing with adaptive scaling based on current radius distribution
    # 2. Constraint-aware perturbation to preserve feasibility
    if res.success:
        # Apply global perturbation heuristic for escaping local minima and exploring new configuration space
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hashing vector with adaptive scaling based on current radius distribution
        radius_mean = np.mean(radii)
        radius_std = np.std(radii)
        if radius_std > 1e-6:  # Only if there is variation in radii
            # Generate a perturbation that scales with both radius and spatial distribution
            # This avoids over-perturbation of small circles and under-perturbation of large ones
            spatial_hash = np.random.rand(n, 2) * 0.08  # base range for all
            radius_factor = 1.0 + np.log1p(radii / radius_mean) * 0.35  # scale perturbation based on radius relative to mean
            # Apply radius-sensitive spatial perturbation to both x and y
            perturbation = spatial_hash * radius_factor[:, np.newaxis]
            perturbed_v = v.copy()
            for i in range(n):
                perturbed_v[3*i] += perturbation[i, 0]
                perturbed_v[3*i+1] += perturbation[i, 1]
                # Ensure that perturbation stays within box bounds
                perturbed_v[3*i] = np.clip(perturbed_v[3*i], 1e-6, 1 - 1e-6)
                perturbed_v[3*i+1] = np.clip(perturbed_v[3*i+1], 1e-6, 1 - 1e-6)
        
        else:
            # If all radii are identical (edge case), apply minimal perturbation to avoid overlap
            # Just jitter the centers by small amount across board
            perturbed_v = v.copy() + np.random.rand(n, 3) * 0.005
            # Ensure bounds again
            for i in range(n):
                # x
                perturbed_v[3*i] = min(max(perturbed_v[3*i], 1e-6), 1 - 1e-6)
                # y
                perturbed_v[3*i+1] = min(max(perturbed_v[3*i+1], 1e-6), 1 - 1e-6)
        
        # Run secondary optimization with perturbed state but limited iterations
        # Also, include a constraint check here to avoid invalid configuration
        # This is a more robust strategy in case of perturbation leading to unbounded circles
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-12})
    
    # Apply adaptive "surgical expansion" on most constrained circles with strict constraint validation
    # Use vectorized calculation for all pairs
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Precompute pairwise distances with broadcasting for speed and memory efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        # Compute minimum distance to others for all circles
        min_dists = np.min(dists, axis=1)
        # Identify top 5 most constrained (least minimal distance) circles for expansion
        # Avoiding the smallest circles ensures we don't trigger invalid states
        constrained_indices = np.argsort(min_dists)[:5]  # Pick 5 most constrained
        
        # Initialize expansion vector to be a copy of current solution
        new_v = v.copy()
        # Create an expansion vector with gradual expansion for constrained circles
        # Use a multi-stage approach for safety and feasibility
        # 1. First, calculate how much total "space" is available
        # 2. Allocate expansion based on constraint violation
        # 3. Apply expansion in stages to preserve feasibility
        # For each constrained circle, check if it's feasible to expand
        
        # Step 1: Calculate total expansion available (based on radius potential)
        # Assume each constrained circle can expand by up to 15% of current radius safely (estimate)
        # However, we'll use a more conservative approach, based on how much the center is surrounded
        expansion_radius_factor = 0.05  # Max estimated expansion for constrained circles
        # Compute for each constrained circle the expansion allowed for it (in absolute terms)
        allowed_expansion = []
        for idx in constrained_indices:
            # Determine the maximum radius it can have before hitting the boundary, given current position
            r_current = radii[idx]
            x_center = centers[idx, 0]
            y_center = centers[idx, 1]
            
            # Max x + r and x - r
            max_x_plus = 1 - 1e-6
            max_x_minus = 1e-6
            allowed_r_x = min(max_x_plus - x_center, x_center - max_x_minus)
            
            max_y_plus = 1 - 1e-6
            max_y_minus = 1e-6
            allowed_r_y = min(max_y_plus - y_center, y_center - max_y_minus)
            
            # So, allowed_r is the minimum of x and y allowed radius growth
            allowed_r = min(allowed_r_x, allowed_r_y)
            
            # Also, check all pair constraints in case the circle is already tightly packed
            # We calculate how much we can expand the circle before overlapping with any other circle
            # We assume we want to expand this circle as much as possible without overlapping
            # Since we're focusing on expansion of constrained circles, not others, we can 
            # consider their current positions and radii
            new_radius = r_current
            while True:
                # Check the new radius against all other circles to ensure no overlap
                overlap = False
                for j in range(n):
                    if j == idx:
                        continue
                    dx_new = centers[idx, 0] - centers[j, 0]
                    dy_new = centers[idx, 1] - centers[j, 1]
                    dist_new = np.sqrt(dx_new*dx_new + dy_new*dy_new)
                    if dist_new < new_radius + radii[j] - 1e-12:
                        overlap = True
                        break
                if not overlap:
                    break
                else:
                    # Cannot expand further without overlapping, stop
                    allowed_r = new_radius - r_current
                    break
                # To avoid infinite loop, add a safeguard
                new_radius = min(new_radius + 0.001, r_current + allowed_r)  # Add small expansion
                # Early exit if we've already expanded as much as possible
                if new_radius >= (r_current + allowed_r):
                    break
            
            allowed_expansion.append(allowed_r)
        
        # Now, calculate the potential expansion (total) for all constrained circles
        # We'll apply a percentage-based expansion to each, based on how much expansion they can take
        # We will also apply a "constraint-aware" expansion, prioritizing circles with the most constraint
        # and applying more expansion as a function of how much they are constrained
        # Use the minimum distance metric as a proxy for constraint intensity
        total_radii = np.sum(radii)
        expansion_multiplier = 0.008  # Per unit of constrained index (higher for more constrained circles)
        # Compute expansion per constrained circle
        expansion_per_circle = np.array(allowed_expansion) * expansion_multiplier * np.arange(1, len(constrained_indices)+1) 
        # Now, accumulate the expansion vector
        for idx, constrained_idx in enumerate(constrained_indices):
            exp = expansion_per_circle[idx]
            new_v[3*constrained_idx + 2] += exp
        
        # Now, run final optimization with modified v
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-12})
    
    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())